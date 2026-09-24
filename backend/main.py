import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

app = FastAPI(title="FraudLens API")

models_cache = {}
explainers_cache = {}
scalers_cache = {}
metrics_cache = {}
data_cache = {"df": None, "X_train": None, "X_test": None, "y_train": None, "y_test": None, "X_val": None, "y_val": None}

def load_data(filepath="creditcard.csv"):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        data_cache["df"] = df
        
        y = df["Class"]
        X = df.drop(["Class", "Time", "Amount"], axis=1, errors='ignore')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=8)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17, random_state=8)
        
        data_cache["X_train"] = X_train
        data_cache["X_test"] = X_test
        data_cache["y_train"] = y_train
        data_cache["y_test"] = y_test
        data_cache["X_val"] = X_val
        data_cache["y_val"] = y_val
        return True
    return False

@app.on_event("startup")
def startup_event():
    if os.path.exists("creditcard.csv"):
        load_data("creditcard.csv")
    elif os.path.exists("../creditcard.csv"):
        load_data("../creditcard.csv")

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * 0.75)),
            nn.ReLU(),
            nn.Linear(int(input_dim * 0.75), int(input_dim * 0.5)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(input_dim * 0.5), int(input_dim * 0.75)),
            nn.ReLU(),
            nn.Linear(int(input_dim * 0.75), input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class TrainRequest(BaseModel):
    model_name: str
    features: List[str]

@app.post("/api/train")
def train_model(req: TrainRequest):
    if data_cache["X_train"] is None:
        if not load_data("creditcard.csv") and not load_data("../creditcard.csv"):
            raise HTTPException(status_code=400, detail="Data not loaded.")
    
    X_train = data_cache["X_train"][req.features]
    y_train = data_cache["y_train"]
    X_val = data_cache["X_val"][req.features]
    y_val = data_cache["y_val"]
    
    model_key = f"{req.model_name}_{'-'.join(req.features)}"
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    scalers_cache[model_key] = scaler
    
    if req.model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight="balanced")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        explainers_cache[model_key] = shap.TreeExplainer(model)
        
    elif req.model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        probs = model.predict_proba(X_val_scaled)[:, 1]
        explainers_cache[model_key] = shap.LinearExplainer(model, X_train_scaled)
        
    elif req.model_name == "Gaussian Mixture (GMM)":
        model = GaussianMixture(n_components=1, random_state=0)
        model.fit(X_train[y_train == 0].values)
        scores = model.score_samples(X_val.values)
        preds = np.array([1 if s < (-50) else 0 for s in scores])
        probs = scores # proxy
        
    elif req.model_name == "XGBoost":
        scale_pos = sum(y_train==0) / max(1, sum(y_train==1))
        model = xgb.XGBClassifier(n_estimators=100, random_state=0, scale_pos_weight=scale_pos)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        explainers_cache[model_key] = shap.TreeExplainer(model)
        
    elif req.model_name == "Deep Autoencoder":
        input_dim = len(req.features)
        model = Autoencoder(input_dim)
        
        X_normal = X_train_scaled[y_train == 0]
        dataset = TensorDataset(torch.FloatTensor(X_normal))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(5): # Fast training for demo
            for batch in loader:
                inputs = batch[0]
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            reconstructions = model(X_val_tensor)
            mse = torch.mean((X_val_tensor - reconstructions)**2, dim=1).numpy()
            
        threshold = np.percentile(mse, 95) # Top 5% anomaly
        preds = np.array([1 if e > threshold else 0 for e in mse])
        probs = mse # proxy
        models_cache[model_key + "_threshold"] = threshold
        
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    models_cache[model_key] = model
    
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(y_val, probs)
    except:
        auc = None
        
    try:
        pr_auc = average_precision_score(y_val, probs)
    except:
        pr_auc = None
        
    metrics_dict = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None
    }
    metrics_cache[model_key] = metrics_dict
        
    return {
        "message": f"{req.model_name} trained successfully",
        "metrics": metrics_dict
    }

@app.get("/api/models/metrics")
def get_metrics():
    return {"metrics": metrics_cache}

class PredictRequest(BaseModel):
    model_name: str
    features: List[str]
    input_data: Dict[str, float]

@app.post("/api/predict")
def predict(req: PredictRequest):
    model_key = f"{req.model_name}_{'-'.join(req.features)}"
    if model_key not in models_cache:
        raise HTTPException(status_code=400, detail="Model not trained. Train first.")
    
    model = models_cache[model_key]
    scaler = scalers_cache.get(model_key)
    input_df = pd.DataFrame([req.input_data])[req.features]
    
    shap_values_dict = None
    
    if req.model_name == "Deep Autoencoder":
        scaled_input = scaler.transform(input_df)
        tensor_input = torch.FloatTensor(scaled_input)
        model.eval()
        with torch.no_grad():
            reconstructions = model(tensor_input)
            mse = torch.mean((tensor_input - reconstructions)**2, dim=1).numpy()[0]
        
        threshold = models_cache[model_key + "_threshold"]
        is_fraud = 1 if mse > threshold else 0
        return {"prediction": is_fraud, "score": float(mse), "threshold": float(threshold)}
        
    elif req.model_name == "Gaussian Mixture (GMM)":
        score = model.score_samples(input_df.values)[0]
        is_fraud = 1 if score < (-50) else 0
        return {"prediction": is_fraud, "score": float(score)}
        
    else:
        if req.model_name == "Logistic Regression":
            X_infer = scaler.transform(input_df)
            prob = model.predict_proba(X_infer)[0][1]
            is_fraud = int(model.predict(X_infer)[0])
            if model_key in explainers_cache:
                explainer = explainers_cache[model_key]
                shaps = explainer.shap_values(X_infer)[0]
                shap_values_dict = {feat: float(val) for feat, val in zip(req.features, shaps)}
        else:
            prob = model.predict_proba(input_df)[0][1]
            is_fraud = int(model.predict(input_df)[0])
            if model_key in explainers_cache:
                explainer = explainers_cache[model_key]
                try:
                    shaps = explainer.shap_values(input_df)[0]
                    if hasattr(shaps, "shape") and len(shaps.shape) == 2:
                        shaps = shaps[:, 1]
                    elif hasattr(shaps, "shape") and len(shaps.shape) == 1:
                        pass
                    elif isinstance(shaps, list) and len(shaps) > 1:
                        shaps = shaps[1]
                    shap_values_dict = {feat: float(val) for feat, val in zip(req.features, shaps)}
                except Exception as e:
                    print(e)
                    pass

        res = {"prediction": is_fraud, "probability": float(prob)}
        if shap_values_dict:
            res["shap_values"] = shap_values_dict
        return res

@app.get("/api/data/info")
def data_info():
    if data_cache["df"] is None:
        if not load_data("creditcard.csv") and not load_data("../creditcard.csv"):
            return {"loaded": False}
    df = data_cache["df"]
    
    class_counts = df["Class"].value_counts().to_dict()
    features = [c for c in df.columns if c not in ["Class", "Time", "Amount"]]
    
    return {
        "loaded": True,
        "total_transactions": len(df),
        "num_features": len(features),
        "class_counts": class_counts,
        "features": features
    }
