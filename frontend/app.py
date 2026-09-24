import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os

st.set_page_config(page_title="FraudLens Dashboard", layout="wide")
API_URL = "http://127.0.0.1:8000/api"

# Load local data for EDA
@st.cache_data
def load_local_data():
    paths_to_try = ["creditcard.csv", "../creditcard.csv"]
    for p in paths_to_try:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df = load_local_data()

st.sidebar.header("⚙️ Settings")

features = []
if df is not None:
    features = [c for c in df.columns if c not in ["Class", "Time", "Amount"]]
else:
    # Try getting from API
    try:
        resp = requests.get(f"{API_URL}/data/info").json()
        if resp.get("loaded"):
            features = resp.get("features", [])
    except:
        pass

features_selected = st.sidebar.multiselect(
    "Select features for Modeling",
    options=features,
    default=["V14", "V17", "V12"] if features else []
)

st.title("FraudLens: ML-Powered Transaction Anomaly Detection")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview & EDA", "⚙️ Model Training", "📈 Model Comparison", "🧾 Real-time Prediction"]
)

with tab1:
    st.subheader("Dataset Overview & Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        col1.metric("Total Transactions", df.shape[0])
        col2.metric("Features (excluding Time/Amount/Class)", df.shape[1] - 3)

        st.write("### Transaction Class Distribution")
        class_counts = df["Class"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6,4))
        
        # Use log scale so the tiny Fraud bar is visible
        sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#2ca02c", "#d62728"], ax=ax)
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        ax.set_yscale('log')
        ax.set_ylabel("Count (Log Scale)")
        
        # Add text labels on top of the bars
        for i, count in enumerate(class_counts.values):
            ax.text(i, count, f"{count}", ha='center', va='bottom', fontweight='bold')
            
        st.pyplot(fig)

        feature = st.selectbox("Select a feature to visualize (Train set proxy)", features)
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        sns.kdeplot(data=df[df['Class']==0], x=feature, fill=True, color="#4dd0e1", label="Non-Fraud", ax=ax2)
        sns.kdeplot(data=df[df['Class']==1], x=feature, fill=True, color="#f06292", label="Fraud", ax=ax2)
        ax2.set_title(f"Distribution of {feature}")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Dataset not found locally for EDA. Please place `creditcard.csv` in the root folder.")

with tab2:
    st.subheader("Train Models via API")
    if not features_selected:
        st.warning("Select features first.")
    else:
        model_to_train = st.selectbox("Select Model to Train", ["Random Forest", "Logistic Regression", "Gaussian Mixture (GMM)", "XGBoost", "Deep Autoencoder"])
        
        if st.button("🚀 Train Model"):
            with st.spinner(f"Training {model_to_train} on Backend..."):
                try:
                    resp = requests.post(f"{API_URL}/train", json={
                        "model_name": model_to_train,
                        "features": features_selected
                    })
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(data["message"])
                        st.write("### Validation Metrics:")
                        metrics = data["metrics"]
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Precision", f"{metrics['precision']:.4f}")
                        m2.metric("Recall", f"{metrics['recall']:.4f}")
                        m3.metric("F1 Score", f"{metrics['f1']:.4f}")
                        m4.metric("ROC AUC", f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A")
                        m5.metric("PR AUC", f"{metrics['pr_auc']:.4f}" if metrics.get('pr_auc') else "N/A")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

with tab3:
    st.subheader("Model Comparison")
    if st.button("🔄 Refresh Metrics"):
        try:
            resp = requests.get(f"{API_URL}/models/metrics")
            if resp.status_code == 200:
                data = resp.json().get("metrics", {})
                if not data:
                    st.info("No models have been trained yet. Train models in Tab 2 first.")
                else:
                    rows = []
                    for model_key, mets in data.items():
                        parts = model_key.split('_')
                        model_name = parts[0]
                        # Join features back if there are multiple parts
                        features_str = "_".join(parts[1:])
                        rows.append({
                            "Model": model_name,
                            "Features": features_str,
                            "Precision": mets.get("precision"),
                            "Recall": mets.get("recall"),
                            "F1": mets.get("f1"),
                            "ROC AUC": mets.get("roc_auc"),
                            "PR AUC": mets.get("pr_auc")
                        })
                    res_df = pd.DataFrame(rows)
                    st.dataframe(res_df)
                    
                    try:
                        chart_df = res_df.set_index("Model")[["Precision", "Recall", "F1"]].astype(float)
                        st.bar_chart(chart_df)
                    except Exception:
                        pass
        except Exception as e:
            st.error(f"Could not fetch metrics: {e}")

with tab4:
    st.subheader("Real-time Prediction Simulator & Explainability")
    if not features_selected:
        st.warning("Select features first.")
    else:
        model_choice = st.selectbox("Select Model for Prediction", ["Random Forest", "Logistic Regression", "Gaussian Mixture (GMM)", "XGBoost", "Deep Autoencoder"])
        
        st.write("Input Feature Values:")
        input_vals = {}
        cols = st.columns(len(features_selected))
        for i, feat in enumerate(features_selected):
            input_vals[feat] = cols[i].number_input(feat, value=0.0, format="%.6f")

        if st.button("🔎 Predict Fraud"):
            with st.spinner("Calling API..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json={
                        "model_name": model_choice,
                        "features": features_selected,
                        "input_data": input_vals
                    })
                    if resp.status_code == 200:
                        res = resp.json()
                        st.metric("Prediction", "Fraud 🚨" if res["prediction"] == 1 else "Non-Fraud ✅")
                        if "probability" in res:
                            st.progress(res["probability"])
                            st.write(f"Fraud Probability: {res['probability']:.4f}")
                        elif "score" in res:
                            st.write(f"Anomaly Score: {res['score']:.4f}")
                            if "threshold" in res:
                                st.write(f"Anomaly Threshold: {res['threshold']:.4f}")
                        
                        if "shap_values" in res:
                            st.write("### Model Explainability (SHAP)")
                            st.write("This chart shows how much each feature contributed to the model's prediction.")
                            shap_df = pd.DataFrame(list(res["shap_values"].items()), columns=["Feature", "SHAP Value"])
                            shap_df = shap_df.sort_values(by="SHAP Value", ascending=True)
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            colors = ['red' if x > 0 else 'blue' for x in shap_df["SHAP Value"]]
                            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
                            ax.set_xlabel("SHAP Value (Impact on Prediction)")
                            ax.axvline(x=0, color='black', linestyle='--')
                            st.pyplot(fig)
                            
                    else:
                        st.error(f"Error: {resp.text}. Did you train the model first?")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
