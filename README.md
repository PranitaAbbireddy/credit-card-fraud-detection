# FraudLens: ML-Powered Transaction Anomaly Detection

## Project Overview

Credit card fraud is a major challenge for financial institutions and consumers. Detecting fraudulent transactions accurately is crucial for minimizing economic losses. This project presents a decoupled machine learning architecture for analyzing and detecting fraudulent transactions in credit card datasets.

It combines exploratory data analysis, dynamic model training, automated metric tracking, and an interpretable fraud prediction simulator into an interactive application powered by a FastAPI backend and a Streamlit frontend.

## System Capabilities

FraudLens

│
├── Data & EDA
├── Model Training
│   ├── Logistic Regression
│   ├── Random Forest
│   ├── XGBoost
│   ├── GMM
│   └── Autoencoder
│
├── Evaluation
│   ├── Precision / Recall / F1
│   ├── ROC-AUC
│   └── PR-AUC (Average Precision)
│
├── Explainability
│   └── SHAP
│
└── Application
    ├── FastAPI (Backend API)
    └── Streamlit (Frontend Dashboard)

## Methodology

### 1. Data preprocessing

* Features V1-V28 are pre-transformed via Principal Component Analysis (PCA) to preserve privacy.

* The backend strictly manages train/test/validation splitting and stratification to prevent data leakage.

* Feature scaling (StandardScaler) is applied appropriately for distance-based and gradient-descent models (e.g., Logistic Regression, Autoencoder) while preserving unscaled inputs for tree-based models.

### 2. Supervised learning

* **Logistic Regression:** Serves as a linear baseline.

* **Random Forest:** A robust baseline supervised ensemble model.

* **XGBoost:** An advanced gradient boosting framework optimized for tabular data and severe class imbalance.

### 3. Unsupervised anomaly detection

* **Gaussian Mixture Model (GMM):** An unsupervised probabilistic model for density estimation.

* **Autoencoder (PyTorch):** A deep neural network trained exclusively on legitimate (non-fraudulent) transactions to learn latent representations of normal behavior.

### 4. Model evaluation

Given the extreme class imbalance (less than 0.2% fraud), accuracy is an insufficient metric. Models are strictly evaluated on:

* **Precision and Recall**

* **F1 Score**

* **ROC-AUC**

* **PR-AUC (Average Precision):** Used to evaluate the precision–recall trade-off under severe class imbalance.

### 5. Threshold selection

For GMM and the Autoencoder, anomaly scores are calculated using log-likelihood and reconstruction error respectively. Thresholds are selected using the validation set and evaluated based on the resulting precision–recall trade-off.

### 6. Explainability

To improve model transparency, SHapley Additive exPlanations (SHAP) are integrated into the prediction pipeline. The dashboard visualizes the positive or negative contribution of the transformed V1-V28 PCA components to the model's prediction.

## System Architecture

FraudLens follows a decoupled frontend/backend architecture that separates the interactive dashboard from the machine learning and inference layer.

```text
                         ┌──────────────────────────┐
                         │     Credit Card Data     │
                         │      creditcard.csv      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      Data Processing     │
                         │                          │
                         │ • Validation             │
                         │ • Train/Val/Test Split   │
                         │ • Feature Scaling        │
                         │ • Preprocessing          │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                  ┌─────────────────────────────────────────┐
                  │          FastAPI Backend API            │
                  │                                         │
                  │  ┌───────────────────────────────────┐  │
                  │  │       Model Training & Inference  │  │
                  │  │                                   │  │
                  │  │  • Logistic Regression            │  │
                  │  │  • Random Forest                  │  │
                  │  │  • XGBoost                        │  │
                  │  │  • GMM                            │  │
                  │  │  • Autoencoder                    │  │
                  │  └───────────────────────────────────┘  │
                  │                                         │
                  │  ┌───────────────────────────────────┐  │
                  │  │ Evaluation & Explainability       │  │
                  │  │                                   │  │
                  │  │  • Precision / Recall / F1        │  │
                  │  │  • ROC-AUC / PR-AUC               │  │
                  │  │  • Threshold Analysis             │  │
                  │  │  • SHAP Explanations              │  │
                  │  └───────────────────────────────────┘  │
                  └────────────────────┬────────────────────┘
                                       │
                              REST API / Predictions
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │    Streamlit Frontend    │
                         │                          │
                         │ • EDA & Data Analysis    │
                         │ • Model Comparison       │
                         │ • Prediction Simulator   │
                         │ • Metrics Visualization  │
                         │ • SHAP Explanations      │
                         └──────────────────────────┘