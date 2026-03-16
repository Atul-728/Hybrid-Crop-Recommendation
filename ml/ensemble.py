import os
import joblib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ARTIFACTS_PATH = os.path.join("ml", "artifacts")

# ----------------------------
# Load Processed Data
# ----------------------------
def load_data():
    X_train = joblib.load(os.path.join(ARTIFACTS_PATH, "X_train.pkl"))
    X_test = joblib.load(os.path.join(ARTIFACTS_PATH, "X_test.pkl"))
    y_train = joblib.load(os.path.join(ARTIFACTS_PATH, "y_train.pkl"))
    y_test = joblib.load(os.path.join(ARTIFACTS_PATH, "y_test.pkl"))
    return X_train, X_test, y_train, y_test

# ----------------------------
# Load Base Models (Including TabNet)
# ----------------------------
def load_base_models():
    rf_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_rf.pkl"))
    xgb_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_xgb.pkl"))
    lgbm_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_lgbm.pkl"))
    cat_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_catboost.pkl"))
    tabnet_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_tabnet.pkl"))
    return rf_model, xgb_model, lgbm_model, cat_model, tabnet_model

# ----------------------------
# Create Meta Features
# ----------------------------
def create_meta_features(models, X):
    meta_features = []
    for model in models:
        # TabNet returns predictions slightly differently, handle if needed, but predict_proba works usually
        probs = model.predict_proba(X)
        meta_features.append(probs)
    return np.hstack(meta_features)

# ----------------------------
# Train Meta-Learner
# ----------------------------
def train_meta_learner(X_meta_train, y_train):
    print("Training Stacked Meta-Learner (with TabNet)...")
    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(X_meta_train, y_train)
    joblib.dump(meta_model, os.path.join(ARTIFACTS_PATH, "stacked_model.pkl"))
    return meta_model

# ----------------------------
# Evaluate Ensemble
# ----------------------------
def evaluate_ensemble(meta_model, X_meta_test, y_test):
    print("Evaluating Stacked Ensemble...")
    y_pred = meta_model.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {"accuracy": accuracy, "f1_score": f1, "classification_report": report}
    with open(os.path.join(ARTIFACTS_PATH, "ensemble_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Ensemble Accuracy:", accuracy)
    return metrics

# ----------------------------
# Run Ensemble Pipeline
# ----------------------------
def run_ensemble():
    X_train, X_test, y_train, y_test = load_data()
    models = load_base_models()
    
    X_meta_train = create_meta_features(models, X_train)
    X_meta_test = create_meta_features(models, X_test)
    
    meta_model = train_meta_learner(X_meta_train, y_train)
    evaluate_ensemble(meta_model, X_meta_test, y_test)
    print("Stacked Ensemble training completed successfully.")

if __name__ == "__main__":
    run_ensemble()