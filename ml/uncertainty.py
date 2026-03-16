import os
import joblib
import numpy as np
from scipy.stats import entropy

ARTIFACTS_PATH = os.path.join("ml", "artifacts")

def load_uncertainty_models():
    # Load NGBoost and TabNet for uncertainty estimation
    ngboost_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_ngboost.pkl"))
    tabnet_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_tabnet.pkl"))
    return ngboost_model, tabnet_model

def calculate_uncertainty(X_sample):
    ngboost_model, tabnet_model = load_uncertainty_models()
    
    # 1. NGBoost Probabilistic Variance
    ngb_probs = ngboost_model.predict_proba(X_sample)[0]
    ngb_entropy = entropy(ngb_probs) # Higher entropy means higher uncertainty
    
    # 2. Monte Carlo Dropout Approximation (Deep Tabular Model)
    # Simulating MC Dropout by adding slight Gaussian noise over multiple forward passes
    mc_preds = []
    for _ in range(10):
        noise = np.random.normal(0, 0.01, X_sample.shape)
        mc_preds.append(tabnet_model.predict_proba(X_sample + noise)[0])
    
    mc_variance = np.var(np.array(mc_preds), axis=0).mean()
    
    # Combined Risk Score based on NGBoost and MC Dropout
    risk_score = (ngb_entropy * 0.5) + (mc_variance * 0.5)
    
    confidence = np.max(ngb_probs)
    predicted_class = int(np.argmax(ngb_probs))

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "risk_score": float(risk_score)
    }

if __name__ == "__main__":
    print("Uncertainty module updated with NGBoost and MC Dropout.")