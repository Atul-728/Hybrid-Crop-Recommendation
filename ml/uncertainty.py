import os
import joblib
import numpy as np


ARTIFACTS_PATH = os.path.join("ml", "artifacts")


# ----------------------------
# Load Models
# ----------------------------

def load_models():
    rf_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_rf.pkl"))
    xgb_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_xgb.pkl"))
    lgbm_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_lgbm.pkl"))
    cat_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_catboost.pkl"))
    stacked_model = joblib.load(os.path.join(ARTIFACTS_PATH, "stacked_model.pkl"))

    return rf_model, xgb_model, lgbm_model, cat_model, stacked_model


# ----------------------------
# Calculate Uncertainty
# ----------------------------

def calculate_uncertainty(X_sample):
    rf_model, xgb_model, lgbm_model, cat_model, stacked_model = load_models()

    base_models = [rf_model, xgb_model, lgbm_model, cat_model]

    prob_list = []

    for model in base_models:
        probs = model.predict_proba(X_sample)
        prob_list.append(probs)

    prob_array = np.array(prob_list)

    # Mean probability across models
    mean_probs = np.mean(prob_array, axis=0)

    # Variance across models
    variance = np.var(prob_array, axis=0)

    # Confidence = highest class probability
    confidence = np.max(mean_probs)

    # Risk score = mean variance
    risk_score = np.mean(variance)

    predicted_class = np.argmax(mean_probs)

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "risk_score": float(risk_score)
    }


if __name__ == "__main__":
    print("Uncertainty module ready.")