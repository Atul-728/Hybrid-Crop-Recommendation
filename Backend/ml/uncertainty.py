from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy

try:
    from . import tabnet_surrogate as _tabnet_surrogate_compat
except ImportError:
    try:
        import tabnet_surrogate as _tabnet_surrogate_compat  # noqa: F401
    except ImportError:
        _tabnet_surrogate_compat = None


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = BASE_DIR / "ml" / "artifacts"

FEATURE_ORDER = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "seasonal_index", "npk_ratio", "humidity_rainfall_interaction"
]


def load_artifact(name):
    path = ARTIFACTS_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def load_models():
    models = {
        "rf": load_artifact("model_rf.pkl"),
        "xgb": load_artifact("model_xgb.pkl"),
        "lgbm": load_artifact("model_lgbm.pkl"),
        "catboost": load_artifact("model_catboost.pkl"),
    }

    tabnet_path = ARTIFACTS_PATH / "model_tabnet.pkl"
    if tabnet_path.exists():
        models["tabnet"] = joblib.load(tabnet_path)

    meta_model = load_artifact("stacked_model.pkl")
    label_encoder = load_artifact("label_encoder.pkl")
    imputer = load_artifact("imputer.pkl") if (ARTIFACTS_PATH / "imputer.pkl").exists() else None
    scaler = load_artifact("scaler.pkl") if (ARTIFACTS_PATH / "scaler.pkl").exists() else None

    return models, meta_model, label_encoder, imputer, scaler


def build_input_array(input_data):
    try:
        data = dict(input_data)
        N = float(data.get("N", 0))
        P = float(data.get("P", 0))
        K = float(data.get("K", 0))
        temp = float(data.get("temperature", 0))
        hum = float(data.get("humidity", 0))
        rain = float(data.get("rainfall", 0))
        
        data["seasonal_index"] = (temp * rain) / 100.0
        data["npk_ratio"] = N / (P + K + 1.0)
        data["humidity_rainfall_interaction"] = (hum * rain) / 1000.0
        
        row = [[float(data[f]) for f in FEATURE_ORDER]]
    except KeyError as e:
        raise ValueError(f"Missing feature: {str(e)}")
    except ValueError:
        raise ValueError("All input values must be numeric")
    return pd.DataFrame(row, columns=FEATURE_ORDER, dtype=float)


def preprocess_input(input_data):
    _, _, _, imputer, scaler = load_models()
    X = build_input_array(input_data)
    if imputer is not None:
        X = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_ORDER)
    return X


def tabnet_mc_proba(tabnet_model, X, mc_samples=25):
    if tabnet_model is None:
        return None
    if hasattr(tabnet_model, "predict_proba_mc"):
        return tabnet_model.predict_proba_mc(X, mc_samples=mc_samples)
    return None


def calculate_uncertainty(input_data):
    models, meta_model, label_encoder, imputer, scaler = load_models()

    X = build_input_array(input_data)
    if imputer is not None:
        X = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_ORDER)

    base_probabilities = []
    tabnet_probs = None
    tabnet_model = models.get("tabnet")
    for name, model in models.items():
        prob = model.predict_proba(X)[0]
        base_probabilities.append(prob)
        if name == "tabnet":
            tabnet_probs = prob

    base_probabilities = np.array(base_probabilities)
    mean_probs = base_probabilities.mean(axis=0)

    ensemble_input = np.hstack([m.predict_proba(X) for m in models.values()])
    ensemble_probs = meta_model.predict_proba(ensemble_input)[0]

    mc_samples = None
    if tabnet_model is not None and hasattr(tabnet_model, "predict_proba_mc"):
        _, mc_samples = tabnet_model.predict_proba_mc(X, mc_samples=30, return_samples=True)

    predicted_index = int(np.argmax(ensemble_probs))
    predicted_crop = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(ensemble_probs) * 100)

    disagreement = float(np.mean(np.std(base_probabilities, axis=0)))
    entropy_score = float(entropy(mean_probs + 1e-12) / np.log(len(ensemble_probs)))
    mc_score = 0.0
    if mc_samples is not None:
        mc_score = float(np.mean(np.std(mc_samples[:, 0, :], axis=0)))
    risk_score = float(np.clip(0.45 * disagreement + 0.45 * entropy_score + 0.10 * mc_score, 0.0, 1.0))

    return {
        "predicted_crop": predicted_crop,
        "confidence": round(confidence, 2),
        "uncertainty_score": round(risk_score, 6),
        "model_disagreement": round(disagreement, 6),
        "entropy": round(entropy_score, 6),
        "class_probabilities": {
            str(label_encoder.inverse_transform([i])[0]): round(float(p) * 100, 2)
            for i, p in enumerate(ensemble_probs)
        },
    }


if __name__ == "__main__":
    sample_input = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.5,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }

    print(calculate_uncertainty(sample_input))
