import os
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv

from ml.uncertainty import calculate_uncertainty
from app.price_fallback import REALISTIC_PRICE_DATA, REGION_MULTIPLIERS
from app.crop_mapping import CROP_MAPPING

# -------------------------------
# ENV CONFIG
# -------------------------------

load_dotenv()
API_KEY = os.getenv("MANDI_API_KEY")

# -------------------------------
# MODEL ARTIFACT PATH
# -------------------------------

ARTIFACTS_PATH = os.path.join("ml", "artifacts")


# -------------------------------
# LOAD ML MODELS
# -------------------------------

def load_all_models():
    models = {
        "rf": joblib.load(os.path.join(ARTIFACTS_PATH, "model_rf.pkl")),
        "xgb": joblib.load(os.path.join(ARTIFACTS_PATH, "model_xgb.pkl")),
        "lgbm": joblib.load(os.path.join(ARTIFACTS_PATH, "model_lgbm.pkl")),
        "cat": joblib.load(os.path.join(ARTIFACTS_PATH, "model_catboost.pkl")),
        "tabnet": joblib.load(os.path.join(ARTIFACTS_PATH, "model_tabnet.pkl"))
    }

    meta_model = joblib.load(os.path.join(ARTIFACTS_PATH, "stacked_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    le = joblib.load(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))
    te = joblib.load(os.path.join(ARTIFACTS_PATH, "target_encoder.pkl"))

    return models, meta_model, scaler, le, te


# -------------------------------
# LIVE MANDI PRICE FETCH
# -------------------------------

def fetch_live_price(crop_name, region, state):
    state = state.upper() if state else ""
    crop = crop_name.lower()

    rm = REGION_MULTIPLIERS.get(region, {"cost":1.0,"price":1.0})

    mapping = CROP_MAPPING.get(crop)

    if mapping:
        api_names = mapping["api_names"]
    else:
        api_names = [crop.upper()]

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    # -----------------------
    # TRY API WITH ALIASES
    # -----------------------

    try:

        for commodity in api_names:

            params = {
                "api-key": API_KEY,
                "format": "json",
                "limit": 5,
                "filters[commodity]": commodity,
                "filters[state]": state
            }

            r = requests.get(url, params=params, timeout=5)

            if r.status_code != 200:
                continue

            data = r.json()

            if "records" in data and len(data["records"]) > 0:

                prices = []

                for rec in data["records"]:
                    if rec.get("modal_price"):
                        prices.append(float(rec["modal_price"]))

                if prices:

                    avg_price = sum(prices) / len(prices)

                    cost = avg_price * 0.65

                    return round(cost * rm["cost"], 2), round(avg_price * rm["price"], 2)

    except Exception as e:
        print("API price fetch failed:", e)

    # -----------------------
    # FALLBACK LOCAL DATA
    # -----------------------

    crop_data = REALISTIC_PRICE_DATA.get(crop, REALISTIC_PRICE_DATA["default"])

    cost = crop_data["cost"] * rm["cost"]
    price = crop_data["price"] * rm["price"]

    return round(cost, 2), round(price, 2)


# -------------------------------
# MAIN PREDICTION PIPELINE
# -------------------------------

def predict_crop(input_data: dict):

    models, meta_model, scaler, le, te = load_all_models()

    month = datetime.now().month
    seasonal_index = np.sin(2 * np.pi * month / 12)

    try:
        region_encoded = int(te.transform([input_data["region"]])[0])
    except Exception:
        region_encoded = 0

    profit_ratio = (
        (input_data["market_price"] - input_data["production_cost"])
        / input_data["production_cost"]
        if input_data["production_cost"] > 0
        else 0
    )

    input_array = np.array([
        [
            input_data["N"],
            input_data["P"],
            input_data["K"],
            input_data["temperature"],
            input_data["humidity"],
            input_data["ph"],
            input_data["rainfall"],
            month,
            seasonal_index,
            input_data["rainfall"],  # rolling_rainfall approx
            input_data["market_price"],
            input_data["production_cost"],
            profit_ratio,
            region_encoded
        ]
    ])

    input_scaled = scaler.transform(input_array)

    meta_features = []

    for name, m in models.items():
        probs = m.predict_proba(input_scaled)
        meta_features.append(probs)

    X_meta = np.hstack(meta_features)

    final_probs = meta_model.predict_proba(X_meta)[0]

    top3_idx = np.argsort(final_probs)[-3:][::-1]

    top3 = []
    profit_scores = []

    user_region = input_data["region"]

    for idx in top3_idx:

        crop_name = le.inverse_transform([idx])[0]

        confidence = float(final_probs[idx]) * 100

        dynamic_cost, dynamic_price = fetch_live_price(
            crop_name,
            user_region,
            input_data["state"]
        )

        # realistic profit
        profit_estimate = dynamic_price - dynamic_cost

        top3.append({
            "name": crop_name,
            "confidence": round(confidence, 2),
            "dynamic_cost": dynamic_cost,
            "dynamic_price": dynamic_price,
            "estimated_profit": round(profit_estimate, 2)
        })

        profit_scores.append(profit_estimate)

    best_index = int(np.argmax(profit_scores))
    best_crop = top3[0]
    uncertainty = calculate_uncertainty(input_scaled)

    # -------------------------------
    # AGRICULTURAL RISK
    # -------------------------------

    if best_crop["confidence"] >= 75:
        risk_level = "Low Risk – Highly Suitable for your Soil"
    elif best_crop["confidence"] >= 40:
        risk_level = "Moderate Risk – Marginally Suitable"
    else:
        risk_level = "High Risk – Poor Match for your Soil/Climate"

    # -------------------------------
    # FINANCIAL ALERT
    # -------------------------------

    if best_crop["estimated_profit"] < 0:

        if best_crop["confidence"] >= 75:
            alert_reason = (
                f"ALERT: {best_crop['name']} is highly suitable for your land "
                f"({best_crop['confidence']}% match), but current market rates "
                f"show a LOSS."
            )

        else:
            alert_reason = (
                f"WARNING: {best_crop['name']} is both a High Risk for your soil "
                f"and shows a potential financial LOSS."
            )

    else:

        if best_crop["confidence"] < 50:
            alert_reason = (
                f"CAUTION: {best_crop['name']} shows a potential profit but "
                f"is agriculturally risky ({best_crop['confidence']}% match)."
            )

        else:
            alert_reason = (
                f"SUCCESS: {best_crop['name']} is highly suitable for your land "
                f"and shows a positive expected profit."
            )

    return {
        "top1": best_crop["name"],
        "top3": top3,
        "confidence": best_crop["confidence"],
        "risk_score": round(float(uncertainty["risk_score"]), 6),
        "expected_profit": round(best_crop["estimated_profit"], 2),
        "risk_level": risk_level,
        "reason": alert_reason
    }