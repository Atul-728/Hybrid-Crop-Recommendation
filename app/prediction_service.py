import os
import joblib
import numpy as np
from ml.uncertainty import calculate_uncertainty

ARTIFACTS_PATH = os.path.join("ml", "artifacts")


# ----------------------------
# Load Production Bundle
# ----------------------------

def load_production_bundle():
    return joblib.load(os.path.join(ARTIFACTS_PATH, "production_bundle.pkl"))


# ----------------------------
# Farmer Friendly Explanation
# ----------------------------

def generate_reason(input_data, predicted_crop):

    reasons = []

    # Rainfall logic
    if input_data["rainfall"] > 200:
        reasons.append("Your field receives high rainfall which supports crops that require more water.")
    elif input_data["rainfall"] < 100:
        reasons.append("Your field has low rainfall, so crops suitable for dry conditions are preferred.")

    # Temperature logic
    if 20 <= input_data["temperature"] <= 35:
        reasons.append("Temperature in your area is ideal for healthy crop growth.")
    else:
        reasons.append("Temperature conditions are manageable for this crop.")

    # Soil pH logic
    if 6 <= input_data["ph"] <= 7:
        reasons.append("Soil pH is balanced which helps in better nutrient absorption.")
    else:
        reasons.append("Soil pH is slightly outside ideal range but still acceptable.")

    # Nutrients
    if input_data["N"] > 40:
        reasons.append("Nitrogen level is good for strong leaf development.")

    if input_data["P"] > 40:
        reasons.append("Phosphorus level supports root and flower growth.")

    if input_data["K"] > 40:
        reasons.append("Potassium level improves plant strength and disease resistance.")

    explanation = " ".join(reasons)

    return explanation + f" Based on these conditions, {predicted_crop} is highly suitable for your land."


# ----------------------------
# Risk Category Logic
# ----------------------------

def get_risk_level(confidence_percent, risk_score):

    if confidence_percent >= 85 and risk_score < 0.2:
        return "Low Risk – Highly Suitable Crop"

    elif confidence_percent >= 70:
        return "Moderate Risk – Suitable with proper care"

    else:
        return "High Risk – Consider alternative crops"


# ----------------------------
# Predict Crop
# ----------------------------

def predict_crop(input_data: dict):

    bundle = load_production_bundle()

    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]

    # Convert input dict to array
    input_array = np.array([[
        input_data["N"],
        input_data["P"],
        input_data["K"],
        input_data["temperature"],
        input_data["humidity"],
        input_data["ph"],
        input_data["rainfall"]
    ]])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict probabilities
    probs = model.predict_proba(input_scaled)[0]

    # Get top 3 indices
    top3_indices = np.argsort(probs)[-3:][::-1]

    # Create structured top3 list with confidence
    top3 = []
    for idx in top3_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        crop_conf = round(float(probs[idx]) * 100, 2)

        top3.append({
            "name": crop_name,
            "confidence": crop_conf
        })

    # Top 1
    top1 = top3[0]["name"]
    confidence_percent = top3[0]["confidence"]

    # Uncertainty calculation
    uncertainty_result = calculate_uncertainty(input_scaled)
    raw_risk_score = float(uncertainty_result["risk_score"])

    # Convert risk level
    risk_level = get_risk_level(confidence_percent, raw_risk_score)

    # Generate farmer explanation
    reason = generate_reason(input_data, top1)

    return {
        "top1": top1,
        "top3": top3,
        "confidence": confidence_percent,
        "risk_score": round(raw_risk_score, 6),
        "risk_level": risk_level,
        "reason": reason
    }