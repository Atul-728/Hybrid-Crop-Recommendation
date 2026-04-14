from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
import google.generativeai as genai
import os
import re

try:
    from .ml import tabnet_surrogate as _tabnet_surrogate_compat
except ImportError:
        _tabnet_surrogate_compat = None

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_PATH = BASE_DIR / "ml" / "artifacts"

FEATURE_ORDER = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "seasonal_index", "npk_ratio", "humidity_rainfall_interaction"
]

CROP_ECONOMICS = {
    "rice": {"cost": 2000, "price": 3500},
    "maize": {"cost": 1400, "price": 2500},
    "wheat": {"cost": 1800, "price": 3200},
    "mango": {"cost": 2500, "price": 6000},
    "pigeonpeas": {"cost": 3200, "price": 7500},
    "mothbeans": {"cost": 3000, "price": 6200},
    "cotton": {"cost": 4200, "price": 8000},
    "apple": {"cost": 6000, "price": 14000},
    "banana": {"cost": 1800, "price": 3800},
    "coffee": {"cost": 8000, "price": 18000},
    "jute": {"cost": 3500, "price": 5500},
    "sugarcane": {"cost": 2200, "price": 4000},
    "coconut": {"cost": 3000, "price": 6500},
    "papaya": {"cost": 900, "price": 2500},
    "orange": {"cost": 2800, "price": 5500},
    "grapes": {"cost": 4500, "price": 9000},
    "pomegranate": {"cost": 3500, "price": 8000},
    "watermelon": {"cost": 1200, "price": 2800},
    "muskmelon": {"cost": 1300, "price": 2800},
    "blackgram": {"cost": 2800, "price": 6000},
    "mungbean": {"cost": 2800, "price": 6000},
    "lentil": {"cost": 2500, "price": 5500},
    "chickpea": {"cost": 3000, "price": 6000},
    "kidneybeans": {"cost": 3200, "price": 7000},
    "default": {"cost": 2500, "price": 5000},
}

REGION_MULTIPLIERS = {
    "north": {"cost": 1.0, "price": 1.05}, "south": {"cost": 1.1, "price": 1.08},
    "east": {"cost": 0.95, "price": 0.97}, "west": {"cost": 1.15, "price": 1.12},
    "central": {"cost": 1.0, "price": 1.0},
}

# 🔥 GLOBAL CACHE
_MODELS_CACHE = None
_PRICE_CACHE = {} # Cache for API responses

# ============================
# GROQ (Primary) + GEMINI (Fallback) — AI Text Generation
# ============================
# Groq Free Tier: 30 RPM, 14,400 RPD — much better than Gemini's 5-15 RPM
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Best quality, 30 RPM
    "gemma2-9b-it",              # Fast, good for structured output
    "llama-3.1-8b-instant",      # Ultra-fast, lightweight
]
_groq_model_index = 0

# Gemini as fallback when Groq is exhausted
GEMINI_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]
_gemini_model_index = 0

def gemini_generate(prompt: str, api_key: str, timeout: int = 5) -> str | None:
    """Try Groq first (30 RPM free), then fall back to Gemini if Groq fails."""
    import requests
    
    # === STEP 1: Try Agent Router (Priority - Unlimited Quota) ===
    agentrouter_key = os.environ.get("AGENTROUTER_API_KEY")
    if agentrouter_key:
        try:
            res = requests.post(
                "https://agentrouter.org/v1/chat/completions",
                json={
                    "model": "deepseek-v3.1",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                headers={
                    "Authorization": f"Bearer {agentrouter_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout
            )
            if res.status_code == 200:
                text = res.json()["choices"][0]["message"]["content"]
                print("[AgentRouter] Success")
                return text
            else:
                print(f"[AgentRouter] Error {res.status_code}, falling back to Groq...")
        except Exception as e:
            print(f"[AgentRouter] Error: {e}, falling back to Groq...")

    # === STEP 2: Try Groq (Secondary) ===
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        global _groq_model_index
        n = len(GROQ_MODELS)
        start_idx = _groq_model_index % n
        _groq_model_index += 1
        
        for i in range(n):
            model = GROQ_MODELS[(start_idx + i) % n]
            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    },
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=timeout
                )
                if res.status_code == 200:
                    text = res.json()["choices"][0]["message"]["content"]
                    print(f"[Groq] Success {model}")
                    return text
                elif res.status_code in (429, 503):
                    print(f"[Groq] {model} returned {res.status_code}, rotating...")
                    continue
                else:
                    print(f"[Groq] {model} returned {res.status_code}, rotating...")
                    continue
            except Exception as e:
                print(f"[Groq] {model} error: {e}, rotating...")
                continue
        print("[Groq] All models exhausted, falling back to Gemini...")
    
    # === STEP 2: Gemini Fallback ===
    global _gemini_model_index
    n = len(GEMINI_MODELS)
    start_idx = _gemini_model_index % n
    _gemini_model_index += 1
    
    for i in range(n):
        model = GEMINI_MODELS[(start_idx + i) % n]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            res = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            if res.status_code == 200:
                print(f"[Gemini] Success {model}")
                return res.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif res.status_code in (429, 503):
                print(f"[Gemini] {model} returned {res.status_code}, rotating...")
                continue
            else:
                print(f"[Gemini] {model} returned {res.status_code}, rotating...")
                continue
        except Exception as e:
            print(f"[Gemini] {model} error: {e}, rotating...")
            continue
    
    print("[AI] All Groq + Gemini models exhausted.")
    return None

def fetch_gemini_economics(crop_name, city, state, region):
    cache_key = f"{crop_name}_{city}_{state}".lower()
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]
    
    api_key = os.getenv("AGENTROUTER_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
        
    try:
        prompt = f"""
        Act as an expert Indian agricultural economist. Estimate the current market price and production cost strictly in INR PER 100 KG (per quintal) for '{crop_name}' farming in the city of '{city}', state of '{state}' ({region} region).
        CRITICAL: Do NOT give the price per kg. You MUST give the price per 100 kg (quintal). For example, coffee is often ₹15000 to ₹35000 per quintal.
        Respond ONLY with a raw JSON object and no other markdown or backticks. Format: {{"production_cost": integer, "market_price": integer}}
        """
        raw_text = gemini_generate(prompt, api_key, timeout=2)  # 2s per model = stays fast
        if not raw_text:
            return None
        clean_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
        json_str = re.search(r'\{.*\}', clean_text, re.DOTALL).group()
        data = json.loads(json_str)
        _PRICE_CACHE[cache_key] = (float(data["production_cost"]), float(data["market_price"]))
        return _PRICE_CACHE[cache_key]
    except Exception as e:
        print(f"Gemini Economics Error: {e}")
        return None

def load_artifact(name):
    path = ARTIFACTS_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)

def load_all_models():
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE

    models = {
        "rf": load_artifact("model_rf.pkl"), "xgb": load_artifact("model_xgb.pkl"),
        "lgbm": load_artifact("model_lgbm.pkl"), "catboost": load_artifact("model_catboost.pkl"),
    }

    tabnet_path = ARTIFACTS_PATH / "model_tabnet.pkl"
    tabnet_model = joblib.load(tabnet_path) if tabnet_path.exists() else None
    
    ngb_path = ARTIFACTS_PATH / "model_ngboost.pkl"
    ngb_model = joblib.load(ngb_path) if ngb_path.exists() else None

    meta_model = load_artifact("stacked_model.pkl")
    label_encoder = load_artifact("label_encoder.pkl")
    imputer = joblib.load(ARTIFACTS_PATH / "imputer.pkl") if (ARTIFACTS_PATH / "imputer.pkl").exists() else None
    scaler = joblib.load(ARTIFACTS_PATH / "scaler.pkl") if (ARTIFACTS_PATH / "scaler.pkl").exists() else None

    _MODELS_CACHE = (models, tabnet_model, ngb_model, meta_model, label_encoder, imputer, scaler)
    return _MODELS_CACHE

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
        
        row = [[float(data[feature]) for feature in FEATURE_ORDER]]
    except KeyError as e:
        raise ValueError(f"Missing feature: {str(e)}")
    except ValueError:
        raise ValueError("All feature values must be numeric")
    return pd.DataFrame(row, columns=FEATURE_ORDER, dtype=float)

def normalize_region(region):
    if not region: return "central"
    return str(region).strip().lower()

def economics_for_crop(crop_name, region, state, city, market_price=0, production_cost=0):
    if market_price and production_cost:
        cost = float(production_cost)
        price = float(market_price)
    else:
        # Try Gemini API First (with all 4 models)
        gemini_data = fetch_gemini_economics(crop_name, city, state, region)
        if gemini_data:
            cost, price = gemini_data
        else:
            # Robust fallback: normalize crop name for lookup
            crop_key = crop_name.lower().strip().replace(" ", "")
            base = CROP_ECONOMICS.get(crop_key) or CROP_ECONOMICS.get(
                next((k for k in CROP_ECONOMICS if k in crop_key or crop_key in k), "default"),
                CROP_ECONOMICS["default"]
            )
            region_key = normalize_region(region)
            multiplier = REGION_MULTIPLIERS.get(region_key, REGION_MULTIPLIERS["central"])
            cost = float(base["cost"]) * float(multiplier["cost"])
            price = float(base["price"]) * float(multiplier["price"])

    profit = price - cost
    return round(cost, 2), round(price, 2), round(profit, 2)

def risk_level_from_score(score, confidence):
    if confidence >= 75.0 and score < 0.35: return "Low Risk"
    elif confidence >= 50.0 and score < 0.60: return "Moderate Risk"
    else: return "High Risk"

def predict_crop(input_data):
    models, tabnet_model, ngb_model, meta_model, label_encoder, imputer, scaler = load_all_models()
    X = build_input_array(input_data)

    if imputer is not None: X = pd.DataFrame(imputer.transform(X), columns=FEATURE_ORDER)
    if scaler is not None: X = pd.DataFrame(scaler.transform(X), columns=FEATURE_ORDER)
    
    base_probs = []
    stacked_arrays = []
    for model in models.values():
        prob = model.predict_proba(X)
        base_probs.append(prob[0])
        stacked_arrays.append(prob)
    if tabnet_model is not None:
        tab_prob = tabnet_model.predict_proba(X)
        base_probs.append(tab_prob[0])
        stacked_arrays.append(tab_prob)
        
    base_probs = np.asarray(base_probs)
    stacked_input = np.hstack(stacked_arrays)
    final_proba = meta_model.predict_proba(stacked_input)[0]
    
    mc_samples = None
    if tabnet_model is not None and hasattr(tabnet_model, "predict_proba_mc"):
        _, mc_samples = tabnet_model.predict_proba_mc(X, mc_samples=20, return_samples=True)

    mean_probs = base_probs.mean(axis=0)
    disagreement = float(np.mean(np.std(base_probs, axis=0)))
    entropy_score = float(entropy(mean_probs + 1e-12) / np.log(len(final_proba)))
    
    mc_score = 0.0
    if mc_samples is not None:
        mc_score = float(np.mean(np.std(mc_samples[:, 0, :], axis=0)))
        
    ngb_var = 0.0
    if ngb_model is not None:
        try:
            ngb_dist = ngb_model.pred_dist(X)
            ngb_probs = ngb_dist.params
            ngb_var = float(entropy(ngb_probs[0] + 1e-12) / np.log(len(ngb_probs[0])))
        except Exception:
            pass

    # Paper alignment: uncertainty incorporates NGBoost dispersion, model disagreement, and MC dropout
    risk_score = float(np.clip(0.35 * disagreement + 0.25 * entropy_score + 0.20 * mc_score + 0.20 * ngb_var, 0.0, 1.0))

    region = input_data.get("region", "Central")
    state = input_data.get("state", "Unknown")
    city = input_data.get("city", "Unknown")
    market_price = float(input_data.get("market_price", 0) or 0)
    production_cost = float(input_data.get("production_cost", 0) or 0)

    evaluated_crops = []
    for i, proba in enumerate(final_proba):
        crop_name = str(label_encoder.inverse_transform([i])[0])
        evaluated_crops.append({
            "name": crop_name,
            "ml_probability": float(proba)
        })

    # RANKING: Sorted PURELY by ML probability from the CSV-trained model.
    # Economics (from Gemini/Groq) only affects expected_profit display, NOT ranking.
    top_candidates = sorted(evaluated_crops, key=lambda x: x["ml_probability"], reverse=True)[:5]

    # 🚀 PARALLEL economics calls — 5 crops at once instead of sequential
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def _fetch_econ(crop):
        dynamic_cost, dynamic_price, estimated_profit = economics_for_crop(
            crop["name"], region, state, city, market_price=market_price, production_cost=production_cost
        )
        return crop["name"], dynamic_cost, dynamic_price, estimated_profit
    
    econ_results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_econ, crop): crop for crop in top_candidates}
        for future in as_completed(futures):
            try:
                name, cost, price, profit = future.result()
                econ_results[name] = (cost, price, profit)
            except Exception:
                crop = futures[future]
                econ_results[crop["name"]] = (0, 0, 0)

    scored_crops = []
    for crop in top_candidates:
        dynamic_cost, dynamic_price, estimated_profit = econ_results.get(
            crop["name"], (0, 0, 0)
        )
        conf = round(crop["ml_probability"] * 100, 2)  # % from CSV-trained model
        risk_penalty = 1.0 - (0.25 * risk_score)
        
        # Expected Profit formula:
        # (Price - Cost) × (Soil_Suitability% / 100) × (1 - 0.25 × Risk)
        exp_profit = estimated_profit * crop["ml_probability"] * risk_penalty
        
        # Growth Suitability: how suitable is this crop overall (soil + risk factored in)
        growth_suit = round(conf * (1.0 - 0.25 * risk_score), 2)
        
        scored_crops.append({
            "name": crop["name"],
            "confidence": conf,
            "dynamic_cost": dynamic_cost,
            "dynamic_price": dynamic_price,
            "estimated_profit": round(estimated_profit, 2),
            "final_economic_score": round(exp_profit, 2),
            "ml_probability": crop["ml_probability"],
            "growth_suitability": growth_suit
        })

    # Sort PURELY by ML probability — CSV data drives the recommendation
    ranked = sorted(scored_crops, key=lambda x: x["ml_probability"], reverse=True)
    top3 = ranked[:3]

    best_crop = top3[0]

    probs_by_crop = {
        str(label_encoder.inverse_transform([i])[0]): float(p)
        for i, p in enumerate(final_proba)
    }

    if best_crop["estimated_profit"] < 0:
        reason_text = f"WARNING: {best_crop['name'].title()} is selected, but current market rates in {city} suggest a potential loss."
    elif risk_score > 0.6:
        reason_text = f"CAUTION: {best_crop['name'].title()} is recommended based on economics, but soil conditions have high uncertainty."
    else:
        reason_text = f"SUCCESS: {best_crop['name'].title()} offers the optimal balance of biological yield and market profitability in {city}!"

    expected_profit = best_crop["final_economic_score"]

    return {
        "top1": best_crop["name"],
        "top3": top3,
        "confidence": best_crop["confidence"],
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level_from_score(risk_score, best_crop["confidence"]), 
        "growth_suitability": best_crop["growth_suitability"],
        "expected_profit": round(float(expected_profit), 2),
        "market_price": best_crop["dynamic_price"],
        "production_cost": best_crop["dynamic_cost"],
        "reason": reason_text,
        "all_probabilities": {k: round(v * 100, 2) for k, v in probs_by_crop.items()} 
    }