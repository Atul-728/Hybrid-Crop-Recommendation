import json
import os

path = "ml/artifacts"

files = {
    "Random Forest": "rf_metrics.json",
    "XGBoost": "xgb_metrics.json",
    "LightGBM": "lgbm_metrics.json",
    "CatBoost": "catboost_metrics.json",
    "TabNet": "tabnet_metrics.json",
    "NGBoost": "ngboost_metrics.json",
    "Ensemble": "ensemble_metrics.json"
}

print("\nMODEL PERFORMANCE TABLE\n")

for name, file in files.items():
    with open(os.path.join(path, file)) as f:
        data = json.load(f)
        acc = round(data["accuracy"], 6)
        f1 = round(data["f1_score"], 6)
        print(f"{name:<15}  Accuracy: {acc}   F1: {f1}")