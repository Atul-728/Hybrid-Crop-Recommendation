import os
import joblib
import json


ARTIFACTS_PATH = os.path.join("ml", "artifacts")


# ----------------------------
# Load Metrics
# ----------------------------

def load_metrics():
    metric_files = [
        "rf_metrics.json",
        "xgb_metrics.json",
        "lgbm_metrics.json",
        "catboost_metrics.json",
        "ensemble_metrics.json"
    ]

    metrics = {}

    for file in metric_files:
        path = os.path.join(ARTIFACTS_PATH, file)
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics[file] = json.load(f)

    return metrics


# ----------------------------
# Select Best Model
# ----------------------------

def select_best_model(metrics):
    best_model = None
    best_accuracy = 0

    for file_name, metric in metrics.items():
        accuracy = metric.get("accuracy", 0)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = file_name.replace("_metrics.json", "")

    print("Best Model Selected:", best_model)
    print("Best Accuracy:", best_accuracy)

    return best_model


# ----------------------------
# Create Production Bundle
# ----------------------------

def create_production_bundle(best_model_name):
    scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))

    if best_model_name == "ensemble":
        model = joblib.load(os.path.join(ARTIFACTS_PATH, "stacked_model.pkl"))
    else:
        model = joblib.load(os.path.join(ARTIFACTS_PATH, f"model_{best_model_name}.pkl"))

    production_bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "model_name": best_model_name
    }

    joblib.dump(production_bundle, os.path.join(ARTIFACTS_PATH, "production_bundle.pkl"))

    print("Production bundle saved successfully.")


# ----------------------------
# Run Full Save Process
# ----------------------------

def run_save_pipeline():
    metrics = load_metrics()
    best_model_name = select_best_model(metrics)
    create_production_bundle(best_model_name)


if __name__ == "__main__":
    run_save_pipeline()