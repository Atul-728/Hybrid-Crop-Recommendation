import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARTIFACTS_PATH = os.path.join("ml", "artifacts")
CHARTS_PATH = os.path.join("static", "charts")

os.makedirs(CHARTS_PATH, exist_ok=True)


FEATURE_NAMES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


def load_model():
    model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_rf.pkl"))
    return model


def generate_feature_importance():
    print("Generating Feature Importance Plot...")

    model = load_model()

    importances = model.feature_importances_

    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh(
        np.array(FEATURE_NAMES)[sorted_idx],
        importances[sorted_idx]
    )

    plt.xlabel("Feature Importance Score")
    plt.title("RandomForest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, "feature_importance.png"))
    plt.close()

    print("Feature importance plot saved successfully.")


if __name__ == "__main__":
    generate_feature_importance()