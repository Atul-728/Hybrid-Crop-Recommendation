import os
import joblib
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_PATH = os.path.join("ml", "artifacts")
CHARTS_PATH = os.path.join("static", "charts")

os.makedirs(CHARTS_PATH, exist_ok=True)

def load_data_and_model():
    model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_xgb.pkl"))
    X_train = joblib.load(os.path.join(ARTIFACTS_PATH, "X_train.pkl"))
    return model, X_train

def generate_shap_explanations():
    print("Generating SHAP Feature Importance Plot...")
    
    model, X_train = load_data_and_model()
    
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample of the training data
    sample_idx = np.random.choice(X_train.shape[0], 100, replace=False)
    X_sample = X_train[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    # FIX: Handle multi-class 3D array returned by newer XGBoost/SHAP versions
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Convert (samples, features, classes) to a list of 2D arrays
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    
    # Generate SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        
    plt.title("SHAP Feature Importance (Global Explainability)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, "shap_summary.png"))
    plt.close()

    print("SHAP explanation plot saved successfully.")

if __name__ == "__main__":
    generate_shap_explanations()