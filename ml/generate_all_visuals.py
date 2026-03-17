import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

def generate_all_real_visualizations():
    print("🚀 Starting REAL ML visualization generation (TEST DATA)...")
    
    # Paths
    ARTIFACTS_PATH = os.path.join("ml", "artifacts")
    STATIC_DIR = "static"
    DATA_PATH = os.path.join("data", "crop_recommendation.csv")
    os.makedirs(STATIC_DIR, exist_ok=True)

    print("Loading models and TEST data...")

    # Load test data
    X_test = joblib.load(os.path.join(ARTIFACTS_PATH, "X_test.pkl"))
    y_test = joblib.load(os.path.join(ARTIFACTS_PATH, "y_test.pkl"))

    le = joblib.load(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))

    # Load models
    models = {
        "Random Forest": joblib.load(os.path.join(ARTIFACTS_PATH, "model_rf.pkl")),
        "XGBoost": joblib.load(os.path.join(ARTIFACTS_PATH, "model_xgb.pkl")),
        "LightGBM": joblib.load(os.path.join(ARTIFACTS_PATH, "model_lgbm.pkl")),
        "CatBoost": joblib.load(os.path.join(ARTIFACTS_PATH, "model_catboost.pkl")),
        "TabNet": joblib.load(os.path.join(ARTIFACTS_PATH, "model_tabnet.pkl"))
    }

    # 👇 NGBoost alag rakho (sirf table ke liye)
    ngboost_model = joblib.load(os.path.join(ARTIFACTS_PATH, "model_ngboost.pkl"))

    meta_model = joblib.load(os.path.join(ARTIFACTS_PATH, "stacked_model.pkl"))

    print("Calculating test accuracies...")

    accuracies = {}
    meta_features = []

    # ----------------------------
    # Base Models
    # ----------------------------
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = round(acc * 100, 2)

        if hasattr(model, "predict_proba"):
            meta_features.append(model.predict_proba(X_test))

    # ----------------------------
    # 🔥 ENSEMBLE PREDICTION (IMPORTANT)
    # ----------------------------
    X_meta = np.hstack(meta_features)
    y_pred_meta = meta_model.predict(X_meta)

    stacked_acc = accuracy_score(y_test, y_pred_meta)
    accuracies["Ensemble"] = round(stacked_acc * 100, 2)

    # =========================================================
    # 🔥 MODEL PERFORMANCE TABLE (NEW)
    # =========================================================
    print("\nMODEL PERFORMANCE TABLE\n")

    table_data = []

    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"{name:<15} Accuracy: {round(acc,6)}   F1: {round(f1,6)}")
        table_data.append([name, round(acc,6), round(f1,6)])

    # Ensemble
    f1_meta = f1_score(y_test, y_pred_meta, average="weighted")

    print(f"{'Ensemble':<15} Accuracy: {round(stacked_acc,6)}   F1: {round(f1_meta,6)}")
    table_data.append(["Ensemble", round(stacked_acc,6), round(f1_meta,6)])

    # Save CSV
    df_table = pd.DataFrame(table_data, columns=["Model", "Accuracy", "F1 Score"])
    df_table.to_csv(os.path.join(STATIC_DIR, "model_performance_table.csv"), index=False)

    # ---------------------------------------------------------
    # 1. MODEL COMPARISON GRAPH
    # ---------------------------------------------------------
    print("\n1/5 Generating Model Comparison Graph...")

    plt.figure(figsize=(12, 6))
    model_names = list(accuracies.keys())
    model_accs = list(accuracies.values())

    sns.set_style("whitegrid")
    ax = sns.barplot(x=model_names, y=model_accs, hue=model_names, palette="viridis", legend=False)

    plt.ylim(min(model_accs) - 2, 100.5)
    plt.title("Algorithm Performance Comparison (Test Data)", fontsize=18, weight="bold")
    plt.ylabel("Accuracy (%)")

    for i, v in enumerate(model_accs):
        ax.text(i, v + 0.1, f"{v}%", ha='center', weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. CONFUSION MATRIX (ENSEMBLE)
    # ---------------------------------------------------------
    print("2/5 Generating Confusion Matrix...")

    cat_model = models["CatBoost"]
    y_pred_cat = cat_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_cat)
    class_names = le.inverse_transform(np.arange(len(le.classes_)))

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=False)

    plt.title("Confusion Matrix of CatBoost Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. CATBOOST CLASSIFICATION REPORT
    # ---------------------------------------------------------
    print("3/5 Generating CatBoost Classification Report...")

    cat_model = models["CatBoost"]
    y_pred_cat = cat_model.predict(X_test)

    report_dict = classification_report(y_test, y_pred_cat, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    report_df.index = [
        le.inverse_transform([int(i)])[0] if i.isdigit() else i
        for i in report_df.index
    ]

    report_df.to_csv(os.path.join(STATIC_DIR, "catboost_classification_report.csv"))

    print("\n📊 CatBoost Classification Report:")
    print(report_df)

    # ---------------------------------------------------------
    # 4. CORRELATION HEATMAP
    # ---------------------------------------------------------
    print("4/5 Generating Correlation Heatmap...")

    df = pd.read_csv(DATA_PATH)
    numeric_df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 5. CLASS DISTRIBUTION
    # ---------------------------------------------------------
    print("5/5 Generating Class Distribution...")

    counts = df["label"].value_counts()

    plt.figure(figsize=(14, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45)

    plt.title("Crop Dataset Class Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    print("\n✅ ALL DONE! Everything is now consistent.")


if __name__ == "__main__":
    generate_all_real_visualizations()