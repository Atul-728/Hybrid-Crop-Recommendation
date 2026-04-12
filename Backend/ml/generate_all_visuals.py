from pathlib import Path
import json
import joblib
import numpy as np
import os

try:
    from . import tabnet_surrogate as _tabnet_surrogate_compat
except ImportError:
    try:
        import tabnet_surrogate as _tabnet_surrogate_compat  # noqa: F401
    except ImportError:
        _tabnet_surrogate_compat = None

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
try:
    from .tabnet_surrogate import TabNetLiteClassifier
except ImportError:
    from tabnet_surrogate import TabNetLiteClassifier

FEATURES = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "seasonal_index", "npk_ratio", "humidity_rainfall_interaction"
]


def get_project_paths():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    artifacts_path = project_root / "Backend" / "ml" / "artifacts"
    data_path = project_root / "Backend" / "data" / "crop_recommendation.csv"
    charts_path = project_root / "Frontend" / "static" / "charts"

    charts_path.mkdir(parents=True, exist_ok=True)

    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return artifacts_path, data_path, charts_path


def load_artifact(artifacts_path, name):
    path = artifacts_path / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def ensure_numpy(x):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def load_data_and_models(artifacts_path):
    X_test = ensure_numpy(load_artifact(artifacts_path, "X_test.pkl"))
    y_test = ensure_numpy(load_artifact(artifacts_path, "y_test.pkl"))

    label_encoder_path = artifacts_path / "label_encoder.pkl"
    if label_encoder_path.exists():
        label_encoder = joblib.load(label_encoder_path)
        class_names = list(label_encoder.classes_)
    else:
        class_names = []

    # Load models
    models = {
        "Random Forest": joblib.load(artifacts_path / "model_rf.pkl"),
        "XGBoost": joblib.load(artifacts_path / "model_xgb.pkl"),
        "LightGBM": joblib.load(artifacts_path / "model_lgbm.pkl"),
        "CatBoost": joblib.load(artifacts_path / "model_catboost.pkl"),
        "TabNet": joblib.load(artifacts_path / "model_tabnet.pkl")
    }

    stacked_model_path = artifacts_path / "stacked_model.pkl"
    stacked_model = joblib.load(stacked_model_path) if stacked_model_path.exists() else None

    return X_test, y_test, class_names, models, stacked_model


def build_meta_features(models, X):
    parts = []
    for model in models.values():
        parts.append(model.predict_proba(X))
    return np.hstack(parts)


def plot_bar_comparison(names, values, output_path):
    colors = [
        "#5B4B8A",
        "#4E6A8E",
        "#3F7C85",
        "#379683",
        "#5FA777",
        "#95C04F",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=colors[:len(names)])

    ax.set_ylim(min(values) - 1, 100)
    ax.set_title("Algorithm Performance Comparison (Test Data)", fontsize=14)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.1,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    cm = confusion_matrix(y_true, y_pred)

    if not class_names:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    fig.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(data_path, output_path):
    df = pd.read_csv(data_path)
    
    # Generate derived features so they exist in df before we slice FEATURES
    df["seasonal_index"] = (df["temperature"] * df["rainfall"]) / 100.0
    df["npk_ratio"] = df["N"] / (df["P"] + df["K"] + 1.0)
    df["humidity_rainfall_interaction"] = (df["humidity"] * df["rainfall"]) / 1000.0

    corr = df[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values)
    ax.set_title("Feature Correlation Heatmap")

    ticks = np.arange(len(FEATURES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURES)

    fig.colorbar(im, ax=ax)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(data_path, output_path):
    df = pd.read_csv(data_path)
    counts = df["label"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(counts.index, counts.values)
    ax.set_title("Crop Distribution")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Crop")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(y_true, y_pred, class_names, output_path):
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, output_dict=True)

    pd.DataFrame(report).transpose().to_csv(output_path)


def generate_table_image(rows, output_path):
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2 + len(df)*0.5))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#D3D3D3")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_visualizations():
    artifacts_path, data_path, charts_path = get_project_paths()

    X_test, y_test, class_names, models, stacked_model = load_data_and_models(artifacts_path)

    rows = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average="weighted")
        })

    if stacked_model is not None:
        X_meta = build_meta_features(models, X_test)
        y_pred = stacked_model.predict(X_meta)
        predictions["Ensemble"] = y_pred
        rows.append({
            "Model": "Ensemble",
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average="weighted")
        })
        
        # 🔥 PRINTING THE ENSEMBLE CLASSIFICATION REPORT FOR TABLE 3
        print("\n" + "="*60)
        print(" HYBRID ENSEMBLE CLASSIFICATION REPORT (COPY THIS FOR TABLE 3)")
        print("="*60)
        if class_names:
            report_str = classification_report(y_test, predictions["Ensemble"], target_names=class_names, digits=4)
        else:
            report_str = classification_report(y_test, predictions["Ensemble"], digits=4)
        print(report_str)
        print("="*60 + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(charts_path / "model_performance_table.csv", index=False)

    print("\nModel Performance Table:\n")
    print(df.to_string(index=False))

    generate_table_image(rows, charts_path / "model_performance_table.png")

    plot_bar_comparison(
        [r["Model"] for r in rows],
        [r["Accuracy"] * 100 for r in rows],
        charts_path / "model_comparison.png"
    )

    best_model_name = max(rows, key=lambda x: x["Accuracy"])["Model"]
    best_pred = predictions[best_model_name]

    plot_confusion_matrix(
        y_test,
        best_pred,
        class_names,
        charts_path / "confusion_matrix.png",
        f"{best_model_name} Confusion Matrix"
    )

    if "Ensemble" in predictions:
        plot_confusion_matrix(
            y_test,
            predictions["Ensemble"],
            class_names,
            charts_path / "ensemble_confusion_matrix.png",
            "Hybrid Ensemble Confusion Matrix"
        )
        save_classification_report(
            y_test,
            predictions["Ensemble"],
            class_names,
            charts_path / "ensemble_classification_report.csv"
        )

    save_classification_report(
        y_test,
        predictions["CatBoost"],
        class_names,
        charts_path / "catboost_classification_report.csv"
    )

    plot_correlation_heatmap(data_path, charts_path / "correlation_heatmap.png")
    plot_class_distribution(data_path, charts_path / "class_distribution.png")

    with open(artifacts_path / "visuals_summary.json", "w") as f:
        json.dump({"charts_path": str(charts_path)}, f)


if __name__ == "__main__":
    generate_all_visualizations()