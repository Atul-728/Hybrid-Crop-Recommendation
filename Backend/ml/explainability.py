from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent.parent
ARTIFACTS_PATH = ML_DIR / "artifacts"
CHARTS_PATH = PROJECT_ROOT / "Frontend" / "static" / "charts"
CHARTS_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "seasonal_index", "npk_ratio", "humidity_rainfall_interaction"
]

OUTPUT_IMAGE = CHARTS_PATH / "shap_summary.png"
MAX_CLASSES_TO_SHOW = 8


def load_artifact(name):
    path = ARTIFACTS_PATH / name
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def choose_model():
    preferred_models = [
        "model_catboost.pkl",
        "model_xgb.pkl",
        "model_lgbm.pkl",
        "model_rf.pkl",
    ]

    for model_name in preferred_models:
        model_path = ARTIFACTS_PATH / model_name
        if model_path.exists():
            return joblib.load(model_path), model_name

    raise FileNotFoundError("No trained model file found in the artifacts folder.")


def load_training_data():
    x_train = load_artifact("X_train.pkl")

    if isinstance(x_train, pd.DataFrame):
        df = x_train.copy()
    else:
        df = pd.DataFrame(x_train, columns=FEATURES)

    return df


def get_label_names():
    label_encoder_path = ARTIFACTS_PATH / "label_encoder.pkl"
    if label_encoder_path.exists():
        encoder = joblib.load(label_encoder_path)
        return list(encoder.classes_)
    return []


def sample_data(df, sample_size=150):
    if len(df) <= sample_size:
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)


def normalize_shap_values(shap_values, n_samples, n_features):
    if isinstance(shap_values, list):
        return np.stack([np.asarray(v) for v in shap_values], axis=0)

    arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return arr[None, :, :]

    if arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return np.transpose(arr, (2, 0, 1))
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            return arr
        if arr.shape[0] == n_features and arr.shape[1] == n_samples:
            return np.transpose(arr, (2, 1, 0))

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def reduce_classes(shap_array, class_names, max_classes=8):
    class_importance = np.mean(np.abs(shap_array), axis=(1, 2))
    order = np.argsort(class_importance)[::-1]

    if len(order) <= max_classes:
        selected_idx = order
        selected_names = [
            class_names[i] if i < len(class_names) else f"Crop {i + 1}"
            for i in selected_idx
        ]
        return shap_array[selected_idx], selected_names

    selected_idx = order[: max_classes - 1]
    other_idx = order[max_classes - 1 :]

    selected_names = [
        class_names[i] if i < len(class_names) else f"Crop {i + 1}"
        for i in selected_idx
    ]
    selected_names.append("Other crops")

    selected_values = shap_array[selected_idx]
    other_values = shap_array[other_idx].sum(axis=0, keepdims=True)

    combined = np.concatenate([selected_values, other_values], axis=0)
    return combined, selected_names


def plot_clean_stacked_summary(shap_array, feature_names, class_names, output_path):
    feature_importance = np.mean(np.abs(shap_array), axis=(0, 1))
    feature_order = np.argsort(feature_importance)[::-1]

    sorted_features = np.array(feature_names)[feature_order]
    shap_array = shap_array[:, :, feature_order]

    class_count = shap_array.shape[0]
    colors = get_cmap("tab20").colors if class_count <= 20 else get_cmap("hsv")(np.linspace(0, 1, class_count))

    fig, ax = plt.subplots(figsize=(15, 9))

    left = np.zeros(len(sorted_features))

    for i in range(class_count):
        values = np.mean(np.abs(shap_array[i]), axis=0)
        ax.barh(
            sorted_features,
            values,
            left=left,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.6,
            label=class_names[i]
        )
        left += values

    ax.set_title("SHAP Feature Importance by Crop", fontsize=16, pad=12)
    ax.set_xlabel("Average impact on prediction", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Crops",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        title_fontsize=10,
        frameon=False
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def generate_shap_visual():
    model, model_name = choose_model()
    x_train = load_training_data()
    class_names = get_label_names()

    x_sample = sample_data(x_train, sample_size=150)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    shap_array = normalize_shap_values(
        shap_values,
        n_samples=x_sample.shape[0],
        n_features=x_sample.shape[1]
    )

    shap_array, selected_class_names = reduce_classes(
        shap_array,
        class_names,
        max_classes=MAX_CLASSES_TO_SHOW
    )

    plot_clean_stacked_summary(
        shap_array=shap_array,
        feature_names=FEATURES,
        class_names=selected_class_names,
        output_path=OUTPUT_IMAGE
    )

    info_path = ARTIFACTS_PATH / "shap_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Model used: {model_name}\n")
        f.write(f"Output image: {OUTPUT_IMAGE}\n")
        f.write("Displayed crops:\n")
        for name in selected_class_names:
            f.write(f"{name}\n")

    print(f"Saved SHAP image to: {OUTPUT_IMAGE}")
    print(f"Saved info file to: {info_path}")


if __name__ == "__main__":
    generate_shap_visual()