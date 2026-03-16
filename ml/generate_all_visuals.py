import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Force non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def generate_all_real_visualizations():
    print("🚀 Starting real ML visualization generation...")
    
    # Paths
    ARTIFACTS_PATH = os.path.join("ml", "artifacts")
    STATIC_DIR = "static"
    DATA_PATH = os.path.join("data", "crop_recommendation.csv")
    os.makedirs(STATIC_DIR, exist_ok=True)

    print("Loading data, models, and encoders...")
    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    le = joblib.load(os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))
    te = joblib.load(os.path.join(ARTIFACTS_PATH, "target_encoder.pkl"))
    meta_model = joblib.load(os.path.join(ARTIFACTS_PATH, "stacked_model.pkl"))
    
    models = {
        "Random Forest": joblib.load(os.path.join(ARTIFACTS_PATH, "model_rf.pkl")),
        "XGBoost": joblib.load(os.path.join(ARTIFACTS_PATH, "model_xgb.pkl")),
        "LightGBM": joblib.load(os.path.join(ARTIFACTS_PATH, "model_lgbm.pkl")),
        "CatBoost": joblib.load(os.path.join(ARTIFACTS_PATH, "model_catboost.pkl")),
        "TabNet": joblib.load(os.path.join(ARTIFACTS_PATH, "model_tabnet.pkl"))
    }

    # FIX: Filter out any crops from the CSV that the LabelEncoder hasn't seen during training
    known_labels = set(le.classes_)
    df = df[df['label'].isin(known_labels)].copy()
    
    # Extract Actual Labels from the filtered dataset
    y_true_labels = df['label'].values
    y_true = le.transform(y_true_labels)

    print("Preparing 14-feature input array from dataset...")
    X = []
    for _, row in df.iterrows():
        N = row.get('N', 50)
        P = row.get('P', 50)
        K = row.get('K', 50)
        temp = row.get('temperature', 25)
        hum = row.get('humidity', 60)
        ph = row.get('ph', 6.5)
        rain = row.get('rainfall', 100)

        month = row.get('month', 6)
        seasonal_index = row.get('seasonal_index', np.sin(2 * np.pi * month / 12))
        market_price = row.get('market_price', 2000)
        production_cost = row.get('production_cost', 1000)
        profit_ratio = (market_price - production_cost) / production_cost if production_cost > 0 else 0
        
        region = row.get('region', 'North')
        try:
            region_enc = int(te.transform(pd.DataFrame({'region': [region]}))['region'].values[0])
        except Exception:
            region_enc = 0

        X.append([
            N, P, K, temp, hum, ph, rain, month, seasonal_index, 
            rain, market_price, production_cost, profit_ratio, region_enc
        ])

    X = np.array(X)
    X_scaled = scaler.transform(X)

    print("Running predictions to calculate REAL accuracies...")
    accuracies = {}
    meta_features = []
    
    for name, m in models.items():
        probs = m.predict_proba(X_scaled)
        meta_features.append(probs)
        # Getting actual predictions for base models
        preds = np.argmax(probs, axis=1) 
        accuracies[name] = round(accuracy_score(y_true, preds) * 100, 2)

    X_meta = np.hstack(meta_features)
    y_pred_meta = meta_model.predict(X_meta)
    
    # Stacked Model Accuracy
    stacked_acc = round(accuracy_score(y_true, y_pred_meta) * 100, 2)
    accuracies["Stacked Model"] = stacked_acc

    # ---------------------------------------------------------
    # 1. REAL MODEL COMPARISON GRAPH
    # ---------------------------------------------------------
    print("1/4 Generating Model Comparison Graph...")
    plt.figure(figsize=(12, 6))
    model_names = list(accuracies.keys())
    model_accs = list(accuracies.values())
    
    sns.set_style("whitegrid")
    ax = sns.barplot(x=model_names, y=model_accs, hue=model_names, palette="viridis", legend=False)
    plt.ylim(min(model_accs) - 2, 100.5) # Dynamic Y-axis based on actual lowest accuracy
    plt.title("Algorithm Performance Comparison (Actual Data)", fontsize=18, weight="bold", pad=20)
    plt.ylabel("Accuracy (%)", fontsize=12, weight="bold")
    
    for i, v in enumerate(model_accs):
        ax.text(i, v + 0.1, f"{v}%", ha='center', weight='bold', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=300, transparent=True)
    plt.close()

    # ---------------------------------------------------------
    # 2. REAL CONFUSION MATRIX
    # ---------------------------------------------------------
    print("2/4 Generating Real Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred_meta)
    class_names = le.inverse_transform(np.arange(len(le.classes_)))

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 10, "weight": "bold"})
    plt.title("Actual Stacked Model Confusion Matrix", fontsize=20, weight='bold', pad=20, color="#1976d2")
    plt.xlabel("Predicted Crop", fontsize=14, weight='bold', labelpad=15)
    plt.ylabel("Actual Crop", fontsize=14, weight='bold', labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=300, transparent=False, facecolor='white')
    plt.close()

    # ---------------------------------------------------------
    # 3. CORRELATION HEATMAP (From Dataset)
    # ---------------------------------------------------------
    print("3/4 Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    numeric_df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
    plt.title("Feature Correlation Heatmap", fontsize=18, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "correlation_heatmap.png"), dpi=300, transparent=True)
    plt.close()

    # ---------------------------------------------------------
    # 4. CLASS DISTRIBUTION BAR GRAPH (From Dataset)
    # ---------------------------------------------------------
    print("4/4 Generating Class Distribution Graph...")
    counts = df["label"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(14, 6))
    colors = sns.color_palette("mako", len(counts))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors, hue=counts.index, legend=False)
    plt.title("Crop Dataset Class Distribution", fontsize=18, weight="bold", color="#2e7d32")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "class_distribution.png"), dpi=300, transparent=True)
    plt.close()

    print("\n✅ ALL DONE! All 4 real charts have been saved to the 'static' folder.")

if __name__ == "__main__":
    generate_all_real_visualizations()