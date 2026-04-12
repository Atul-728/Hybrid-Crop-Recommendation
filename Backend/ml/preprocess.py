from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "crop_recommendation.csv"
ARTIFACTS_PATH = BASE_DIR / "ml" / "artifacts"
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
# Derived features to align with paper
DERIVED_FEATURES = ["seasonal_index", "npk_ratio", "humidity_rainfall_interaction"]
TARGET = "label"

def add_derived_features(df):
    df_new = df.copy()
    # Seasonal index as a proxy for climate suitability
    df_new["seasonal_index"] = (df_new["temperature"] * df_new["rainfall"]) / 100.0
    # NPK ratio for nutrient balance
    df_new["npk_ratio"] = df_new["N"] / (df_new["P"] + df_new["K"] + 1.0)
    # Humidity and rainfall interaction
    df_new["humidity_rainfall_interaction"] = (df_new["humidity"] * df_new["rainfall"]) / 1000.0
    return df_new

def run_preprocessing():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    
    # 0. Add derived features before splitting
    df = add_derived_features(df)
    
    ALL_FEATURES = FEATURES + DERIVED_FEATURES
    df = df[ALL_FEATURES + [TARGET]].copy()

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df[ALL_FEATURES].astype(float)

    # 1. SPLIT FIRST (70% Train, 15% Val, 15% Test)
    print("Splitting data (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 2. ITERATIVE IMPUTATION (Fit on Train only)
    print("Applying Iterative Imputer...")
    imputer = IterativeImputer(random_state=42)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=ALL_FEATURES)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=ALL_FEATURES)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=ALL_FEATURES)

    # 3. IQR OUTLIER REMOVAL (Train only to prevent leakage)
    print("Removing Outliers (IQR per class)...")
    train_df = pd.concat([X_train_imputed, pd.Series(y_train, name=TARGET).reset_index(drop=True)], axis=1)
    
    clean_dfs = []
    for label_val in train_df[TARGET].unique():
        class_df = train_df[train_df[TARGET] == label_val].copy()
        for col in ALL_FEATURES:
            Q1 = class_df[col].quantile(0.25)
            Q3 = class_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Sirf tabhi filter lagao jab IQR > 0 ho, taaki data safe rahe
            if IQR > 0:
                class_df = class_df[(class_df[col] >= Q1 - 1.5 * IQR) & (class_df[col] <= Q3 + 1.5 * IQR)]
        
        # Agar saare rows delete ho gaye galti se, toh original class_df wapas daal do
        if len(class_df) == 0:
            class_df = train_df[train_df[TARGET] == label_val]
            
        clean_dfs.append(class_df)
    
    train_df = pd.concat(clean_dfs, ignore_index=True)
    
    X_train_clean = train_df.drop(columns=[TARGET])
    y_train_clean = train_df[TARGET].values

    # 4. SCALING (Fit on Train only)
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 5. SMOTE CLASS BALANCING (Train only)
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_clean)

    print("Saving artifacts...")
    joblib.dump(le, ARTIFACTS_PATH / "label_encoder.pkl")
    joblib.dump(imputer, ARTIFACTS_PATH / "imputer.pkl")
    joblib.dump(scaler, ARTIFACTS_PATH / "scaler.pkl")
    joblib.dump(X_train_res, ARTIFACTS_PATH / "X_train.pkl")
    joblib.dump(X_val_scaled, ARTIFACTS_PATH / "X_val.pkl")
    joblib.dump(X_test_scaled, ARTIFACTS_PATH / "X_test.pkl")
    joblib.dump(y_train_res, ARTIFACTS_PATH / "y_train.pkl")
    joblib.dump(y_val, ARTIFACTS_PATH / "y_val.pkl")
    joblib.dump(y_test, ARTIFACTS_PATH / "y_test.pkl")

    with open(ARTIFACTS_PATH / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(ALL_FEATURES, f, indent=2)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()