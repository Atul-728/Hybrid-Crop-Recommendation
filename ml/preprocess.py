import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
import joblib

DATA_PATH = os.path.join("data", "crop_recommendation.csv")
ARTIFACTS_PATH = os.path.join("ml", "artifacts")

os.makedirs(ARTIFACTS_PATH, exist_ok=True)

def load_and_synthesize_data():
    df = pd.read_csv(DATA_PATH)
    
    # Synthesize Missing Paper Features (Geo, Temporal, Economic)
    np.random.seed(42)
    regions = ["North", "South", "East", "West", "Central"]
    df["region"] = np.random.choice(regions, size=len(df))
    
    # Random dates for seasonal index
    start_date = datetime(2023, 1, 1)
    df["date"] = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(len(df))]
    df["month"] = df["date"].dt.month
    df["seasonal_index"] = np.sin(2 * np.pi * df["month"] / 12)
    
    # Synthesize rolling rainfall & market prices
    df["rolling_rainfall"] = df["rainfall"].rolling(window=3, min_periods=1).mean()
    df["market_price"] = np.random.uniform(1000, 5000, size=len(df))
    df["production_cost"] = np.random.uniform(500, 2000, size=len(df))
    df["profitability_ratio"] = (df["market_price"] - df["production_cost"]) / df["production_cost"]
    
    df = df.drop(columns=["date"])
    return df

def handle_missing_and_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = IterativeImputer(random_state=42)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

def encode_features(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    joblib.dump(le, os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))
    
    # Target Encoding for Region
    te = TargetEncoder()
    df["region_encoded"] = te.fit_transform(df["region"], df["label"])
    joblib.dump(te, os.path.join(ARTIFACTS_PATH, "target_encoder.pkl"))
    
    df = df.drop(columns=["region"])
    return df, le

def split_and_scale(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)  # fit on array to avoid feature name warnings
    joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def run_preprocessing():
    print("Loading and synthesizing data...")
    df = load_and_synthesize_data()
    df = df.drop_duplicates()
    
    print("Handling missing values and outliers...")
    df = handle_missing_and_outliers(df)
    
    print("Encoding features (Target Encoding)...")
    df, le = encode_features(df)
    
    print("Scaling and splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(df)
    
    print("Applying SMOTE...")
    X_train, y_train = apply_smote(X_train, y_train)
    
    joblib.dump(X_train, os.path.join(ARTIFACTS_PATH, "X_train.pkl"))
    joblib.dump(X_val, os.path.join(ARTIFACTS_PATH, "X_val.pkl"))
    joblib.dump(X_test, os.path.join(ARTIFACTS_PATH, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(ARTIFACTS_PATH, "y_train.pkl"))
    joblib.dump(y_val, os.path.join(ARTIFACTS_PATH, "y_val.pkl"))
    joblib.dump(y_test, os.path.join(ARTIFACTS_PATH, "y_test.pkl"))
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    run_preprocessing()