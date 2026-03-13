import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


DATA_PATH = os.path.join("data", "crop_recommendation.csv")
ARTIFACTS_PATH = os.path.join("ml", "artifacts")

os.makedirs(ARTIFACTS_PATH, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


def handle_missing_values(df):
    df = df.dropna()
    return df


def remove_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


def encode_labels(df):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])
    
    joblib.dump(le, os.path.join(ARTIFACTS_PATH, "label_encoder.pkl"))
    
    return df, le


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    
    return X_scaled, scaler


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    joblib.dump(X_train, os.path.join(ARTIFACTS_PATH, "X_train.pkl"))
    joblib.dump(X_val, os.path.join(ARTIFACTS_PATH, "X_val.pkl"))
    joblib.dump(X_test, os.path.join(ARTIFACTS_PATH, "X_test.pkl"))
    
    joblib.dump(y_train, os.path.join(ARTIFACTS_PATH, "y_train.pkl"))
    joblib.dump(y_val, os.path.join(ARTIFACTS_PATH, "y_val.pkl"))
    joblib.dump(y_test, os.path.join(ARTIFACTS_PATH, "y_test.pkl"))


def run_preprocessing():
    print("Loading dataset...")
    df = load_data()
    
    print("Removing duplicates...")
    df = remove_duplicates(df)
    
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Removing outliers using IQR...")
    df = remove_outliers_iqr(df)
    
    print("Encoding labels...")
    df, le = encode_labels(df)
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    print("Scaling features...")
    X_scaled, scaler = scale_features(X)
    
    print("Splitting dataset (70-15-15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)
    
    print("Saving processed datasets...")
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    run_preprocessing()