import os
import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


ARTIFACTS_PATH = os.path.join("ml", "artifacts")


def load_data():
    X_train = joblib.load(os.path.join(ARTIFACTS_PATH, "X_train.pkl"))
    X_val = joblib.load(os.path.join(ARTIFACTS_PATH, "X_val.pkl"))
    X_test = joblib.load(os.path.join(ARTIFACTS_PATH, "X_test.pkl"))

    y_train = joblib.load(os.path.join(ARTIFACTS_PATH, "y_train.pkl"))
    y_val = joblib.load(os.path.join(ARTIFACTS_PATH, "y_val.pkl"))
    y_test = joblib.load(os.path.join(ARTIFACTS_PATH, "y_test.pkl"))

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": report
    }

    with open(os.path.join(ARTIFACTS_PATH, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{model_name} Accuracy:", accuracy)
    print(f"{model_name} F1 Score:", f1)

    return metrics


# ----------------------------
# Base Model Training
# ----------------------------

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_rf.pkl"))

    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_xgb.pkl"))

    return model


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_lgbm.pkl"))

    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )

    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_catboost.pkl"))

    return model


# ----------------------------
# Main Training Pipeline
# ----------------------------

def run_training():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "rf")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "xgb")

    # LightGBM
    lgbm_model = train_lightgbm(X_train, y_train)
    evaluate_model(lgbm_model, X_test, y_test, "lgbm")

    # CatBoost
    cat_model = train_catboost(X_train, y_train)
    evaluate_model(cat_model, X_test, y_test, "catboost")

    print("All base models trained successfully.")


if __name__ == "__main__":
    run_training()