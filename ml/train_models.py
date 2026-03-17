import os
import joblib
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from sklearn.model_selection import cross_val_score
import torch

ARTIFACTS_PATH = os.path.join("ml", "artifacts")


# ----------------------------
# LOAD DATA
# ----------------------------
def load_data():
    X_train = joblib.load(os.path.join(ARTIFACTS_PATH, "X_train.pkl"))
    X_val = joblib.load(os.path.join(ARTIFACTS_PATH, "X_val.pkl"))
    X_test = joblib.load(os.path.join(ARTIFACTS_PATH, "X_test.pkl"))
    y_train = joblib.load(os.path.join(ARTIFACTS_PATH, "y_train.pkl"))
    y_val = joblib.load(os.path.join(ARTIFACTS_PATH, "y_val.pkl"))
    y_test = joblib.load(os.path.join(ARTIFACTS_PATH, "y_test.pkl"))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# EVALUATION
# ----------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")

    # ✅ CV only for sklearn-compatible models
    if model_name in ["tabnet", "ngboost"]:
        print(f"{model_name} CV skipped (not sklearn-compatible)")
        cv_mean = None
        cv_std = None
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

    # Test Prediction
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "cv_mean": cv_mean,
        "cv_std": cv_std
    }

    with open(os.path.join(ARTIFACTS_PATH, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{model_name} Accuracy:", accuracy)
    print(f"{model_name} CV Mean:", cv_mean)


# ----------------------------
# BASE MODELS
# ----------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_rf.pkl"))
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_xgb.pkl"))
    return model


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
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
# TABNET (WITH EARLY STOPPING)
# ----------------------------
def train_tabnet(X_train, y_train, X_val, y_val):
    print("Training TabNet...")
    model = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        verbose=0
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        patience=10,
        max_epochs=100
    )
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_tabnet.pkl"))
    return model


# ----------------------------
# NGBOOST
# ----------------------------
def train_ngboost(X_train, y_train):
    print("Training NGBoost...")
    k = len(np.unique(y_train))
    model = NGBClassifier(
        Dist=k_categorical(k),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(ARTIFACTS_PATH, "model_ngboost.pkl"))
    return model


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_training():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    rf = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_train, y_train, X_test, y_test, "rf")

    xgb = train_xgboost(X_train, y_train)
    evaluate_model(xgb, X_train, y_train, X_test, y_test, "xgb")

    lgbm = train_lightgbm(X_train, y_train)
    evaluate_model(lgbm, X_train, y_train, X_test, y_test, "lgbm")

    cat = train_catboost(X_train, y_train)
    evaluate_model(cat, X_train, y_train, X_test, y_test, "catboost")

    tabnet = train_tabnet(X_train, y_train, X_val, y_val)
    evaluate_model(tabnet, X_train, y_train, X_test, y_test, "tabnet")

    ngb = train_ngboost(X_train, y_train)
    evaluate_model(ngb, X_train, y_train, X_test, y_test, "ngboost")

    print("All models trained + evaluated successfully.")


if __name__ == "__main__":
    run_training()