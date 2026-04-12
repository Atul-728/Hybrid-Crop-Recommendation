from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from ngboost.distns import k_categorical

try:
    from .tabnet_surrogate import TabNetLiteClassifier
except ImportError:
    from tabnet_surrogate import TabNetLiteClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = BASE_DIR / "ml" / "artifacts"


def load_data():
    X_train = joblib.load(ARTIFACTS_PATH / "X_train.pkl")
    X_val = joblib.load(ARTIFACTS_PATH / "X_val.pkl")
    X_test = joblib.load(ARTIFACTS_PATH / "X_test.pkl")
    y_train = joblib.load(ARTIFACTS_PATH / "y_train.pkl")
    y_val = joblib.load(ARTIFACTS_PATH / "y_val.pkl")
    y_test = joblib.load(ARTIFACTS_PATH / "y_test.pkl")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_label_count():
    y_train = joblib.load(ARTIFACTS_PATH / "y_train.pkl")
    return int(np.max(y_train)) + 1


def save_metrics(name, model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    with open(ARTIFACTS_PATH / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    return metrics


def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, ARTIFACTS_PATH / "model_rf.pkl")
    return model


def train_xgboost(X_train, y_train, X_val, y_val, n_classes):
    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.5,
        min_child_weight=2,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model, ARTIFACTS_PATH / "model_xgb.pkl")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, n_classes):
    print("Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        objective="multiclass",
        num_class=n_classes,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    joblib.dump(model, ARTIFACTS_PATH / "model_lgbm.pkl")
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    print("Training CatBoost...")
    model = CatBoostClassifier(
        iterations=220,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=6,
        loss_function="MultiClass",
        random_seed=42,
        verbose=0,
        od_type="Iter",
        od_wait=25,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    joblib.dump(model, ARTIFACTS_PATH / "model_catboost.pkl")
    return model


def train_tabnet(X_train, y_train, X_val, y_val, n_classes):
    print("Training TabNet...")
    model = TabNetLiteClassifier(
        input_dim=X_train.shape[1],
        n_classes=n_classes,
        hidden_dim=32,
        n_steps=2,
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=200,
        patience=3,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    joblib.dump(model, ARTIFACTS_PATH / "model_tabnet.pkl")
    return model

# 🔥 FIX: NGBoost Added Back for Uncertainty Claim
def train_ngboost(X_train, y_train):
    print("Training NGBoost (For Uncertainty Quantification)...")
    n_classes = len(np.unique(y_train))
    model = NGBClassifier(
        Dist=k_categorical(n_classes),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    joblib.dump(model, ARTIFACTS_PATH / "model_ngboost.pkl")
    return model


def model_factories(n_classes, input_dim):
    return {
        "rf": lambda: RandomForestClassifier(n_estimators=180, max_depth=10, min_samples_split=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "xgb": lambda: XGBClassifier(n_estimators=220, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, min_child_weight=2, objective="multi:softprob", num_class=n_classes, random_state=42, tree_method="hist", eval_metric="mlogloss"),
        "lgbm": lambda: LGBMClassifier(n_estimators=220, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.0, class_weight="balanced", random_state=42, objective="multiclass", num_class=n_classes),
        "catboost": lambda: CatBoostClassifier(iterations=220, learning_rate=0.05, depth=6, l2_leaf_reg=6, loss_function="MultiClass", random_seed=42, verbose=0, allow_writing_files=False),
        # 🔥 FIX: TabNet added to factories for Hybrid Ensemble Stacking
        "tabnet": lambda: TabNetLiteClassifier(input_dim=input_dim, n_classes=n_classes, hidden_dim=32, n_steps=2, dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=80, patience=3, random_state=42, verbose=0)
    }


def build_oof_meta_features(model_builders, X_train, y_train, X_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_parts = []
    test_parts = []
    fitted_models = {}

    for name, builder in model_builders.items():
        print(f"Generating OOF features for {name}...")
        estimator = builder()
        oof_train = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
        train_parts.append(oof_train)

        fitted = builder()
        fitted.fit(X_train, y_train)
        fitted_models[name] = fitted
        test_parts.append(fitted.predict_proba(X_test))

    X_meta_train = np.hstack(train_parts)
    X_meta_test = np.hstack(test_parts)
    return X_meta_train, X_meta_test, fitted_models


def run_training():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    n_classes = load_label_count()

    rf = train_random_forest(X_train, y_train)
    save_metrics("rf", rf, X_train, y_train, X_test, y_test)

    xgb = train_xgboost(X_train, y_train, X_val, y_val, n_classes)
    save_metrics("xgb", xgb, X_train, y_train, X_test, y_test)

    lgbm = train_lightgbm(X_train, y_train, X_val, y_val, n_classes)
    save_metrics("lgbm", lgbm, X_train, y_train, X_test, y_test)

    cat = train_catboost(X_train, y_train, X_val, y_val)
    save_metrics("catboost", cat, X_train, y_train, X_test, y_test)

    tabnet = train_tabnet(X_train, y_train, X_val, y_val, n_classes)
    save_metrics("tabnet", tabnet, X_train, y_train, X_test, y_test)

    ngb = train_ngboost(X_train, y_train)
    save_metrics("ngboost", ngb, X_train, y_train, X_test, y_test)

    builders = model_factories(n_classes, X_train.shape[1])
    X_meta_train, X_meta_test, fitted_models = build_oof_meta_features(builders, X_train, y_train, X_test)

    print("Training Stacked Ensemble (Logistic Regression)...")
    
    # Paper Alignment: Cost-sensitive sample weighting based on crop profit margins
    CROP_ECONOMICS = {
        "rice": {"cost": 2000, "price": 3500}, "maize": {"cost": 1400, "price": 2500},
        "wheat": {"cost": 1800, "price": 3200}, "mango": {"cost": 2500, "price": 6000},
        "pigeonpeas": {"cost": 3200, "price": 7500}, "mothbeans": {"cost": 3000, "price": 6200},
        "cotton": {"cost": 4200, "price": 8000}, "apple": {"cost": 6000, "price": 14000},
        "banana": {"cost": 1800, "price": 3800}, "coffee": {"cost": 8000, "price": 18000},
        "jute": {"cost": 3500, "price": 5500}, "sugarcane": {"cost": 2200, "price": 4000},
        "coconut": {"cost": 3000, "price": 6500}, "papaya": {"cost": 900, "price": 2500},
        "orange": {"cost": 2800, "price": 5500}, "grapes": {"cost": 4500, "price": 9000},
        "pomegranate": {"cost": 3500, "price": 8000}, "watermelon": {"cost": 1200, "price": 2800},
        "muskmelon": {"cost": 1300, "price": 2800}, "blackgram": {"cost": 2800, "price": 6000},
        "mungbean": {"cost": 2800, "price": 6000}, "lentil": {"cost": 2500, "price": 5500},
        "chickpea": {"cost": 3000, "price": 6000}, "kidneybeans": {"cost": 3200, "price": 7000},
        "default": {"cost": 2500, "price": 5000},
    }
    
    label_encoder = joblib.load(ARTIFACTS_PATH / "label_encoder.pkl")
    
    class_weights = {}
    for i, class_name in enumerate(label_encoder.classes_):
        econ = CROP_ECONOMICS.get(class_name.lower().strip().replace(" ", ""), CROP_ECONOMICS["default"])
        profit = max(100, econ["price"] - econ["cost"]) 
        class_weights[i] = profit / 1000.0 # Scale to prevent extreme gradients
        
    sample_weights_train = np.array([class_weights[y] for y in y_train])
    
    meta_model = LogisticRegression(max_iter=4000, C=0.8)
    meta_model.fit(X_meta_train, y_train, sample_weight=sample_weights_train)
    y_pred = meta_model.predict(X_meta_test)

    ensemble_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "base_models": list(fitted_models.keys()),
    }
    with open(ARTIFACTS_PATH / "ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_metrics, f, indent=4)

    joblib.dump(meta_model, ARTIFACTS_PATH / "stacked_model.pkl")
    joblib.dump(fitted_models["rf"], ARTIFACTS_PATH / "model_rf.pkl")
    joblib.dump(fitted_models["xgb"], ARTIFACTS_PATH / "model_xgb.pkl")
    joblib.dump(fitted_models["lgbm"], ARTIFACTS_PATH / "model_lgbm.pkl")
    joblib.dump(fitted_models["catboost"], ARTIFACTS_PATH / "model_catboost.pkl")
    joblib.dump(fitted_models["tabnet"], ARTIFACTS_PATH / "model_tabnet.pkl")

    with open(ARTIFACTS_PATH / "base_model_order.json", "w", encoding="utf-8") as f:
        json.dump(["rf", "xgb", "lgbm", "catboost", "tabnet"], f, indent=2)


if __name__ == "__main__":
    run_training()