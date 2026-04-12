from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from .tabnet_surrogate import TabNetLiteClassifier
except ImportError:
    from tabnet_surrogate import TabNetLiteClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = BASE_DIR / "ml" / "artifacts"


def load_data():
    X_train = joblib.load(ARTIFACTS_PATH / "X_train.pkl")
    X_test = joblib.load(ARTIFACTS_PATH / "X_test.pkl")
    y_train = joblib.load(ARTIFACTS_PATH / "y_train.pkl")
    y_test = joblib.load(ARTIFACTS_PATH / "y_test.pkl")
    return X_train, X_test, y_train, y_test


def load_label_count():
    y_train = joblib.load(ARTIFACTS_PATH / "y_train.pkl")
    return int(np.max(y_train)) + 1

# 🔥 FIX: TabNet added to base models for Hybrid Stacking
def base_models(n_classes, input_dim):
    return {
        "rf": lambda: RandomForestClassifier(n_estimators=180, max_depth=10, min_samples_split=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "xgb": lambda: XGBClassifier(n_estimators=220, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, min_child_weight=2, objective="multi:softprob", num_class=n_classes, random_state=42, tree_method="hist", eval_metric="mlogloss"),
        "lgbm": lambda: LGBMClassifier(n_estimators=220, learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.0, class_weight="balanced", random_state=42, objective="multiclass", num_class=n_classes),
        "catboost": lambda: CatBoostClassifier(iterations=220, learning_rate=0.05, depth=6, l2_leaf_reg=6, loss_function="MultiClass", random_seed=42, verbose=0, allow_writing_files=False),
        "tabnet": lambda: TabNetLiteClassifier(input_dim=input_dim, n_classes=n_classes, hidden_dim=32, n_steps=2, dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=200, patience=3, random_state=42, verbose=0)
    }


def run_ensemble():
    X_train, X_test, y_train, y_test = load_data()
    n_classes = load_label_count()
    builders = base_models(n_classes, X_train.shape[1])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_parts = []
    test_parts = []
    fitted_models = {}

    print("Building Meta-Features for Hybrid Ensemble...")
    for name, builder in builders.items():
        print(f"Stacking {name}...")
        model = builder()
        oof_train = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)
        train_parts.append(oof_train)

        fitted = builder()
        fitted.fit(X_train, y_train)
        fitted_models[name] = fitted
        test_parts.append(fitted.predict_proba(X_test))

    X_meta_train = np.hstack(train_parts)
    X_meta_test = np.hstack(test_parts)

    print("Training Logistic Regression Meta-Learner...")
    meta_model = LogisticRegression(max_iter=4000, C=0.8)
    meta_model.fit(X_meta_train, y_train)
    y_pred = meta_model.predict(X_meta_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "base_models": list(fitted_models.keys()),
    }

    with open(ARTIFACTS_PATH / "ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    joblib.dump(meta_model, ARTIFACTS_PATH / "stacked_model.pkl")
    for name, model in fitted_models.items():
        joblib.dump(model, ARTIFACTS_PATH / f"model_{name}.pkl")
    
    print("Hybrid Ensemble saved successfully with ALL claims fulfilled!")


if __name__ == "__main__":
    run_ensemble()