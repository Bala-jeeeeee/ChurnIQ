"""ChurnIQ – Model training loop."""

import os, time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)

from config import MODELS_DIR, RANDOM_SEED

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _build_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, random_state=RANDOM_SEED
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1, 
            eval_metric="logloss", random_state=RANDOM_SEED,
            scale_pos_weight=9, n_jobs=-1,
        )
    return models


def _evaluate(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)),                  4),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": prec.tolist(),
        "recall_curve":    rec.tolist(),
        "y_proba": y_proba.tolist(),
        "y_test":  y_test.tolist(),
    }


def train_all(X_train, X_test, y_train, y_test,
              progress_callback=None) -> tuple[dict, str]:
    """Train all models; return results dict and best model name."""
    models  = _build_models()
    results = {}
    os.makedirs(MODELS_DIR, exist_ok=True)

    for i, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(i, name)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)
        metrics = _evaluate(model, X_test, y_test)
        metrics["train_time"] = elapsed
        results[name] = metrics
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

    # Best by ROC-AUC
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    joblib.dump(
        joblib.load(os.path.join(MODELS_DIR, f"{best_name.replace(' ', '_')}.pkl")),
        os.path.join(MODELS_DIR, "best_model.pkl"),
    )
    # Save best name
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    if progress_callback:
        progress_callback(len(models), "Done")

    return results, best_name


def load_results() -> tuple[dict | None, str | None]:
    name_path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if not os.path.exists(name_path):
        return None, None
    with open(name_path) as f:
        best_name = f.read().strip()
    return None, best_name   # results not persisted; retrain to get them
