"""ChurnIQ – Prediction engine."""

import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass

from config import MODELS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RISK_THRESHOLDS


@dataclass
class PredictionResult:
    prediction: int          # 0 or 1
    probability: float       # churn probability
    risk_tier: str           # Low / Medium / High
    recommendations: list[str]


def _get_risk_tier(prob: float) -> str:
    if prob < RISK_THRESHOLDS["Low"]:
        return "Low"
    elif prob < RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "High"


def _get_recommendations(risk_tier: str, row: dict) -> list[str]:
    base = {
        "High": [
            "🔴 Immediate customer success call recommended",
            "💰 Offer targeted discount or promotional pricing",
            "📋 Present contract upgrade incentive",
            "🎯 Assign dedicated account manager",
            "📞 Escalate to retention team within 48 hours",
        ],
        "Medium": [
            "🟡 Enroll in loyalty rewards program",
            "📧 Send personalized re-engagement email campaign",
            "🎁 Offer feature trial or add-on service",
            "📊 Schedule a quarterly business review",
        ],
        "Low": [
            "🟢 Explore upsell or cross-sell opportunities",
            "⭐ Invite to referral or ambassador program",
            "📣 Gather NPS feedback for product roadmap",
        ],
    }
    recs = list(base.get(risk_tier, []))
    # Context-aware additions
    if row.get("support_tickets", 0) and int(row.get("support_tickets", 0)) > 3:
        recs.append("🔧 Review open support tickets and expedite resolution")
    if row.get("payment_failures", 0) and int(row.get("payment_failures", 0)) > 0:
        recs.append("💳 Reach out about billing/payment assistance")
    if row.get("csat_score", 5) and float(row.get("csat_score", 5)) < 3:
        recs.append("📝 Send CSAT survey follow-up and escalate feedback")
    return recs


class Predictor:
    def __init__(self):
        self._model = None
        self._preprocessor = None

    def _load(self):
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        prep_path  = os.path.join(MODELS_DIR, "preprocessor.pkl")
        if not os.path.exists(model_path) or not os.path.exists(prep_path):
            raise FileNotFoundError(
                "Trained model not found. Please train a model first."
            )
        self._model        = joblib.load(model_path)
        self._preprocessor = joblib.load(prep_path)

    def _ensure_loaded(self):
        if self._model is None:
            self._load()

    def predict_single(self, input_dict: dict) -> PredictionResult:
        self._ensure_loaded()
        feat_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        row = {c: input_dict.get(c, np.nan) for c in feat_cols}
        df  = pd.DataFrame([row])
        X   = self._preprocessor.transform(df)
        pred  = int(self._model.predict(X)[0])
        prob  = float(self._model.predict_proba(X)[0][1])
        tier  = _get_risk_tier(prob)
        recs  = _get_recommendations(tier, input_dict)
        return PredictionResult(pred, round(prob, 4), tier, recs)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded()
        feat_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        available = [c for c in feat_cols if c in df.columns]
        X_raw = df[available].reindex(columns=feat_cols)
        X = self._preprocessor.transform(X_raw)
        preds = self._model.predict(X)
        probas = self._model.predict_proba(X)[:, 1]
        out = df.copy()
        out["Predicted_Churn"] = preds
        out["Churn_Probability"] = probas.round(4)
        out["Risk_Tier"] = [_get_risk_tier(p) for p in probas]
        out["Prediction_Label"] = ["Likely to Churn" if p == 1 else "Likely to Stay"
                                   for p in preds]
        return out

    def is_model_ready(self) -> bool:
        return (
            os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")) and
            os.path.exists(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        )


# Singleton
predictor = Predictor()
