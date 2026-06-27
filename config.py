"""ChurnIQ – Central Configuration"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "customer_churn_business_dataset.csv")
DB_PATH    = os.path.join(BASE_DIR, "data", "churniq.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Target ─────────────────────────────────────────────────────────────────
TARGET_COL = "Churn"
ID_COL     = "customer_id"

# ── Feature Groups ─────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender", "country", "city", "customer_segment",
    "signup_channel", "contract_type", "payment_method",
    "discount_applied", "price_increase_last_3m",
    "complaint_type", "survey_response",
]

NUMERICAL_FEATURES = [
    "age", "tenure_months", "monthly_logins", "weekly_active_days",
    "avg_session_time", "features_used", "usage_growth_rate",
    "last_login_days_ago", "monthly_fee", "total_revenue",
    "payment_failures", "support_tickets", "avg_resolution_time",
    "csat_score", "escalations", "email_open_rate",
    "marketing_click_rate", "nps_score", "referral_count",
]

# ── Models ─────────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "Decision Tree",
    "XGBoost",
]

# ── Risk Tiers ──────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {"Low": 0.30, "Medium": 0.60, "High": 1.01}

# ── App Settings ───────────────────────────────────────────────────────────
APP_TITLE   = "ChurnIQ"
APP_ICON    = "📊"
LAYOUT      = "wide"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
