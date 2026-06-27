"""ChurnIQ – Data loading and preprocessing pipeline."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib, os

from config import (
    DATA_PATH, MODELS_DIR, TARGET_COL, ID_COL,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    RANDOM_SEED, TEST_SIZE,
)


# ── Load ────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def get_filtered_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to a dataframe."""
    out = df.copy()
    for col, values in filters.items():
        if values and col in out.columns:
            out = out[out[col].isin(values)]
    return out


# ── Summary stats ───────────────────────────────────────────────────────────
def dataset_summary(df: pd.DataFrame) -> dict:
    churn_counts = df[TARGET_COL].value_counts()
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "numerical": len(NUMERICAL_FEATURES),
        "categorical": len(CATEGORICAL_FEATURES),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "churn_count": int(churn_counts.get(1, 0)),
        "retain_count": int(churn_counts.get(0, 0)),
        "churn_rate": round(float(churn_counts.get(1, 0)) / len(df) * 100, 2),
    }


# ── Preprocessing pipeline ──────────────────────────────────────────────────
def build_pipeline() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    transformer = ColumnTransformer([
        ("num", num_pipe, NUMERICAL_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")
    return transformer


def prepare_data(df: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test, preprocessor (fitted)."""
    feat_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    available = [c for c in feat_cols if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_pipeline()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Save preprocessor
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor
