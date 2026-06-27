"""ChurnIQ – Home Page."""

import streamlit as st
import pandas as pd
from core.preprocess import load_data, dataset_summary
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL


def render(df: pd.DataFrame):
    st.title("📊 ChurnIQ – Customer Churn Analytics Dashboard")
    st.markdown(
        "Welcome to **ChurnIQ**, a production-ready dashboard for exploring, "
        "analysing, and predicting customer churn using machine learning."
    )

    # ── Business Context ──────────────────────────────────────────────────
    st.header("Business Objective")
    st.info(
        "**Customer churn** occurs when a customer stops doing business with a company. "
        "Predicting churn early allows retention teams to take proactive action, "
        "reducing revenue loss and improving customer lifetime value (CLV). "
        "A 5% reduction in churn can increase profits by 25–95% (Harvard Business Review)."
    )

    # ── Dataset Summary ───────────────────────────────────────────────────
    st.header("Dataset Overview")
    summary = dataset_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",   f"{summary['rows']:,}")
    c2.metric("Total Features",    f"{summary['cols'] - 2}")
    c3.metric("Numerical Features", summary["numerical"])
    c4.metric("Categorical Features", summary["categorical"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Missing Values",   summary["missing"])
    c6.metric("Duplicate Rows",   summary["duplicates"])
    c7.metric("Churned Customers", f"{summary['churn_count']:,}")
    c8.metric("Churn Rate",        f"{summary['churn_rate']}%")

    # ── Target Variable ───────────────────────────────────────────────────
    st.header("Target Variable")
    col_a, col_b = st.columns(2)
    col_a.success(f"✅ **Retained** – {summary['retain_count']:,} customers "
                  f"({100 - summary['churn_rate']:.2f}%)")
    col_b.error(f"🔴 **Churned** – {summary['churn_count']:,} customers "
                f"({summary['churn_rate']}%)")

    st.info(
        "**Note:** The dataset is imbalanced (~8.8:1 retain-to-churn ratio). "
        "Models are trained with `class_weight='balanced'` to address this, "
        "and ROC-AUC is used as the primary evaluation metric."
    )

    # ── Schema Table ──────────────────────────────────────────────────────
    st.header("Dataset Schema")
    schema_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        unique  = int(df[col].nunique())
        feature_type = (
            "ID" if col == "customer_id" else
            "Target" if col == TARGET_COL else
            "Numerical" if col in NUMERICAL_FEATURES else
            "Categorical"
        )
        schema_rows.append({
            "Column": col,
            "Type": dtype,
            "Feature Type": feature_type,
            "Missing": missing,
            "Unique Values": unique,
        })
    schema_df = pd.DataFrame(schema_rows)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    # ── Feature Groups ────────────────────────────────────────────────────
    with st.expander("📋 Numerical Features"):
        st.write(", ".join(NUMERICAL_FEATURES))

    with st.expander("🏷️ Categorical Features"):
        st.write(", ".join(CATEGORICAL_FEATURES))

    # ── Navigation Hint ──────────────────────────────────────────────────
    st.header("How to Use ChurnIQ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1️⃣ EDA")
        st.write("Explore the dataset with interactive visualisations, KPIs, and business insights.")
    with col2:
        st.subheader("2️⃣ Training")
        st.write("Train and compare 5 ML models. The best model is auto-selected by ROC-AUC.")
    with col3:
        st.subheader("3️⃣ Prediction")
        st.write("Predict churn for a single customer or upload a CSV for batch prediction.")
