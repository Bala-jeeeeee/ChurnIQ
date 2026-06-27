"""ChurnIQ – EDA & Business Analytics Page."""

import streamlit as st
import pandas as pd
from ui import charts
from config import NUMERICAL_FEATURES


def render(df: pd.DataFrame):
    st.title("🔍 EDA & Business Analytics")
    st.caption(f"Analysing {len(df):,} customers after applied filters.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Dataset Overview", "📈 KPIs", "📊 Visualisations", "💡 Business Insights"]
    )

    # ── Tab 1: Overview ───────────────────────────────────────────────────
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Shape")
        st.write(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")

        st.subheader("Missing Values")
        miss = df.isnull().sum()
        miss = miss[miss > 0].reset_index()
        miss.columns = ["Column", "Missing Count"]
        if miss.empty:
            st.success("No missing values in the filtered dataset.")
        else:
            st.dataframe(miss, use_container_width=True, hide_index=True)

        st.subheader("Data Types")
        dtype_df = df.dtypes.reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

        dup = df.duplicated().sum()
        if dup > 0:
            st.warning(f"⚠️ {dup} duplicate rows found.")
        else:
            st.success("✅ No duplicate rows.")

    # ── Tab 2: KPIs ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("Key Performance Indicators")
        total = len(df)
        churned = df["Churn"].sum()
        retained = total - churned
        churn_rate = churned / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers",    f"{total:,}")
        c2.metric("Churned",            f"{churned:,}")
        c3.metric("Retained",           f"{retained:,}")
        c4.metric("Churn Rate",         f"{churn_rate:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg Monthly Fee",    f"${df['monthly_fee'].mean():.2f}")
        c6.metric("Avg Total Revenue",  f"${df['total_revenue'].mean():.2f}")
        c7.metric("Avg CSAT Score",     f"{df['csat_score'].mean():.2f}")
        c8.metric("Avg NPS Score",      f"{df['nps_score'].mean():.1f}")

        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Avg Tenure (months)", f"{df['tenure_months'].mean():.1f}")
        c10.metric("Avg Support Tickets", f"{df['support_tickets'].mean():.2f}")
        c11.metric("Avg Login Frequency", f"{df['monthly_logins'].mean():.1f}")
        c12.metric("Avg Session Time",    f"{df['avg_session_time'].mean():.1f} min")

        st.divider()
        st.subheader("Revenue by Churn Status")
        rev_grp = df.groupby("Churn")["total_revenue"].agg(["mean", "sum"]).reset_index()
        rev_grp.columns = ["Churn", "Avg Revenue", "Total Revenue"]
        rev_grp["Churn"] = rev_grp["Churn"].map({0: "Retained", 1: "Churned"})
        st.dataframe(rev_grp.style.format({"Avg Revenue": "${:.2f}", "Total Revenue": "${:,.0f}"}),
                     use_container_width=True, hide_index=True)

    # ── Tab 3: Visualisations ─────────────────────────────────────────────
    with tab3:
        st.subheader("Churn Overview")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(charts.churn_distribution(df), use_container_width=True)
        with col_b:
            st.plotly_chart(charts.churn_by_category(df, "contract_type",
                            "Churn by Contract Type"), use_container_width=True)

        st.divider()
        st.subheader("Demographics")
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(charts.churn_rate_by_category(df, "gender",
                            "Churn Rate by Gender"), use_container_width=True)
        with col_d:
            st.plotly_chart(charts.churn_rate_by_category(df, "country",
                            "Churn Rate by Country"), use_container_width=True)

        st.plotly_chart(charts.churn_rate_by_category(df, "customer_segment",
                        "Churn Rate by Customer Segment"), use_container_width=True)

        st.divider()
        st.subheader("Revenue & Financials")
        col_e, col_f = st.columns(2)
        with col_e:
            st.plotly_chart(charts.revenue_boxplot(df), use_container_width=True)
        with col_f:
            st.plotly_chart(charts.numeric_histogram(df, "monthly_fee",
                            "Monthly Fee Distribution"), use_container_width=True)

        st.divider()
        st.subheader("Engagement Metrics")
        col_g, col_h = st.columns(2)
        with col_g:
            st.plotly_chart(charts.tenure_histogram(df), use_container_width=True)
        with col_h:
            st.plotly_chart(charts.numeric_histogram(df, "monthly_logins",
                            "Monthly Logins Distribution"), use_container_width=True)

        st.divider()
        st.subheader("Satisfaction Metrics")
        col_i, col_j = st.columns(2)
        with col_i:
            st.plotly_chart(charts.nps_violin(df), use_container_width=True)
        with col_j:
            st.plotly_chart(charts.csat_bar(df), use_container_width=True)

        st.divider()
        st.subheader("Segment Analysis")
        st.plotly_chart(charts.segment_sunburst(df), use_container_width=True)

        st.divider()
        st.subheader("Correlation Heatmap")
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        st.plotly_chart(charts.correlation_heatmap(df, num_cols), use_container_width=True)

    # ── Tab 4: Business Insights ──────────────────────────────────────────
    with tab4:
        st.subheader("Automated Business Insights")

        # Highest churn segment
        seg_churn = df.groupby("customer_segment")["Churn"].mean()
        worst_seg = seg_churn.idxmax()
        best_seg  = seg_churn.idxmin()
        st.error(
            f"🔴 **Highest Risk Segment:** `{worst_seg}` with "
            f"{seg_churn[worst_seg]*100:.1f}% churn rate"
        )
        st.success(
            f"🟢 **Best Performing Segment:** `{best_seg}` with "
            f"{seg_churn[best_seg]*100:.1f}% churn rate"
        )

        # Contract churn
        contract_churn = df.groupby("contract_type")["Churn"].mean()
        worst_contract = contract_churn.idxmax()
        st.warning(
            f"⚠️ **Highest Churn Contract Type:** `{worst_contract}` "
            f"({contract_churn[worst_contract]*100:.1f}%)"
        )

        # Most profitable segment
        seg_rev = df.groupby("customer_segment")["total_revenue"].mean()
        best_rev_seg = seg_rev.idxmax()
        st.info(
            f"💰 **Most Profitable Segment:** `{best_rev_seg}` – "
            f"avg revenue ${seg_rev[best_rev_seg]:,.0f}"
        )

        # Risk Indicators
        st.divider()
        st.subheader("Risk Indicators")
        high_tickets = df[df["support_tickets"] > 3]["Churn"].mean()
        high_fail    = df[df["payment_failures"] > 0]["Churn"].mean()
        low_csat     = df[df["csat_score"] <= 2]["Churn"].mean()
        no_login     = df[df["last_login_days_ago"] > 30]["Churn"].mean()

        ri_cols = st.columns(4)
        ri_cols[0].metric("Churn – >3 Support Tickets", f"{high_tickets*100:.1f}%")
        ri_cols[1].metric("Churn – Payment Failures",   f"{high_fail*100:.1f}%")
        ri_cols[2].metric("Churn – Low CSAT (≤2)",      f"{low_csat*100:.1f}%")
        ri_cols[3].metric("Churn – Inactive >30 days",  f"{no_login*100:.1f}%")

        st.divider()
        st.subheader("Country-wise Churn Rate")
        country_churn = (
            df.groupby("country")["Churn"].mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        country_churn.columns = ["Country", "Churn Rate"]
        country_churn["Churn Rate %"] = (country_churn["Churn Rate"] * 100).round(2)
        st.dataframe(country_churn[["Country", "Churn Rate %"]],
                     use_container_width=True, hide_index=True)
