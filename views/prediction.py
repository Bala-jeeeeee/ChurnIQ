"""ChurnIQ – Prediction Page."""

import streamlit as st
import pandas as pd
import io
from core.predictor import predictor
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ── Default value helpers ─────────────────────────────────────────────────
_CAT_OPTIONS = {
    "gender":               ["Male", "Female"],
    "country":              ["USA", "UK", "Canada", "Australia", "Germany",
                             "France", "India", "Bangladesh", "Brazil", "Japan"],
    "city":                 ["London", "New York", "Sydney", "Dhaka", "Paris",
                             "Berlin", "Toronto", "Tokyo", "Mumbai", "Sao Paulo"],
    "customer_segment":     ["Individual", "SME", "Enterprise"],
    "signup_channel":       ["Web", "Mobile", "Referral", "Agent"],
    "contract_type":        ["Monthly", "Yearly", "Two-Year"],
    "payment_method":       ["Card", "PayPal", "Bank Transfer", "Cash"],
    "discount_applied":     ["Yes", "No"],
    "price_increase_last_3m": ["Yes", "No"],
    "complaint_type":       ["Billing", "Service", "Technical", "Other"],
    "survey_response":      ["Satisfied", "Neutral", "Dissatisfied"],
}


def _single_prediction_ui():
    st.subheader("Single Customer Prediction")
    st.caption("Fill in the customer details below and click Predict.")

    with st.form("single_pred_form"):
        st.markdown("#### Demographics & Account")
        r1c1, r1c2, r1c3 = st.columns(3)
        customer_id = r1c1.text_input("Customer ID", value="CUST_NEW")
        gender      = r1c2.selectbox("Gender", _CAT_OPTIONS["gender"])
        age         = r1c3.number_input("Age", 18, 100, 35)

        r2c1, r2c2, r2c3 = st.columns(3)
        country          = r2c1.selectbox("Country", _CAT_OPTIONS["country"])
        city             = r2c2.selectbox("City", _CAT_OPTIONS["city"])
        customer_segment = r2c3.selectbox("Customer Segment", _CAT_OPTIONS["customer_segment"])

        st.markdown("#### Contract & Billing")
        r3c1, r3c2, r3c3 = st.columns(3)
        contract_type     = r3c1.selectbox("Contract Type", _CAT_OPTIONS["contract_type"])
        payment_method    = r3c2.selectbox("Payment Method", _CAT_OPTIONS["payment_method"])
        signup_channel    = r3c3.selectbox("Signup Channel", _CAT_OPTIONS["signup_channel"])

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        monthly_fee       = r4c1.number_input("Monthly Fee ($)", 0, 500, 30)
        total_revenue     = r4c2.number_input("Total Revenue ($)", 0, 100_000, 1000)
        discount_applied  = r4c3.selectbox("Discount Applied", _CAT_OPTIONS["discount_applied"])
        price_increase    = r4c4.selectbox("Price Increase Last 3M", _CAT_OPTIONS["price_increase_last_3m"])

        st.markdown("#### Engagement")
        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        tenure_months     = r5c1.number_input("Tenure (months)", 0, 120, 12)
        monthly_logins    = r5c2.number_input("Monthly Logins", 0, 200, 10)
        weekly_active_days= r5c3.number_input("Weekly Active Days", 0, 7, 3)
        avg_session_time  = r5c4.number_input("Avg Session Time (min)", 0.0, 120.0, 15.0)

        r6c1, r6c2, r6c3 = st.columns(3)
        features_used     = r6c1.number_input("Features Used", 0, 20, 3)
        usage_growth_rate = r6c2.number_input("Usage Growth Rate", -1.0, 5.0, 0.05, step=0.01)
        last_login_days   = r6c3.number_input("Last Login (days ago)", 0, 365, 5)

        st.markdown("#### Support & Satisfaction")
        r7c1, r7c2, r7c3, r7c4 = st.columns(4)
        payment_failures  = r7c1.number_input("Payment Failures", 0, 10, 0)
        support_tickets   = r7c2.number_input("Support Tickets", 0, 20, 1)
        avg_resolution    = r7c3.number_input("Avg Resolution Time (hrs)", 0.0, 72.0, 12.0)
        complaint_type    = r7c4.selectbox("Complaint Type", _CAT_OPTIONS["complaint_type"])

        r8c1, r8c2, r8c3, r8c4 = st.columns(4)
        csat_score        = r8c1.number_input("CSAT Score (1-5)", 1, 5, 4)
        escalations       = r8c2.number_input("Escalations", 0, 10, 0)
        nps_score         = r8c3.number_input("NPS Score (-100–100)", -100, 100, 30)
        survey_response   = r8c4.selectbox("Survey Response", _CAT_OPTIONS["survey_response"])

        st.markdown("#### Marketing")
        r9c1, r9c2, r9c3 = st.columns(3)
        email_open_rate   = r9c1.number_input("Email Open Rate (0–1)", 0.0, 1.0, 0.4, step=0.01)
        mktg_click_rate   = r9c2.number_input("Marketing Click Rate (0–1)", 0.0, 1.0, 0.2, step=0.01)
        referral_count    = r9c3.number_input("Referral Count", 0, 50, 0)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

    if submitted:
        input_data = {
            "gender": gender, "age": age, "country": country, "city": city,
            "customer_segment": customer_segment, "tenure_months": tenure_months,
            "signup_channel": signup_channel, "contract_type": contract_type,
            "monthly_logins": monthly_logins, "weekly_active_days": weekly_active_days,
            "avg_session_time": avg_session_time, "features_used": features_used,
            "usage_growth_rate": usage_growth_rate, "last_login_days_ago": last_login_days,
            "monthly_fee": monthly_fee, "total_revenue": total_revenue,
            "payment_method": payment_method, "payment_failures": payment_failures,
            "discount_applied": discount_applied, "price_increase_last_3m": price_increase,
            "support_tickets": support_tickets, "avg_resolution_time": avg_resolution,
            "complaint_type": complaint_type, "csat_score": csat_score,
            "escalations": escalations, "email_open_rate": email_open_rate,
            "marketing_click_rate": mktg_click_rate, "nps_score": nps_score,
            "survey_response": survey_response, "referral_count": referral_count,
        }

        try:
            result = predictor.predict_single(input_data)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        # ── Result display ────────────────────────────────────────────────
        st.divider()
        st.subheader(f"Prediction Result for `{customer_id}`")

        if result.prediction == 1:
            st.error("🔴 Prediction: **Likely to Churn**")
        else:
            st.success("🟢 Prediction: **Likely to Stay**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{result.probability * 100:.1f}%")
        m2.metric("Risk Tier",         result.risk_tier)
        m3.metric("Confidence",        f"{max(result.probability, 1 - result.probability)*100:.1f}%")

        st.progress(float(result.probability), text=f"Churn Probability: {result.probability:.1%}")

        st.subheader("Retention Recommendations")
        for rec in result.recommendations:
            st.write(rec)

        # Save to history
        user = st.session_state.get("user")
        if user:
            from core.database import save_prediction
            save_prediction(user["id"], customer_id, result.prediction,
                            result.probability, result.risk_tier)


def _batch_prediction_ui():
    st.subheader("Batch Prediction (CSV Upload)")
    st.info(
        "Upload a CSV file with the same columns as the training dataset. "
        "Predictions, probabilities, and risk tiers will be added and available for download."
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    st.write(f"Loaded **{len(batch_df):,}** rows × **{len(batch_df.columns)}** columns.")
    st.dataframe(batch_df.head(10), use_container_width=True)

    if st.button("🔮 Run Batch Prediction", type="primary"):
        try:
            out_df = predictor.predict_batch(batch_df)
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        st.success(f"✅ Predictions complete for {len(out_df):,} customers.")

        # Summary stats
        total_pred = len(out_df)
        churn_pred = int(out_df["Predicted_Churn"].sum())
        high_risk  = int((out_df["Risk_Tier"] == "High").sum())
        med_risk   = int((out_df["Risk_Tier"] == "Medium").sum())
        low_risk   = int((out_df["Risk_Tier"] == "Low").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers",  f"{total_pred:,}")
        c2.metric("Predicted Churn",  f"{churn_pred:,}")
        c3.metric("High Risk",        f"{high_risk:,}")
        c4.metric("Medium Risk",      f"{med_risk:,}")
        c5.metric("Low Risk",         f"{low_risk:,}")

        st.subheader("Prediction Results")
        st.dataframe(out_df[["Predicted_Churn", "Churn_Probability",
                              "Risk_Tier", "Prediction_Label"]
                             + (["customer_id"] if "customer_id" in out_df.columns else [])
                             ].head(200),
                     use_container_width=True)

        # Download button
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_bytes,
            file_name="churniq_predictions.csv",
            mime="text/csv",
        )

        # Risk-tier recommendations
        st.subheader("Retention Recommendations by Risk Tier")
        with st.expander("🔴 High Risk Recommendations"):
            st.write("- Immediate customer success call")
            st.write("- Offer targeted discount or promotional pricing")
            st.write("- Present contract upgrade incentive")
            st.write("- Assign dedicated account manager")
            st.write("- Escalate to retention team within 48 hours")
        with st.expander("🟡 Medium Risk Recommendations"):
            st.write("- Enrol in loyalty rewards program")
            st.write("- Send personalised re-engagement email campaign")
            st.write("- Offer feature trial or add-on service")
            st.write("- Schedule quarterly business review")
        with st.expander("🟢 Low Risk Recommendations"):
            st.write("- Explore upsell / cross-sell opportunities")
            st.write("- Invite to referral or ambassador program")
            st.write("- Gather NPS feedback for product roadmap")


def render():
    st.title("🔮 Churn Prediction")

    if not predictor.is_model_ready():
        st.warning(
            "⚠️ No trained model found. Please go to the **Training** page and "
            "train the models first."
        )
        return

    tab_single, tab_batch, tab_history = st.tabs(
        ["👤 Single Prediction", "📂 Batch Prediction", "📜 History"]
    )

    with tab_single:
        _single_prediction_ui()

    with tab_batch:
        _batch_prediction_ui()

    with tab_history:
        st.subheader("Prediction History")
        user = st.session_state.get("user")
        if not user:
            st.info("Login to view prediction history.")
            return
        from core.database import get_prediction_history
        history = get_prediction_history(user["id"])
        if not history:
            st.info("No predictions recorded yet.")
        else:
            hist_df = pd.DataFrame(history)
            hist_df["prediction"] = hist_df["prediction"].map({0: "Stay", 1: "Churn"})
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
