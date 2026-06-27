"""ChurnIQ – Application Entry Point."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from config import APP_TITLE, APP_ICON, LAYOUT
from core.database import init_db, register_user, login_user
from core.preprocess import load_data, get_filtered_data

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Init DB ───────────────────────────────────────────────────────────────
init_db()


# ── Auth helpers ──────────────────────────────────────────────────────────
def _login_page():
    st.title("📊 ChurnIQ")
    st.subheader("Customer Churn Analytics & Prediction Dashboard")
    st.divider()

    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        st.subheader("Sign In")
        identifier = st.text_input("Username or Email", key="login_id")
        password   = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", type="primary", key="btn_login"):
            if not identifier or not password:
                st.error("Please fill in all fields.")
            else:
                ok, user_info, msg = login_user(identifier, password)
                if ok:
                    st.session_state.user = user_info
                    st.session_state.authenticated = True
                    st.success(f"Welcome back, **{user_info['username']}**! 🎉")
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        st.subheader("Create Account")
        new_username = st.text_input("Username", key="reg_user")
        new_email    = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_pw")
        confirm_pw   = st.text_input("Confirm Password", type="password", key="reg_cpw")

        if st.button("Register", type="primary", key="btn_register"):
            if not all([new_username, new_email, new_password, confirm_pw]):
                st.error("Please fill in all fields.")
            elif new_password != confirm_pw:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.warning("Password must be at least 6 characters.")
            else:
                ok, msg = register_user(new_username, new_email, new_password)
                if ok:
                    st.success(msg + " Please log in.")
                else:
                    st.error(msg)


def _sidebar(df):
    """Render sidebar filters; return filtered dataframes per page."""
    with st.sidebar:
        st.title(f"📊 {APP_TITLE}")
        user = st.session_state.get("user", {})
        st.caption(f"👤 {user.get('username', '')} | {user.get('email', '')}")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["user", "authenticated", "training_results", "best_model_name"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()

        # ── Navigation ──────────────────────────────────────────────────
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 EDA", "🤖 Training", "🔮 Prediction"],
            key="nav_page",
        )

        st.divider()

        # ── Filters ─────────────────────────────────────────────────────
        st.subheader("🎛️ Filters")
        apply_to = st.multiselect(
            "Apply Filters To",
            ["Home", "EDA", "Training"],
            default=["EDA"],
            help="Select which pages the filters below should affect.",
        )

        countries  = sorted(df["country"].dropna().unique().tolist())
        genders    = sorted(df["gender"].dropna().unique().tolist())
        segments   = sorted(df["customer_segment"].dropna().unique().tolist())
        contracts  = sorted(df["contract_type"].dropna().unique().tolist())
        churn_opts = ["All", "Churned", "Retained"]

        sel_countries = st.multiselect("Country",          countries)
        sel_genders   = st.multiselect("Gender",           genders)
        sel_segments  = st.multiselect("Customer Segment", segments)
        sel_contracts = st.multiselect("Contract Type",    contracts)
        sel_churn     = st.selectbox("Churn Status",       churn_opts)

        raw_filters = {
            "country":          sel_countries,
            "gender":           sel_genders,
            "customer_segment": sel_segments,
            "contract_type":    sel_contracts,
        }

        def _apply_filters(base_df):
            out = get_filtered_data(base_df, raw_filters)
            if sel_churn == "Churned":
                out = out[out["Churn"] == 1]
            elif sel_churn == "Retained":
                out = out[out["Churn"] == 0]
            return out

        filtered = {}
        for p in ["Home", "EDA", "Training"]:
            filtered[p] = _apply_filters(df) if p in apply_to else df.copy()

    return page, filtered


def main():
    if not st.session_state.get("authenticated"):
        _login_page()
        return

    df = load_data()
    page, filtered = _sidebar(df)

    # ── Route to page ──────────────────────────────────────────────────
    if page == "🏠 Home":
        from views.home import render
        render(filtered["Home"])

    elif page == "🔍 EDA":
        from views.eda import render
        render(filtered["EDA"])

    elif page == "🤖 Training":
        from views.training import render
        render(filtered["Training"])

    elif page == "🔮 Prediction":
        from views.prediction import render
        render()


if __name__ == "__main__":
    main()
