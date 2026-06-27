"""ChurnIQ – Model Training Page."""

import streamlit as st
import pandas as pd
from core.preprocess import prepare_data
from core.trainer import train_all, load_results
from ui import charts


def render(df: pd.DataFrame):
    st.title("🤖 Model Training & Evaluation")

    # ── Training trigger ──────────────────────────────────────────────────
    st.info(
        "Train five classification models on the Customer Churn dataset. "
        "The best model (by ROC-AUC) is saved automatically for prediction."
    )

    if "training_results" not in st.session_state:
        st.session_state.training_results = None
        st.session_state.best_model_name  = None

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        train_btn = st.button("🚀 Train All Models", type="primary", use_container_width=True)

    if train_btn:
        with st.status("Training models… this may take a minute.", expanded=True) as status:
            try:
                progress_bar = st.progress(0)
                log_area     = st.empty()
                log_lines    = []
                model_names  = ["Logistic Regression", "Random Forest",
                                "Gradient Boosting", "Decision Tree", "XGBoost"]

                def progress_cb(i, name):
                    pct = int(i / len(model_names) * 100)
                    progress_bar.progress(min(pct, 100))
                    if name != "Done":
                        log_lines.append(f"⏳ Training: **{name}**")
                    else:
                        log_lines.append("✅ All models trained!")
                    log_area.markdown("\n\n".join(log_lines))

                st.write("🔧 Preprocessing data…")
                X_tr, X_te, y_tr, y_te, _ = prepare_data(df)
                st.write("✅ Preprocessing complete.")

                results, best_name = train_all(X_tr, X_te, y_tr, y_te, progress_cb)
                st.session_state.training_results = results
                st.session_state.best_model_name  = best_name
                progress_bar.progress(100)
                status.update(label="✅ Training complete!", state="complete")
            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                st.exception(e)
                return

    results   = st.session_state.training_results
    best_name = st.session_state.best_model_name

    if results is None:
        st.warning("No training results yet. Click **Train All Models** to begin.")
        return

    # ── Results ───────────────────────────────────────────────────────────
    st.success(f"🏆 Best Model: **{best_name}** (ROC-AUC: {results[best_name]['roc_auc']:.4f})")

    # ── Metrics Table ─────────────────────────────────────────────────────
    st.header("Model Comparison")
    metric_rows = []
    for name, r in results.items():
        metric_rows.append({
            "Model":        name,
            "Accuracy":     r["accuracy"],
            "Precision":    r["precision"],
            "Recall":       r["recall"],
            "F1 Score":     r["f1"],
            "ROC-AUC":      r["roc_auc"],
            "Train Time (s)": r["train_time"],
            "Best": "⭐" if name == best_name else "",
        })
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(
        metrics_df.style
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                       color="#d4f1d4")
        .format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall": "{:.4f}", "F1 Score": "{:.4f}", "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Visualisations ────────────────────────────────────────────────────
    st.header("Evaluation Charts")

    tab_roc, tab_pr, tab_cmp, tab_cm, tab_prob = st.tabs(
        ["📈 ROC Curves", "📉 PR Curves", "📊 Model Comparison",
         "🗂️ Confusion Matrix", "🎲 Probability Dist."]
    )

    with tab_roc:
        st.plotly_chart(charts.roc_curve_chart(results), use_container_width=True)

    with tab_pr:
        st.plotly_chart(charts.pr_curve_chart(results), use_container_width=True)

    with tab_cmp:
        st.plotly_chart(charts.model_comparison_bar(results), use_container_width=True)

    with tab_cm:
        model_choice = st.selectbox("Select model", list(results.keys()), key="cm_select")
        cm = results[model_choice]["confusion_matrix"]
        st.plotly_chart(charts.confusion_matrix_chart(cm, model_choice),
                        use_container_width=True)

        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negatives",  tn)
        c2.metric("False Positives", fp)
        c3.metric("False Negatives", fn)
        c4.metric("True Positives",  tp)

    with tab_prob:
        st.plotly_chart(charts.probability_distribution(results, best_name),
                        use_container_width=True)

    # ── Per-model cards ───────────────────────────────────────────────────
    st.header("Detailed Model Metrics")
    for name, r in results.items():
        with st.expander(f"{'⭐ ' if name==best_name else ''}{name}"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{r['accuracy']:.4f}")
            c2.metric("Precision", f"{r['precision']:.4f}")
            c3.metric("Recall",    f"{r['recall']:.4f}")
            c4.metric("F1 Score",  f"{r['f1']:.4f}")
            c5.metric("ROC-AUC",   f"{r['roc_auc']:.4f}")
            st.caption(f"Training time: {r['train_time']}s")
