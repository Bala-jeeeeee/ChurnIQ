"""ChurnIQ – Plotly chart factory."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

_PALETTE = px.colors.qualitative.Set2
_CHURN_COLORS = {0: "#2ecc71", 1: "#e74c3c"}


def churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    counts["Label"] = counts["Churn"].map({0: "Retained", 1: "Churned"})
    fig = px.pie(counts, names="Label", values="Count",
                 color="Label",
                 color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
                 title="Churn Distribution", hole=0.4)
    fig.update_traces(textinfo="percent+label")
    return fig


def churn_by_category(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = df.groupby([col, "Churn"]).size().reset_index(name="Count")
    grp["Status"] = grp["Churn"].map({0: "Retained", 1: "Churned"})
    fig = px.bar(grp, x=col, y="Count", color="Status",
                 color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
                 barmode="group", title=title)
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def churn_rate_by_category(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = df.groupby(col)["Churn"].mean().reset_index()
    grp.columns = [col, "Churn Rate"]
    grp["Churn Rate %"] = (grp["Churn Rate"] * 100).round(2)
    grp = grp.sort_values("Churn Rate %", ascending=False)
    fig = px.bar(grp, x=col, y="Churn Rate %",
                 color="Churn Rate %", color_continuous_scale="RdYlGn_r",
                 title=title, text="Churn Rate %")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def numeric_histogram(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = px.histogram(df, x=col, color="Churn",
                       color_discrete_map=_CHURN_COLORS,
                       marginal="box", opacity=0.75,
                       title=title, barmode="overlay",
                       labels={"Churn": "Status"})
    fig.for_each_trace(lambda t: t.update(name="Retained" if t.name == "0" else "Churned"))
    return fig


def correlation_heatmap(df: pd.DataFrame, num_cols: list) -> go.Figure:
    corr = df[num_cols + ["Churn"]].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Correlation Heatmap", text_auto=".2f",
                    aspect="auto")
    return fig


def revenue_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="Churn", y="total_revenue",
                 color="Churn",
                 color_discrete_map=_CHURN_COLORS,
                 title="Revenue Distribution by Churn Status",
                 labels={"Churn": "Status", "total_revenue": "Total Revenue"})
    fig.update_xaxes(ticktext=["Retained", "Churned"], tickvals=[0, 1])
    return fig


def tenure_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="tenure_months", color="Churn",
                       color_discrete_map=_CHURN_COLORS,
                       nbins=30, barmode="overlay", opacity=0.7,
                       title="Tenure Distribution by Churn Status")
    fig.for_each_trace(lambda t: t.update(name="Retained" if t.name == "0" else "Churned"))
    return fig


def nps_violin(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(df, x="Churn", y="nps_score",
                    color="Churn",
                    color_discrete_map=_CHURN_COLORS,
                    box=True, points=False,
                    title="NPS Score Distribution by Churn")
    fig.update_xaxes(ticktext=["Retained", "Churned"], tickvals=[0, 1])
    return fig


def csat_bar(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["csat_score", "Churn"]).size().reset_index(name="Count")
    grp["Status"] = grp["Churn"].map({0: "Retained", 1: "Churned"})
    fig = px.bar(grp, x="csat_score", y="Count", color="Status",
                 color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
                 barmode="group", title="CSAT Score vs Churn")
    return fig


def segment_sunburst(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["customer_segment", "contract_type"])["Churn"].agg(
        ["mean", "count"]).reset_index()
    grp.columns = ["Segment", "Contract", "Churn Rate", "Count"]
    fig = px.sunburst(grp, path=["Segment", "Contract"],
                      values="Count", color="Churn Rate",
                      color_continuous_scale="RdYlGn_r",
                      title="Customer Segment & Contract Type vs Churn Rate")
    return fig


# ── Training / Evaluation charts ──────────────────────────────────────────

def confusion_matrix_chart(cm: list, model_name: str) -> go.Figure:
    labels = ["Retained", "Churned"]
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    annots = [[str(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale="Blues", text=annots, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix – {model_name}",
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    return fig


def roc_curve_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Random"))
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name} (AUC={r['roc_auc']:.3f})",
        ))
    fig.update_layout(title="ROC Curves – All Models",
                      xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    return fig


def pr_curve_chart(results: dict) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["recall_curve"], y=r["precision_curve"],
            mode="lines", name=name,
        ))
    fig.update_layout(title="Precision-Recall Curves – All Models",
                      xaxis_title="Recall", yaxis_title="Precision")
    return fig


def model_comparison_bar(results: dict) -> go.Figure:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, r in results.items():
        for m in metrics:
            rows.append({"Model": name, "Metric": m.upper(), "Score": r[m]})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Metric", y="Score", color="Model",
                 barmode="group", title="Model Comparison by Metric",
                 range_y=[0, 1])
    return fig


def probability_distribution(results: dict, best_name: str) -> go.Figure:
    r = results[best_name]
    df = pd.DataFrame({"y_proba": r["y_proba"], "y_test": r["y_test"]})
    fig = px.histogram(df, x="y_proba", color="y_test",
                       color_discrete_map=_CHURN_COLORS,
                       nbins=40, barmode="overlay", opacity=0.7,
                       title=f"Predicted Probability Distribution – {best_name}")
    fig.for_each_trace(lambda t: t.update(name="Retained" if t.name == "0" else "Churned"))
    return fig
