"""SHAP visualization wrappers for Streamlit."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def plot_shap_waterfall(feature_names: list[str], shap_values: list[float], title: str = "SHAP Feature Impact"):
    """Display a horizontal bar chart of SHAP values (waterfall-style)."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values,
    }).sort_values("SHAP Value", key=abs, ascending=True)

    fig = go.Figure(go.Bar(
        x=df["SHAP Value"],
        y=df["Feature"],
        orientation="h",
        marker_color=["#dc3545" if v > 0 else "#28a745" for v in df["SHAP Value"]],
        text=[f"{v:+.3f}" for v in df["SHAP Value"]],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (impact on prediction)",
        height=max(300, len(feature_names) * 30),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(feature_names: list[str], importance_values: list[float], title: str = "Global Feature Importance"):
    """Display a bar chart of mean absolute SHAP values."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values,
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker_color="#4e79a7",
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP Value|",
        height=max(300, len(feature_names) * 30),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
