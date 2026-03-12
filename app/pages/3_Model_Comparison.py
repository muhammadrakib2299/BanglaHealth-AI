"""Model Comparison — Side-by-side performance metrics."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

st.header("Model Comparison")
st.caption("Side-by-side performance analysis")

with st.sidebar:
    st.markdown("**Configuration**")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"])

ds_key = "diabetes" if dataset == "Diabetes" else "heart"
comp_file = PROJECT_ROOT / "outputs" / f"{ds_key}_model_comparison.csv"

if comp_file.exists():
    df = pd.read_csv(comp_file)
    metrics = [c for c in df.columns if c != "Model"]
    best = df.loc[df["F1-Macro"].idxmax()]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Model", best["Model"])
    k2.metric("F1-Macro", f"{best['F1-Macro']:.4f}")
    k3.metric("ROC-AUC", f"{best['ROC-AUC']:.4f}")
    k4.metric("Models Trained", len(df))

    # Table + Bar chart
    col_t, col_c = st.columns([2, 3])
    with col_t:
        st.subheader("Metrics Table")
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col_c:
        colors = ["#2563EB", "#0891B2", "#6366F1", "#8B5CF6"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(go.Bar(
                name=row["Model"], x=metrics, y=[row[m] for m in metrics],
                marker_color=colors[i % len(colors)],
                text=[f"{row[m]:.3f}" for m in metrics], textposition="outside", textfont=dict(size=10),
            ))
        fig.update_layout(
            barmode="group", height=320,
            yaxis=dict(range=[0, 1.08], showgrid=True, gridcolor="#F1F5F9"),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Radar chart
    st.subheader("Radar Comparison")
    fig2 = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        cats = metrics + [metrics[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=row["Model"],
            fill="toself", opacity=0.5, line=dict(color=colors[i % len(colors)]),
        ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400, paper_bgcolor="white", margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

else:
    st.warning("No comparison data. Run model training notebooks first.")
