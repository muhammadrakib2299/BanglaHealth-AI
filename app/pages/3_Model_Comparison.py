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

st.markdown("""
<div class="toolbar">
    <div class="tb-title">📈 Model Comparison</div>
    <div class="tb-right">Side-by-side performance analysis</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Configuration**")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"])

ds_key = "diabetes" if dataset == "Diabetes" else "heart"
comp_file = PROJECT_ROOT / "outputs" / f"{ds_key}_model_comparison.csv"

if comp_file.exists():
    df = pd.read_csv(comp_file)
    best = df.loc[df["F1-Macro"].idxmax()]
    metrics = [c for c in df.columns if c != "Model"]

    # Best model KPI
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#2563EB;">{best["Model"]}</div><div class="kpi-label">Best Model</div></div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#059669;">{best["F1-Macro"]:.4f}</div><div class="kpi-label">F1-Macro</div></div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#059669;">{best["ROC-AUC"]:.4f}</div><div class="kpi-label">ROC-AUC</div></div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#2563EB;">{len(df)}</div><div class="kpi-label">Models Trained</div></div></div>', unsafe_allow_html=True)

    # Table + Charts
    col_table, col_chart = st.columns([2, 3])

    with col_table:
        st.markdown('<div class="panel"><div class="panel-header">Performance Metrics</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
        st.dataframe(
            df.style.highlight_max(subset=metrics, color="#DBEAFE"),
            use_container_width=True, hide_index=True, height=200,
        )
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_chart:
        colors = ["#2563EB", "#0891B2", "#6366F1", "#8B5CF6"]
        fig = go.Figure()
        for i, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(go.Bar(
                name=row["Model"], x=metrics, y=[row[m] for m in metrics],
                marker_color=colors[i % len(colors)],
                text=[f"{row[m]:.3f}" for m in metrics], textposition="outside", textfont=dict(size=10),
            ))
        fig.update_layout(
            barmode="group", height=260,
            yaxis=dict(range=[0, 1.08], showgrid=True, gridcolor="#F1F5F9", tickfont=dict(size=10)),
            xaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Radar chart
    st.markdown('<div class="panel"><div class="panel-header">Model Radar Comparison</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
    fig2 = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        cats = metrics + [metrics[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=row["Model"],
            fill="toself", opacity=0.5, line=dict(color=colors[i % len(colors)]),
        ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9))),
        height=350, paper_bgcolor="white", margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=10)),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div></div>', unsafe_allow_html=True)

else:
    st.warning("No comparison data. Run model training notebooks first.")
