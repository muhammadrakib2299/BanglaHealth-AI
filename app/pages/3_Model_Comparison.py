"""Model Comparison — Side-by-side metrics for all trained models."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .page-header {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        border-radius: 12px; padding: 28px 32px; color: white; margin-bottom: 24px;
    }
    .page-header h2 { margin: 0; color: white; font-size: 28px; }
    .page-header p { margin: 4px 0 0 0; color: #90CAF9; font-size: 15px; }
    .best-model-card {
        background: linear-gradient(135deg, #059669, #047857);
        border-radius: 12px; padding: 20px 24px; color: white; margin: 16px 0;
    }
    .best-model-card h3 { margin: 0 0 4px 0; font-size: 14px; color: #A7F3D0; text-transform: uppercase; letter-spacing: 1px; }
    .best-model-card .model-name { font-size: 24px; font-weight: 700; color: white; }
    .best-model-card .score { font-size: 16px; color: #D1FAE5; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>📈 Model Comparison</h2>
    <p>Compare performance metrics across all trained machine learning models</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"])

ds_key = "diabetes" if dataset == "Diabetes" else "heart"
outputs_dir = PROJECT_ROOT / "outputs"
comparison_file = outputs_dir / f"{ds_key}_model_comparison.csv"

if comparison_file.exists():
    df = pd.read_csv(comparison_file)

    # Best model highlight
    best = df.loc[df["F1-Macro"].idxmax()]
    st.markdown(f"""
    <div class="best-model-card">
        <h3>Best Model — {dataset}</h3>
        <div class="model-name">{best['Model']}</div>
        <div class="score">F1-Macro: {best['F1-Macro']:.4f} &nbsp;|&nbsp; ROC-AUC: {best['ROC-AUC']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics table
    st.markdown("#### Performance Metrics")
    st.dataframe(df.style.highlight_max(subset=["F1-Macro", "ROC-AUC", "Precision (macro)", "Recall (macro)"],
                                         color="#D1FAE5"),
                 use_container_width=True)

    # Grouped bar chart
    metrics = [c for c in df.columns if c != "Model"]
    colors = ["#0077B6", "#00B4D8", "#48CAE4", "#90E0EF"]

    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(
            name=row["Model"], x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[i % len(colors)],
            text=[f"{row[m]:.3f}" for m in metrics],
            textposition="outside", textfont=dict(size=11),
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"{dataset} — Model Performance Comparison", font=dict(size=18)),
        yaxis=dict(range=[0, 1.08], title="Score", showgrid=True, gridcolor="#F1F5F9"),
        height=450, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    fig2 = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        cats = metrics + [metrics[0]]
        fig2.add_trace(go.Scatterpolar(r=vals, theta=cats, name=row["Model"],
                                        fill="toself", opacity=0.6,
                                        line=dict(color=colors[i % len(colors)])))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=dict(text="Model Radar Chart", font=dict(size=18)),
        height=450, paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("No comparison results found. Train models first by running the notebooks.")
