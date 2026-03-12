"""Model Comparison Page — Side-by-side metrics for all trained models."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("Model Comparison")

dataset = st.sidebar.selectbox("Disease Type", ["Diabetes", "Heart Disease"])
ds_key = "diabetes" if dataset == "Diabetes" else "heart"

# Check for saved comparison results
outputs_dir = PROJECT_ROOT / "outputs"
comparison_file = outputs_dir / f"{ds_key}_model_comparison.csv"

if comparison_file.exists():
    df = pd.read_csv(comparison_file)
    st.subheader(f"{dataset} — Model Performance")

    # Metrics table
    st.dataframe(df, use_container_width=True)

    # Bar chart comparison
    metrics = [c for c in df.columns if c != "Model"]
    fig = go.Figure()
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for i, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group",
        title=f"{dataset} Model Comparison",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Best model highlight
    best = df.loc[df["F1-Macro"].idxmax()]
    st.success(f"**Best Model:** {best['Model']} with F1-Macro of {best['F1-Macro']:.4f}")

else:
    st.warning(
        "No comparison results found. Please run the model training and evaluation "
        "notebooks first to generate comparison data."
    )
    st.markdown("""
    **Steps:**
    1. Run `notebooks/03_model_training.ipynb`
    2. Run `notebooks/04_evaluation.ipynb`
    3. Return to this page
    """)
