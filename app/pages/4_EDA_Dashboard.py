"""EDA Dashboard — Interactive exploration of training datasets."""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="EDA Dashboard", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .page-header {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        border-radius: 12px; padding: 28px 32px; color: white; margin-bottom: 24px;
    }
    .page-header h2 { margin: 0; color: white; font-size: 28px; }
    .page-header p { margin: 4px 0 0 0; color: #90CAF9; font-size: 15px; }
    .stat-card {
        background: white; border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid #E2E8F0;
    }
    .stat-card h3 { margin: 0; font-size: 13px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-card .val { font-size: 36px; font-weight: 700; color: #0077B6; margin: 4px 0; }
    .stat-card .sub { font-size: 12px; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>🔍 Exploratory Data Analysis</h2>
    <p>Explore the training datasets with interactive visualizations</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    dataset = st.selectbox("Dataset", ["Diabetes", "Heart Disease"])

try:
    from src.data_loader import load_diabetes, load_heart

    if dataset == "Diabetes":
        df = load_diabetes()
        target_col = "Outcome"
        numeric_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                        "DiabetesPedigreeFunction", "Age", "Pregnancies"]
    else:
        df = load_heart()
        target_col = "target"
        numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # Overview cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><h3>Samples</h3><div class="val">{df.shape[0]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><h3>Features</h3><div class="val">{df.shape[1] - 1}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><h3>Target Classes</h3><div class="val">{df[target_col].nunique()}</div></div>', unsafe_allow_html=True)
    with c4:
        missing = int(df.isin([0]).sum().sum()) if dataset == "Diabetes" else int(df.isnull().sum().sum())
        label = "Zero Values" if dataset == "Diabetes" else "Missing"
        st.markdown(f'<div class="stat-card"><h3>{label}</h3><div class="val">{missing}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Scatter Plots", "Raw Data"])

    with tab1:
        # Target distribution
        fig = px.histogram(df, x=target_col, color=target_col,
                           title=f"{dataset} — Target Distribution",
                           color_discrete_sequence=["#0077B6", "#DC2626"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#F1F5F9"))
        st.plotly_chart(fig, use_container_width=True)

        # Feature distribution
        selected_feature = st.selectbox("Select Feature", numeric_cols)
        fig = px.histogram(df, x=selected_feature, color=target_col, marginal="box",
                           title=f"{selected_feature} Distribution by {target_col}",
                           color_discrete_sequence=["#0077B6", "#DC2626"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#F1F5F9"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        title="Feature Correlation Heatmap",
                        color_continuous_scale="RdBu_r")
        fig.update_layout(height=550, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col_x, col_y = st.columns(2)
        with col_x:
            x_feat = st.selectbox("X-axis", numeric_cols, index=0)
        with col_y:
            y_feat = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1))

        fig = px.scatter(df, x=x_feat, y=y_feat, color=target_col,
                         title=f"{x_feat} vs {y_feat}",
                         color_discrete_sequence=["#0077B6", "#DC2626"],
                         opacity=0.7)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                          yaxis=dict(showgrid=True, gridcolor="#F1F5F9"))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("#### Full Dataset")
        st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.warning("Dataset files not found. Ensure `data/raw/diabetes.csv` and `data/raw/heart.csv` exist.")
except Exception as e:
    st.error(f"Error loading data: {e}")
