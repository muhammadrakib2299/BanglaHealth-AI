"""EDA Dashboard — Interactive dataset exploration."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="EDA Dashboard", page_icon="🔍", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="toolbar">
    <div class="tb-title">🔍 Exploratory Data Analysis</div>
    <div class="tb-right">Interactive dataset exploration</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Configuration**")
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

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#2563EB;">{df.shape[0]}</div><div class="kpi-label">Samples</div></div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#2563EB;">{df.shape[1]-1}</div><div class="kpi-label">Features</div></div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#2563EB;">{df[target_col].nunique()}</div><div class="kpi-label">Classes</div></div></div>', unsafe_allow_html=True)
    with k4:
        pos_rate = df[target_col].mean()
        st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:#D97706;">{pos_rate:.0%}</div><div class="kpi-label">Positive Rate</div></div></div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Scatter", "Data"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.histogram(df, x=target_col, color=target_col,
                               color_discrete_sequence=["#2563EB", "#DC2626"])
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                              plot_bgcolor="white", paper_bgcolor="white",
                              showlegend=False, title=dict(text="Target Distribution", font=dict(size=12)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            feat = st.selectbox("Feature", numeric_cols)
            fig = px.histogram(df, x=feat, color=target_col, marginal="box",
                               color_discrete_sequence=["#2563EB", "#DC2626"])
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                              plot_bgcolor="white", paper_bgcolor="white",
                              title=dict(text=f"{feat} by {target_col}", font=dict(size=12)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        sc1, sc2 = st.columns(2)
        with sc1:
            x_feat = st.selectbox("X-axis", numeric_cols, index=0)
        with sc2:
            y_feat = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig = px.scatter(df, x=x_feat, y=y_feat, color=target_col,
                         color_discrete_sequence=["#2563EB", "#DC2626"], opacity=0.7)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        st.markdown('<div class="panel"><div class="panel-header">Statistical Summary</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True, height=200)
        st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-header">Raw Data</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
        st.markdown('</div></div>', unsafe_allow_html=True)

except FileNotFoundError:
    st.warning("Datasets not found in data/raw/.")
except Exception as e:
    st.error(f"Error: {e}")
