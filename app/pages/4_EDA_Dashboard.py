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

st.header("Exploratory Data Analysis")
st.caption("Interactive dataset exploration")

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Samples", df.shape[0])
    k2.metric("Features", df.shape[1] - 1)
    k3.metric("Classes", df[target_col].nunique())
    k4.metric("Positive Rate", f"{df[target_col].mean():.0%}")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Scatter", "Raw Data"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.histogram(df, x=target_col, color=target_col,
                               color_discrete_sequence=["#2563EB", "#DC2626"])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                              plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                              title=dict(text="Target Distribution", font=dict(size=13)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            feat = st.selectbox("Feature", numeric_cols)
            fig = px.histogram(df, x=feat, color=target_col, marginal="box",
                               color_discrete_sequence=["#2563EB", "#DC2626"])
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                              plot_bgcolor="white", paper_bgcolor="white",
                              title=dict(text=f"{feat} by {target_col}", font=dict(size=13)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        sc1, sc2 = st.columns(2)
        with sc1:
            x_feat = st.selectbox("X-axis", numeric_cols, index=0)
        with sc2:
            y_feat = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig = px.scatter(df, x=x_feat, y=y_feat, color=target_col,
                         color_discrete_sequence=["#2563EB", "#DC2626"], opacity=0.7)
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab4:
        st.subheader("Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        st.subheader("Full Dataset")
        st.dataframe(df, use_container_width=True, hide_index=True)

except FileNotFoundError:
    st.warning("Datasets not found in data/raw/.")
except Exception as e:
    st.error(f"Error: {e}")
