"""EDA Dashboard — Interactive exploration of training datasets."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="EDA Dashboard", page_icon="🔍", layout="wide")
st.title("Exploratory Data Analysis")

dataset = st.sidebar.selectbox("Dataset", ["Diabetes", "Heart Disease"])

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

    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Target Classes", df[target_col].nunique())

    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Class distribution
    st.subheader("Target Distribution")
    fig = px.histogram(df, x=target_col, color=target_col,
                       title=f"{dataset} Target Distribution",
                       color_discrete_sequence=["#28a745", "#dc3545"])
    st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature", numeric_cols)
    fig = px.histogram(df, x=selected_feature, color=target_col, marginal="box",
                       title=f"{selected_feature} Distribution by {target_col}",
                       color_discrete_sequence=["#4e79a7", "#e15759"])
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu_r")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.subheader("Feature Scatter Plot")
    col_x, col_y = st.columns(2)
    with col_x:
        x_feat = st.selectbox("X-axis", numeric_cols, index=0)
    with col_y:
        y_feat = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols) - 1))

    fig = px.scatter(df, x=x_feat, y=y_feat, color=target_col,
                     title=f"{x_feat} vs {y_feat}",
                     color_discrete_sequence=["#4e79a7", "#e15759"])
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.warning(
        "Dataset files not found. Please download the datasets to `data/raw/` first.\n\n"
        "- `data/raw/diabetes.csv` — Pima Indians Diabetes\n"
        "- `data/raw/heart.csv` — UCI Heart Disease (Cleveland)"
    )
except Exception as e:
    st.error(f"Error loading data: {e}")
