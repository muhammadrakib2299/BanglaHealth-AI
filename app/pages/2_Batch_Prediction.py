"""Batch Prediction — CSV upload bulk risk scoring."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Batch Prediction", page_icon="📊", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="toolbar">
    <div class="tb-title">📊 Batch Prediction</div>
    <div class="tb-right">Upload CSV &rarr; bulk risk scoring</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Configuration**")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"])
    model_name = st.selectbox("ML Model", ["xgboost", "random_forest", "lightgbm", "logistic_regression"])

uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown(f'<div class="panel"><div class="panel-header">Input Data — {len(df)} records</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True, height=200)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_diabetes_features, add_heart_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            ds_key = "heart" if dataset == "Heart Disease" else "diabetes"
            model = load_model(model_name, ds_key)
            scaler = load_scaler(ds_key)

            if dataset == "Diabetes":
                df_feat = add_diabetes_features(df.copy())
                fcols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                         "DiabetesPedigreeFunction", "Age", "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory"]
            else:
                df_feat = add_heart_features(df.copy())
                fcols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                         "thalach", "exang", "oldpeak", "slope", "ca", "thal", "BPxHR", "CholAge", "HeartStress"]

            Xs = pd.DataFrame(scaler.transform(df_feat[fcols]), columns=fcols)
            preds = model.predict(Xs)
            proba = model.predict_proba(Xs)

            results = df.copy()
            results["Risk_Level"] = [RISK_LABEL_MAP[p] for p in preds]
            results["Conf_Low"] = proba[:, 0].round(3)
            results["Conf_Med"] = proba[:, 1].round(3)
            results["Conf_High"] = proba[:, 2].round(3)

            low_n, med_n, high_n = int((preds == 0).sum()), int((preds == 1).sum()), int((preds == 2).sum())

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            for col, label, val, color in [(k1, "Total", len(df), "#2563EB"), (k2, "Low Risk", low_n, "#059669"), (k3, "Medium Risk", med_n, "#D97706"), (k4, "High Risk", high_n, "#DC2626")]:
                with col:
                    st.markdown(f'<div class="panel"><div class="kpi"><div class="kpi-value" style="color:{color};">{val}</div><div class="kpi-label">{label}</div></div></div>', unsafe_allow_html=True)

            # Charts + table
            ch1, ch2 = st.columns([1, 2])
            with ch1:
                fig = go.Figure(go.Pie(
                    labels=["Low", "Medium", "High"], values=[low_n, med_n, high_n],
                    marker_colors=["#059669", "#D97706", "#DC2626"],
                    hole=0.5, textinfo="label+value", textfont=dict(size=11),
                ))
                fig.update_layout(height=240, margin=dict(l=5, r=5, t=5, b=5),
                                  paper_bgcolor="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with ch2:
                st.markdown('<div class="panel"><div class="panel-header">Prediction Results</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True, hide_index=True, height=220)
                st.markdown('</div></div>', unsafe_allow_html=True)

            st.download_button("📥 Export Results CSV", results.to_csv(index=False),
                               f"batch_{ds_key}.csv", "text/csv", use_container_width=True)

        except FileNotFoundError:
            st.error("Models not trained. Run notebooks first.")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="panel"><div class="panel-header">Diabetes CSV Columns</div>
        <div class="panel-body" style="font-size:12px; color:#475569;">
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        </div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="panel"><div class="panel-header">Heart Disease CSV Columns</div>
        <div class="panel-body" style="font-size:12px; color:#475569;">
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        </div></div>
        """, unsafe_allow_html=True)
