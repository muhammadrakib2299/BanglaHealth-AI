"""What-If Analysis — Real-time parameter sensitivity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="What-If Analysis", page_icon="🔬", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="toolbar">
    <div class="tb-title">🔬 What-If Analysis</div>
    <div class="tb-right">Adjust parameters &rarr; watch prediction change live</div>
</div>
""", unsafe_allow_html=True)

BG = {"Low": "#059669", "Medium": "#D97706", "High": "#DC2626"}

with st.sidebar:
    st.markdown("**Configuration**")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"], key="wif_ds")
    model_name = st.selectbox("ML Model", ["xgboost", "random_forest", "lightgbm", "logistic_regression"], key="wif_m")

try:
    from src.feature_engineering import add_diabetes_features, add_heart_features
    from src.model import load_model, load_scaler
    from src.preprocessing import RISK_LABEL_MAP

    if dataset == "Diabetes":
        model = load_model(model_name, "diabetes")
        scaler = load_scaler("diabetes")

        st.markdown('<div class="panel"><div class="panel-header">Parameter Controls — Diabetes</div><div class="panel-body">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pregnancies = st.slider("Pregnancies", 0, 17, 3, key="w_p")
            glucose = st.slider("Glucose", 0, 300, 120, key="w_g")
        with c2:
            bp = st.slider("Blood Pressure", 0, 200, 72, key="w_bp")
            skin = st.slider("Skin Thickness", 0, 100, 29, key="w_sk")
        with c3:
            insulin = st.slider("Insulin", 0, 900, 80, key="w_i")
            bmi = st.slider("BMI", 0.0, 70.0, 28.0, 0.5, key="w_bmi")
        with c4:
            dpf = st.slider("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01, key="w_dpf")
            age = st.slider("Age", 1, 120, 35, key="w_age")
        st.markdown('</div></div>', unsafe_allow_html=True)

        patient = add_diabetes_features(pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }]))
        fcols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                 "DiabetesPedigreeFunction", "Age", "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory"]

    else:
        model = load_model(model_name, "heart")
        scaler = load_scaler("heart")

        st.markdown('<div class="panel"><div class="panel-header">Parameter Controls — Heart Disease</div><div class="panel-body">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            h_age = st.slider("Age", 1, 120, 55, key="w_ha")
            h_sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female", key="w_hs")
            h_cp = st.selectbox("Chest Pain", [0,1,2,3], format_func=lambda x: ["Typical","Atypical","Non-anginal","Asymptomatic"][x], key="w_hc")
        with c2:
            h_bp = st.slider("Resting BP", 0, 300, 130, key="w_hbp")
            h_chol = st.slider("Cholesterol", 0, 600, 250, key="w_hch")
            h_fbs = st.selectbox("Fasting BS>120", [0,1], format_func=lambda x: "Yes" if x else "No", key="w_hf")
            h_ecg = st.selectbox("Rest ECG", [0,1,2], format_func=lambda x: ["Normal","ST-T","LV Hypertrophy"][x], key="w_he")
        with c3:
            h_hr = st.slider("Max Heart Rate", 0, 250, 150, key="w_hhr")
            h_exang = st.selectbox("Exercise Angina", [0,1], format_func=lambda x: "Yes" if x else "No", key="w_hex")
            h_old = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1, key="w_ho")
        with c4:
            h_slope = st.selectbox("ST Slope", [0,1,2], format_func=lambda x: ["Up","Flat","Down"][x], key="w_hsl")
            h_ca = st.selectbox("Major Vessels", [0,1,2,3,4], key="w_hca")
            h_thal = st.selectbox("Thalassemia", [0,1,2,3], format_func=lambda x: ["Normal","Fixed","Reversible","Other"][x], key="w_ht")
        st.markdown('</div></div>', unsafe_allow_html=True)

        patient = add_heart_features(pd.DataFrame([{
            "age": h_age, "sex": h_sex, "cp": h_cp, "trestbps": h_bp,
            "chol": h_chol, "fbs": h_fbs, "restecg": h_ecg,
            "thalach": h_hr, "exang": h_exang, "oldpeak": h_old,
            "slope": h_slope, "ca": h_ca, "thal": h_thal,
        }]))
        fcols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                 "BPxHR", "CholAge", "HeartStress"]

    # Predict
    Xs = pd.DataFrame(scaler.transform(patient[fcols]), columns=fcols)
    rc = int(model.predict(Xs)[0])
    pr = model.predict_proba(Xs)[0]
    rl = RISK_LABEL_MAP[rc]

    # Results row
    r1, r2, r3, r4 = st.columns([1, 1, 1, 2])

    with r1:
        st.markdown(f"""
        <div class="risk-result" style="background: {BG[rl]};">
            <div class="rl-label">Live Result</div>
            <h2>{rl}</h2>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="panel"><div class="kpi">
            <div class="kpi-value" style="color:#059669;">{pr[0]:.0%}</div><div class="kpi-label">Low</div>
        </div></div>
        <div class="panel"><div class="kpi">
            <div class="kpi-value" style="color:#D97706;">{pr[1]:.0%}</div><div class="kpi-label">Medium</div>
        </div></div>
        <div class="panel"><div class="kpi">
            <div class="kpi-value" style="color:#DC2626;">{pr[2]:.0%}</div><div class="kpi-label">High</div>
        </div></div>
        """, unsafe_allow_html=True)

    with r3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pr[2]*100, number=dict(suffix="%", font=dict(size=20)),
            gauge=dict(axis=dict(range=[0,100], tickfont=dict(size=8)), bar=dict(color="#DC2626"),
                       steps=[dict(range=[0,33], color="#D1FAE5"), dict(range=[33,66], color="#FEF3C7"), dict(range=[66,100], color="#FEE2E2")]),
        ))
        fig.update_layout(height=180, margin=dict(l=15, r=15, t=10, b=0), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with r4:
        try:
            from src.explainer import get_shap_explainer
            explainer = get_shap_explainer(model)
            sv_obj = explainer(Xs)
            sv = sv_obj.values[0, :, rc] if len(sv_obj.shape) == 3 else sv_obj.values[0]
            sdf = pd.DataFrame({"Feature": fcols, "SHAP": sv}).sort_values("SHAP", key=abs, ascending=True)

            fig = go.Figure(go.Bar(
                x=sdf["SHAP"], y=sdf["Feature"], orientation="h",
                marker_color=["#DC2626" if v > 0 else "#059669" for v in sdf["SHAP"]],
                text=[f"{v:+.3f}" for v in sdf["SHAP"]], textposition="outside", textfont=dict(size=9),
            ))
            fig.update_layout(
                height=max(220, len(fcols)*20), margin=dict(l=0, r=25, t=4, b=4),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1", tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("SHAP unavailable")

except FileNotFoundError:
    st.error("Models not trained. Run notebooks first.")
except Exception as e:
    st.error(f"Error: {e}")
