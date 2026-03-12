"""What-If Analysis — Adjust patient values with sliders, see risk change in real-time."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="What-If Analysis", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .page-header {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        border-radius: 12px; padding: 28px 32px; color: white; margin-bottom: 24px;
    }
    .page-header h2 { margin: 0; color: white; font-size: 28px; }
    .page-header p { margin: 4px 0 0 0; color: #90CAF9; font-size: 15px; }
    .live-risk {
        border-radius: 16px; padding: 28px; text-align: center; color: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .live-risk h1 { margin: 0; font-size: 48px; color: white; font-weight: 800; }
    .live-risk .sub { font-size: 13px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85; margin-bottom: 8px; }
    .prob-row {
        display: flex; gap: 12px; margin-top: 16px;
    }
    .prob-item {
        flex: 1; background: white; border-radius: 10px; padding: 14px; text-align: center;
        border: 1px solid #E2E8F0;
    }
    .prob-item .label { font-size: 12px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .prob-item .val { font-size: 24px; font-weight: 700; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>🔬 What-If Analysis</h2>
    <p>Adjust patient values with sliders and watch how the risk prediction changes in real-time</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"], key="wif_ds")
    model_name = st.selectbox("ML Model", ["xgboost", "random_forest", "lightgbm", "logistic_regression"], key="wif_model")

try:
    from src.feature_engineering import add_diabetes_features, add_heart_features
    from src.model import load_model, load_scaler
    from src.preprocessing import RISK_LABEL_MAP

    BG = {"Low": "linear-gradient(135deg, #059669, #047857)",
          "Medium": "linear-gradient(135deg, #D97706, #B45309)",
          "High": "linear-gradient(135deg, #DC2626, #B91C1C)"}
    COLORS = {"Low": "#059669", "Medium": "#D97706", "High": "#DC2626"}

    if dataset == "Diabetes":
        model = load_model(model_name, "diabetes")
        scaler = load_scaler("diabetes")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pregnancies = st.slider("Pregnancies", 0, 17, 3, key="w_preg")
            glucose = st.slider("Glucose (mg/dL)", 0, 300, 120, key="w_gluc")
        with col2:
            blood_pressure = st.slider("Blood Pressure", 0, 200, 72, key="w_bp")
            skin_thickness = st.slider("Skin Thickness", 0, 100, 29, key="w_skin")
        with col3:
            insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80, key="w_ins")
            bmi = st.slider("BMI", 0.0, 70.0, 28.0, 0.5, key="w_bmi")
        with col4:
            dpf = st.slider("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01, key="w_dpf")
            age = st.slider("Age", 1, 120, 35, key="w_age")

        patient = add_diabetes_features(pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }]))
        feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                        "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory"]
    else:
        model = load_model(model_name, "heart")
        scaler = load_scaler("heart")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            h_age = st.slider("Age", 1, 120, 55, key="w_h_age")
            h_sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="w_h_sex")
            h_cp = st.selectbox("Chest Pain", [0, 1, 2, 3],
                                format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x], key="w_h_cp")
        with col2:
            h_trestbps = st.slider("Resting BP", 0, 300, 130, key="w_h_bp")
            h_chol = st.slider("Cholesterol", 0, 600, 250, key="w_h_chol")
            h_fbs = st.selectbox("Fasting BS>120", [0, 1], format_func=lambda x: "Yes" if x else "No", key="w_h_fbs")
            h_restecg = st.selectbox("Rest ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T", "LV Hypertrophy"][x], key="w_h_ecg")
        with col3:
            h_thalach = st.slider("Max Heart Rate", 0, 250, 150, key="w_h_hr")
            h_exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x else "No", key="w_h_exang")
            h_oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1, key="w_h_old")
        with col4:
            h_slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Up", "Flat", "Down"][x], key="w_h_slope")
            h_ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4], key="w_h_ca")
            h_thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                                   format_func=lambda x: ["Normal", "Fixed", "Reversible", "Other"][x], key="w_h_thal")

        patient = add_heart_features(pd.DataFrame([{
            "age": h_age, "sex": h_sex, "cp": h_cp, "trestbps": h_trestbps,
            "chol": h_chol, "fbs": h_fbs, "restecg": h_restecg,
            "thalach": h_thalach, "exang": h_exang, "oldpeak": h_oldpeak,
            "slope": h_slope, "ca": h_ca, "thal": h_thal,
        }]))
        feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                        "BPxHR", "CholAge", "HeartStress"]

    # Predict
    X_scaled = pd.DataFrame(scaler.transform(patient[feature_cols]), columns=feature_cols)
    risk_class = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    risk_level = RISK_LABEL_MAP[risk_class]

    st.markdown("---")

    # Results row
    col_risk, col_charts = st.columns([1, 2])

    with col_risk:
        st.markdown(f"""
        <div class="live-risk" style="background: {BG[risk_level]};">
            <div class="sub">Live Prediction</div>
            <h1>{risk_level}</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-item"><div class="label">Low</div><div class="val" style="color:#059669;">{proba[0]:.1%}</div></div>
            <div class="prob-item"><div class="label">Medium</div><div class="val" style="color:#D97706;">{proba[1]:.1%}</div></div>
            <div class="prob-item"><div class="label">High</div><div class="val" style="color:#DC2626;">{proba[2]:.1%}</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_charts:
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba[2] * 100,
            number=dict(suffix="%"),
            title={"text": "High Risk Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#DC2626"},
                "steps": [
                    {"range": [0, 33], "color": "#D1FAE5"},
                    {"range": [33, 66], "color": "#FEF3C7"},
                    {"range": [66, 100], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "#1B2A4A", "width": 3}, "thickness": 0.8, "value": proba[2] * 100},
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # SHAP
    st.markdown("---")
    st.markdown("#### Feature Impact (SHAP)")
    try:
        from src.explainer import get_shap_explainer
        explainer = get_shap_explainer(model)
        shap_vals = explainer(X_scaled)
        sv = shap_vals.values[0, :, risk_class] if len(shap_vals.shape) == 3 else shap_vals.values[0]

        shap_df = pd.DataFrame({"Feature": feature_cols, "SHAP Value": sv}).sort_values("SHAP Value", key=abs, ascending=True)
        fig = go.Figure(go.Bar(
            x=shap_df["SHAP Value"], y=shap_df["Feature"], orientation="h",
            marker_color=["#DC2626" if v > 0 else "#059669" for v in shap_df["SHAP Value"]],
            text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]], textposition="outside", textfont=dict(size=11),
        ))
        fig.update_layout(
            title=dict(text=f"What's Driving the {risk_level} Risk Prediction?", font=dict(size=16)),
            xaxis_title="SHAP Value (positive = increases risk)",
            height=max(380, len(feature_cols) * 28), margin=dict(l=0, r=40, t=50, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#94A3B8"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red bars push toward higher risk. Green bars push toward lower risk. Adjust sliders above to see changes.")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

except FileNotFoundError:
    st.error("Models not trained yet. Please run the training notebooks first.")
except Exception as e:
    st.error(f"Error: {e}")
