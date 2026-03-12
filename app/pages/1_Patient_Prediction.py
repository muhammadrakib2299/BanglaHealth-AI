"""Patient Prediction — Single patient risk assessment."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Patient Prediction", page_icon="🩺", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

BG = {"Low": "#059669", "Medium": "#D97706", "High": "#DC2626"}

# ── Toolbar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="toolbar">
    <div class="tb-title">🩺 Patient Risk Prediction</div>
    <div class="tb-right">Single patient assessment with SHAP explanation</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Configuration**")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"])
    model_name = st.selectbox("ML Model", ["xgboost", "random_forest", "lightgbm", "logistic_regression"])


def show_results(risk_level, confidence, feature_cols, X_scaled, model, risk_class,
                 patient, dataset_type, patient_data_raw):
    """Display prediction results in ERP style."""

    # Top row: Risk result + confidence
    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        st.markdown(f"""
        <div class="risk-result" style="background: {BG[risk_level]};">
            <div class="rl-label">Risk Level</div>
            <h2>{risk_level}</h2>
            <div class="rl-scores">
                L: {confidence['Low']:.0%} &nbsp; M: {confidence['Medium']:.0%} &nbsp; H: {confidence['High']:.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        fig = go.Figure(go.Bar(
            x=list(confidence.values()), y=["Low", "Medium", "High"], orientation="h",
            marker_color=["#059669", "#D97706", "#DC2626"],
            text=[f"{v:.0%}" for v in confidence.values()],
            textposition="inside", textfont=dict(color="white", size=12),
        ))
        fig.update_layout(
            height=140, margin=dict(l=0, r=10, t=4, b=4),
            xaxis=dict(range=[0, 1], visible=False), yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with r3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=confidence["High"] * 100,
            number=dict(suffix="%", font=dict(size=24)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=9)),
                bar=dict(color="#DC2626"),
                steps=[
                    dict(range=[0, 33], color="#D1FAE5"),
                    dict(range=[33, 66], color="#FEF3C7"),
                    dict(range=[66, 100], color="#FEE2E2"),
                ],
            ),
        ))
        fig.update_layout(height=140, margin=dict(l=15, r=15, t=8, b=0), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # SHAP + Alerts row
    sv = None
    col_shap, col_alerts = st.columns([3, 2])

    with col_shap:
        st.markdown('<div class="panel"><div class="panel-header">Feature Impact (SHAP)</div><div class="panel-body" style="padding:4px 8px;">', unsafe_allow_html=True)
        try:
            from src.explainer import get_shap_explainer
            explainer = get_shap_explainer(model)
            shap_vals = explainer(X_scaled)
            sv = shap_vals.values[0, :, risk_class] if len(shap_vals.shape) == 3 else shap_vals.values[0]

            shap_df = pd.DataFrame({"Feature": feature_cols, "SHAP": sv}).sort_values("SHAP", key=abs, ascending=True)
            fig = go.Figure(go.Bar(
                x=shap_df["SHAP"], y=shap_df["Feature"], orientation="h",
                marker_color=["#DC2626" if v > 0 else "#059669" for v in shap_df["SHAP"]],
                text=[f"{v:+.3f}" for v in shap_df["SHAP"]], textposition="outside", textfont=dict(size=10),
            ))
            fig.update_layout(
                height=max(240, len(feature_cols) * 22), margin=dict(l=0, r=30, t=4, b=4),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#CBD5E1", title="", tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.caption(f"SHAP unavailable: {e}")
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_alerts:
        st.markdown('<div class="panel"><div class="panel-header">Clinical Findings</div>', unsafe_allow_html=True)
        alerts = []
        shap_dict = None
        try:
            from src.explainer import generate_clinical_alerts
            if sv is not None:
                alerts = generate_clinical_alerts(
                    patient_data=patient.iloc[0].to_dict(),
                    shap_values_patient=sv, feature_names=feature_cols,
                    dataset_type=dataset_type,
                )
                shap_dict = {f: round(float(v), 4) for f, v in zip(feature_cols, sv)}
                if alerts:
                    for a in alerts:
                        css = "alert-high" if a["severity"] == "high" else "alert-ok"
                        icon = "🔴" if a["severity"] == "high" else "🟢"
                        st.markdown(f'<div class="alert-row {css}"><span class="a-icon">{icon}</span><div><span class="a-feat">{a["feature"]}</span><br><span class="a-msg">{a["message"]}</span></div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="panel-body"><span style="color:#64748B; font-size:12px;">No alerts generated.</span></div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="panel-body"><span style="color:#64748B; font-size:12px;">Alerts unavailable.</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # PDF
        try:
            from app.components.pdf_report import generate_patient_pdf
            pdf_bytes = generate_patient_pdf(
                patient_data=patient_data_raw, risk_level=risk_level,
                confidence=confidence, alerts=alerts,
                dataset_type=dataset_type, shap_values=shap_dict,
            )
            st.download_button("📄 Export PDF Report", pdf_bytes,
                               f"report_{dataset_type}_{risk_level.lower()}.pdf",
                               "application/pdf", use_container_width=True)
        except Exception:
            pass


# ── Diabetes ─────────────────────────────────────────────────────────────────
if dataset == "Diabetes":
    st.markdown('<div class="panel"><div class="panel-header">Patient Data Entry — Diabetes</div><div class="panel-body">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 0, 300, 120)
    with c2:
        blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 200, 72)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29)
    with c3:
        insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
    with c4:
        dpf = st.slider("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01)
        age = st.slider("Age", 1, 120, 30)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_diabetes_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            raw = {"Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
                   "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
                   "DiabetesPedigreeFunction": dpf, "Age": age}
            patient = add_diabetes_features(pd.DataFrame([raw]))
            fcols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                     "DiabetesPedigreeFunction", "Age", "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory"]
            model = load_model(model_name, "diabetes")
            scaler = load_scaler("diabetes")
            Xs = pd.DataFrame(scaler.transform(patient[fcols]), columns=fcols)
            rc = int(model.predict(Xs)[0])
            pr = model.predict_proba(Xs)[0]
            rl = RISK_LABEL_MAP[rc]
            conf = {"Low": float(pr[0]), "Medium": float(pr[1]), "High": float(pr[2])}
            show_results(rl, conf, fcols, Xs, model, rc, patient, "diabetes", raw)
        except FileNotFoundError:
            st.error("Models not trained. Run notebooks first.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Heart Disease ────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="panel"><div class="panel-header">Patient Data Entry — Heart Disease</div><div class="panel-body">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        h_age = st.slider("Age", 1, 120, 55)
        h_sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        h_cp = st.selectbox("Chest Pain", [0, 1, 2, 3], format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x])
    with c2:
        h_trestbps = st.slider("Resting BP", 0, 300, 130)
        h_chol = st.slider("Cholesterol", 0, 600, 250)
        h_fbs = st.selectbox("Fasting BS>120", [0, 1], format_func=lambda x: "Yes" if x else "No")
        h_restecg = st.selectbox("Rest ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T", "LV Hypertrophy"][x])
    with c3:
        h_thalach = st.slider("Max Heart Rate", 0, 250, 150)
        h_exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
        h_oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)
    with c4:
        h_slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Up", "Flat", "Down"][x])
        h_ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
        h_thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed", "Reversible", "Other"][x])
    st.markdown('</div></div>', unsafe_allow_html=True)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_heart_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            raw = {"age": h_age, "sex": h_sex, "cp": h_cp, "trestbps": h_trestbps,
                   "chol": h_chol, "fbs": h_fbs, "restecg": h_restecg,
                   "thalach": h_thalach, "exang": h_exang, "oldpeak": h_oldpeak,
                   "slope": h_slope, "ca": h_ca, "thal": h_thal}
            patient = add_heart_features(pd.DataFrame([raw]))
            fcols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                     "BPxHR", "CholAge", "HeartStress"]
            model = load_model(model_name, "heart")
            scaler = load_scaler("heart")
            Xs = pd.DataFrame(scaler.transform(patient[fcols]), columns=fcols)
            rc = int(model.predict(Xs)[0])
            pr = model.predict_proba(Xs)[0]
            rl = RISK_LABEL_MAP[rc]
            conf = {"Low": float(pr[0]), "Medium": float(pr[1]), "High": float(pr[2])}
            show_results(rl, conf, fcols, Xs, model, rc, patient, "heart", raw)
        except FileNotFoundError:
            st.error("Models not trained. Run notebooks first.")
        except Exception as e:
            st.error(f"Error: {e}")
