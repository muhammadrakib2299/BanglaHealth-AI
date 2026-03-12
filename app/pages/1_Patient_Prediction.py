"""Patient Prediction — Single patient risk assessment with SHAP explanation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Patient Prediction", page_icon="🩺", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}

    .page-header {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        border-radius: 12px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
    }
    .page-header h2 { margin: 0; color: white; font-size: 28px; }
    .page-header p { margin: 4px 0 0 0; color: #90CAF9; font-size: 15px; }

    .risk-card {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        color: white;
        margin: 8px 0 24px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .risk-card h1 { margin: 0; font-size: 52px; color: white; font-weight: 800; }
    .risk-card .label { font-size: 14px; text-transform: uppercase; letter-spacing: 2px;
                         opacity: 0.9; margin-bottom: 8px; }
    .risk-card .scores { font-size: 15px; opacity: 0.85; margin-top: 12px; }

    .alert-high {
        background: #FEF2F2; border-left: 4px solid #DC2626;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0;
    }
    .alert-normal {
        background: #F0FDF4; border-left: 4px solid #16A34A;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0;
    }
    .alert-high strong, .alert-normal strong { font-size: 14px; }
    .alert-high p, .alert-normal p { margin: 2px 0 0 0; font-size: 13px; color: #475569; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h2>🩺 Patient Risk Prediction</h2>
    <p>Enter patient clinical data to get an AI-powered risk assessment with SHAP explanations</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"],
                           help="Select which disease to assess")
    model_name = st.selectbox("ML Model", ["xgboost", "random_forest", "lightgbm", "logistic_regression"],
                              help="Choose the prediction model")
    st.markdown("---")
    st.info("Fill in the patient data and click **Predict** to get results.")


BG_COLORS = {
    "Low": "linear-gradient(135deg, #059669, #047857)",
    "Medium": "linear-gradient(135deg, #D97706, #B45309)",
    "High": "linear-gradient(135deg, #DC2626, #B91C1C)",
}


def show_results(risk_level, confidence, feature_cols, X_scaled, model, risk_class,
                 patient, dataset_type, patient_data_raw):
    """Display prediction results, charts, alerts, and PDF download."""
    # Risk card
    st.markdown(f"""
    <div class="risk-card" style="background: {BG_COLORS[risk_level]};">
        <div class="label">Predicted Risk Level</div>
        <h1>{risk_level} Risk</h1>
        <div class="scores">
            Confidence &mdash; Low: {confidence['Low']:.1%} &nbsp;|&nbsp;
            Medium: {confidence['Medium']:.1%} &nbsp;|&nbsp;
            High: {confidence['High']:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Confidence bar
    with col_a:
        fig = go.Figure(go.Bar(
            x=list(confidence.values()), y=list(confidence.keys()), orientation="h",
            marker_color=["#059669", "#D97706", "#DC2626"],
            text=[f"{v:.1%}" for v in confidence.values()],
            textposition="inside", textfont=dict(color="white", size=14),
        ))
        fig.update_layout(
            title=dict(text="Prediction Confidence", font=dict(size=16)),
            xaxis=dict(range=[0, 1], title="Probability", showgrid=True, gridcolor="#F1F5F9"),
            height=220, margin=dict(l=0, r=20, t=40, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP chart
    sv = None
    with col_b:
        try:
            from src.explainer import get_shap_explainer
            explainer = get_shap_explainer(model)
            shap_vals = explainer(X_scaled)
            sv = shap_vals.values[0, :, risk_class] if len(shap_vals.shape) == 3 else shap_vals.values[0]

            shap_df = pd.DataFrame({"Feature": feature_cols, "SHAP Value": sv}).sort_values("SHAP Value", key=abs, ascending=True)
            fig = go.Figure(go.Bar(
                x=shap_df["SHAP Value"], y=shap_df["Feature"], orientation="h",
                marker_color=["#DC2626" if v > 0 else "#059669" for v in shap_df["SHAP Value"]],
                text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]],
                textposition="outside", textfont=dict(size=11),
            ))
            fig.update_layout(
                title=dict(text="SHAP Feature Impact", font=dict(size=16)),
                xaxis_title="Impact on prediction",
                height=max(280, len(feature_cols) * 28),
                margin=dict(l=0, r=40, t=40, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#94A3B8"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    # Clinical alerts
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
                st.markdown("#### Clinical Findings")
                for alert in alerts:
                    css_class = "alert-high" if alert["severity"] == "high" else "alert-normal"
                    icon = "⚠️" if alert["severity"] == "high" else "✅"
                    st.markdown(
                        f'<div class="{css_class}"><strong>{icon} {alert["feature"]}</strong>'
                        f'<p>{alert["message"]}</p></div>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        pass

    # PDF download
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        from app.components.pdf_report import generate_patient_pdf
        pdf_bytes = generate_patient_pdf(
            patient_data=patient_data_raw, risk_level=risk_level,
            confidence=confidence, alerts=alerts,
            dataset_type=dataset_type, shap_values=shap_dict,
        )
        st.download_button(
            "📄 Download PDF Report", pdf_bytes,
            f"patient_report_{dataset_type}_{risk_level.lower()}.pdf",
            "application/pdf", use_container_width=True,
        )
    except Exception as e:
        st.caption(f"PDF generation unavailable: {e}")


# ── Diabetes ─────────────────────────────────────────────────────────────────
if dataset == "Diabetes":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of pregnancies")
        glucose = st.slider("Glucose (mg/dL)", 0, 300, 120, help="Plasma glucose concentration")
    with col2:
        blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 200, 72, help="Diastolic blood pressure")
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29, help="Triceps skinfold thickness")
    with col3:
        insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80, help="2-Hour serum insulin")
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1, help="Body mass index")
    with col4:
        dpf = st.slider("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01, help="Diabetes pedigree function")
        age = st.slider("Age (years)", 1, 120, 30)

    if st.button("Predict Risk Level", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_diabetes_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            raw_data = {"Pregnancies": pregnancies, "Glucose": glucose,
                        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
                        "Insulin": insulin, "BMI": bmi,
                        "DiabetesPedigreeFunction": dpf, "Age": age}
            patient = add_diabetes_features(pd.DataFrame([raw_data]))
            feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                            "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory"]

            model = load_model(model_name, "diabetes")
            scaler = load_scaler("diabetes")
            X_scaled = pd.DataFrame(scaler.transform(patient[feature_cols]), columns=feature_cols)

            risk_class = int(model.predict(X_scaled)[0])
            proba = model.predict_proba(X_scaled)[0]
            risk_level = RISK_LABEL_MAP[risk_class]
            confidence = {"Low": float(proba[0]), "Medium": float(proba[1]), "High": float(proba[2])}

            show_results(risk_level, confidence, feature_cols, X_scaled, model,
                         risk_class, patient, "diabetes", raw_data)

        except FileNotFoundError:
            st.error("Models not trained yet. Please run the training notebooks first.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Heart Disease ────────────────────────────────────────────────────────────
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        h_age = st.slider("Age (years)", 1, 120, 55)
        h_sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        h_cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
        h_trestbps = st.slider("Resting BP (mmHg)", 0, 300, 130)
        h_chol = st.slider("Cholesterol (mg/dL)", 0, 600, 250)
    with col2:
        h_fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        h_restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                  format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        h_thalach = st.slider("Max Heart Rate", 0, 250, 150)
        h_exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col3:
        h_oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)
        h_slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        h_ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
        h_thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                               format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Other"][x])

    if st.button("Predict Risk Level", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_heart_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            raw_data = {"age": h_age, "sex": h_sex, "cp": h_cp, "trestbps": h_trestbps,
                        "chol": h_chol, "fbs": h_fbs, "restecg": h_restecg,
                        "thalach": h_thalach, "exang": h_exang, "oldpeak": h_oldpeak,
                        "slope": h_slope, "ca": h_ca, "thal": h_thal}
            patient = add_heart_features(pd.DataFrame([raw_data]))
            feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                            "BPxHR", "CholAge", "HeartStress"]

            model = load_model(model_name, "heart")
            scaler = load_scaler("heart")
            X_scaled = pd.DataFrame(scaler.transform(patient[feature_cols]), columns=feature_cols)

            risk_class = int(model.predict(X_scaled)[0])
            proba = model.predict_proba(X_scaled)[0]
            risk_level = RISK_LABEL_MAP[risk_class]
            confidence = {"Low": float(proba[0]), "Medium": float(proba[1]), "High": float(proba[2])}

            show_results(risk_level, confidence, feature_cols, X_scaled, model,
                         risk_class, patient, "heart", raw_data)

        except FileNotFoundError:
            st.error("Models not trained yet. Please run the training notebooks first.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
