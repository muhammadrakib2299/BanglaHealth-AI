"""Patient Prediction Page — Single patient risk assessment with SHAP explanation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Patient Prediction", page_icon="🩺", layout="wide")
st.title("Patient Risk Prediction")

# Dataset selection
dataset = st.sidebar.selectbox("Select Disease Type", ["Diabetes", "Heart Disease"])
model_name = st.sidebar.selectbox("Select Model", ["xgboost", "lightgbm", "random_forest", "logistic_regression"])

st.sidebar.markdown("---")
st.sidebar.info("Enter patient data below and click **Predict** to get a risk assessment.")


def render_risk_card(risk_level: str, confidence: dict):
    """Display a color-coded risk level card."""
    colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
    color = colors.get(risk_level, "#6c757d")

    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px;
                text-align: center; color: white; margin: 10px 0;">
        <h1 style="margin: 0; color: white;">{risk_level} Risk</h1>
        <p style="margin: 5px 0; font-size: 18px; color: white;">
            Low: {confidence.get('Low', 0):.1%} |
            Medium: {confidence.get('Medium', 0):.1%} |
            High: {confidence.get('High', 0):.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_chart(confidence: dict):
    """Display a horizontal bar chart of class probabilities."""
    fig = go.Figure(go.Bar(
        x=list(confidence.values()),
        y=list(confidence.keys()),
        orientation="h",
        marker_color=["#28a745", "#ffc107", "#dc3545"],
        text=[f"{v:.1%}" for v in confidence.values()],
        textposition="auto",
    ))
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Probability",
        yaxis_title="Risk Level",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


if dataset == "Diabetes":
    st.subheader("Diabetes Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 0, 300, 120)
        blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 200, 72)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29)

    with col2:
        insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
        age = st.slider("Age", 1, 120, 30)

    if st.button("Predict Risk", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_diabetes_features
            from src.model import load_model, load_scaler
            from src.explainer import get_shap_explainer, generate_clinical_alerts
            from src.preprocessing import RISK_LABEL_MAP

            patient = pd.DataFrame([{
                "Pregnancies": pregnancies, "Glucose": glucose,
                "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
                "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age,
            }])

            patient = add_diabetes_features(patient)
            feature_cols = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory",
            ]

            model = load_model(model_name, "diabetes")
            scaler = load_scaler("diabetes")

            X = patient[feature_cols]
            X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

            risk_class = int(model.predict(X_scaled)[0])
            probabilities = model.predict_proba(X_scaled)[0]
            risk_level = RISK_LABEL_MAP[risk_class]

            confidence = {
                "Low": float(probabilities[0]),
                "Medium": float(probabilities[1]),
                "High": float(probabilities[2]),
            }

            # Display results
            render_risk_card(risk_level, confidence)

            col_a, col_b = st.columns(2)
            with col_a:
                render_confidence_chart(confidence)

            # SHAP explanation
            with col_b:
                try:
                    explainer = get_shap_explainer(model)
                    shap_vals = explainer(X_scaled)

                    if len(shap_vals.shape) == 3:
                        sv = shap_vals.values[0, :, risk_class]
                    else:
                        sv = shap_vals.values[0]

                    shap_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "SHAP Value": sv,
                    }).sort_values("SHAP Value", key=abs, ascending=True)

                    fig = go.Figure(go.Bar(
                        x=shap_df["SHAP Value"],
                        y=shap_df["Feature"],
                        orientation="h",
                        marker_color=["#dc3545" if v > 0 else "#28a745" for v in shap_df["SHAP Value"]],
                    ))
                    fig.update_layout(title="SHAP Feature Impact", height=400, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

            # Clinical alerts
            try:
                if 'sv' in dir():
                    alerts = generate_clinical_alerts(
                        patient_data=patient.iloc[0].to_dict(),
                        shap_values_patient=sv,
                        feature_names=feature_cols,
                        dataset_type="diabetes",
                    )
                    st.subheader("Clinical Alerts")
                    for alert in alerts:
                        if alert["severity"] == "high":
                            st.error(f"**{alert['feature']}**: {alert['message']}")
                        else:
                            st.success(f"**{alert['feature']}**: {alert['message']}")
            except Exception:
                pass

        except FileNotFoundError:
            st.error("Models not trained yet. Please run the training notebooks first.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Heart Disease Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        h_age = st.slider("Age", 1, 120, 55)
        h_sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        h_cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
        h_trestbps = st.slider("Resting Blood Pressure (mmHg)", 0, 300, 130)
        h_chol = st.slider("Cholesterol (mg/dL)", 0, 600, 250)
        h_fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        h_restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                  format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

    with col2:
        h_thalach = st.slider("Max Heart Rate", 0, 250, 150)
        h_exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        h_oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        h_slope = st.selectbox("ST Slope", [0, 1, 2],
                                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        h_ca = st.selectbox("Major Vessels (fluoroscopy)", [0, 1, 2, 3, 4])
        h_thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                               format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Other"][x])

    if st.button("Predict Risk", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_heart_features
            from src.model import load_model, load_scaler
            from src.explainer import get_shap_explainer, generate_clinical_alerts
            from src.preprocessing import RISK_LABEL_MAP

            patient = pd.DataFrame([{
                "age": h_age, "sex": h_sex, "cp": h_cp, "trestbps": h_trestbps,
                "chol": h_chol, "fbs": h_fbs, "restecg": h_restecg,
                "thalach": h_thalach, "exang": h_exang, "oldpeak": h_oldpeak,
                "slope": h_slope, "ca": h_ca, "thal": h_thal,
            }])

            patient = add_heart_features(patient)
            feature_cols = [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                "BPxHR", "CholAge", "HeartStress",
            ]

            model = load_model(model_name, "heart")
            scaler = load_scaler("heart")

            X = patient[feature_cols]
            X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

            risk_class = int(model.predict(X_scaled)[0])
            probabilities = model.predict_proba(X_scaled)[0]
            risk_level = RISK_LABEL_MAP[risk_class]

            confidence = {
                "Low": float(probabilities[0]),
                "Medium": float(probabilities[1]),
                "High": float(probabilities[2]),
            }

            render_risk_card(risk_level, confidence)

            col_a, col_b = st.columns(2)
            with col_a:
                render_confidence_chart(confidence)

            with col_b:
                try:
                    explainer = get_shap_explainer(model)
                    shap_vals = explainer(X_scaled)

                    if len(shap_vals.shape) == 3:
                        sv = shap_vals.values[0, :, risk_class]
                    else:
                        sv = shap_vals.values[0]

                    shap_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "SHAP Value": sv,
                    }).sort_values("SHAP Value", key=abs, ascending=True)

                    fig = go.Figure(go.Bar(
                        x=shap_df["SHAP Value"],
                        y=shap_df["Feature"],
                        orientation="h",
                        marker_color=["#dc3545" if v > 0 else "#28a745" for v in shap_df["SHAP Value"]],
                    ))
                    fig.update_layout(title="SHAP Feature Impact", height=400, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

            try:
                if 'sv' in dir():
                    alerts = generate_clinical_alerts(
                        patient_data=patient.iloc[0].to_dict(),
                        shap_values_patient=sv,
                        feature_names=feature_cols,
                        dataset_type="heart",
                    )
                    st.subheader("Clinical Alerts")
                    for alert in alerts:
                        if alert["severity"] == "high":
                            st.error(f"**{alert['feature']}**: {alert['message']}")
                        else:
                            st.success(f"**{alert['feature']}**: {alert['message']}")
            except Exception:
                pass

        except FileNotFoundError:
            st.error("Models not trained yet. Please run the training notebooks first.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
