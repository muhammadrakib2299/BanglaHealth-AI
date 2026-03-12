"""What-If Analysis Page — Adjust patient values with sliders, see risk change in real-time."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="What-If Analysis", page_icon="🔬", layout="wide")
st.title("What-If Analysis")
st.markdown(
    "Adjust patient values using the sliders and watch how the **risk prediction changes in real-time**. "
    "This helps understand which factors most influence the prediction."
)

dataset = st.sidebar.selectbox("Disease Type", ["Diabetes", "Heart Disease"])
model_name = st.sidebar.selectbox("Model", ["xgboost", "lightgbm", "random_forest", "logistic_regression"])

try:
    from src.feature_engineering import add_diabetes_features, add_heart_features
    from src.model import load_model, load_scaler
    from src.preprocessing import RISK_LABEL_MAP

    RISK_COLORS = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}

    if dataset == "Diabetes":
        model = load_model(model_name, "diabetes")
        scaler = load_scaler("diabetes")

        st.subheader("Adjust Patient Parameters")

        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.slider("Pregnancies", 0, 17, 3, key="wif_preg")
            glucose = st.slider("Glucose (mg/dL)", 0, 300, 120, key="wif_gluc")
            blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 200, 72, key="wif_bp")
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 29, key="wif_skin")

        with col2:
            insulin = st.slider("Insulin (mu U/ml)", 0, 900, 80, key="wif_ins")
            bmi = st.slider("BMI", 0.0, 70.0, 28.0, 0.5, key="wif_bmi")
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01, key="wif_dpf")
            age = st.slider("Age", 1, 120, 35, key="wif_age")

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

    else:
        model = load_model(model_name, "heart")
        scaler = load_scaler("heart")

        st.subheader("Adjust Patient Parameters")

        col1, col2 = st.columns(2)
        with col1:
            h_age = st.slider("Age", 1, 120, 55, key="wif_h_age")
            h_sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="wif_h_sex")
            h_cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                                format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x], key="wif_h_cp")
            h_trestbps = st.slider("Resting BP (mmHg)", 0, 300, 130, key="wif_h_bp")
            h_chol = st.slider("Cholesterol (mg/dL)", 0, 600, 250, key="wif_h_chol")
            h_fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="wif_h_fbs")
            h_restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                      format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x], key="wif_h_ecg")

        with col2:
            h_thalach = st.slider("Max Heart Rate", 0, 250, 150, key="wif_h_hr")
            h_exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="wif_h_exang")
            h_oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1, key="wif_h_old")
            h_slope = st.selectbox("ST Slope", [0, 1, 2],
                                    format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], key="wif_h_slope")
            h_ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4], key="wif_h_ca")
            h_thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                                   format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Other"][x], key="wif_h_thal")

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

    # Predict
    X = patient[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    risk_class = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]
    risk_level = RISK_LABEL_MAP[risk_class]

    # Display results
    st.markdown("---")

    col_result, col_chart = st.columns([1, 2])

    with col_result:
        color = RISK_COLORS[risk_level]
        st.markdown(f"""
        <div style="background-color: {color}; padding: 30px; border-radius: 15px;
                    text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 48px; color: white;">{risk_level}</h1>
            <p style="margin: 5px 0; font-size: 20px; color: white;">Risk Level</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Confidence metrics
        st.metric("Low Risk", f"{probabilities[0]:.1%}")
        st.metric("Medium Risk", f"{probabilities[1]:.1%}")
        st.metric("High Risk", f"{probabilities[2]:.1%}")

    with col_chart:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilities[2] * 100,
            title={"text": "High Risk Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 33], "color": "#28a745"},
                    {"range": [33, 66], "color": "#ffc107"},
                    {"range": [66, 100], "color": "#dc3545"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": probabilities[2] * 100,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Probability bar chart
        fig2 = go.Figure(go.Bar(
            x=["Low", "Medium", "High"],
            y=[probabilities[0], probabilities[1], probabilities[2]],
            marker_color=["#28a745", "#ffc107", "#dc3545"],
            text=[f"{p:.1%}" for p in probabilities],
            textposition="auto",
        ))
        fig2.update_layout(
            title="Class Probabilities",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # SHAP explanation
    st.markdown("---")
    st.subheader("Feature Impact (SHAP)")

    try:
        from src.explainer import get_shap_explainer

        explainer = get_shap_explainer(model)
        shap_vals = explainer(X_scaled)

        if len(shap_vals.shape) == 3:
            sv = shap_vals.values[0, :, risk_class]
        else:
            sv = shap_vals.values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Value": sv,
            "Abs SHAP": np.abs(sv),
        }).sort_values("Abs SHAP", ascending=True)

        fig3 = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=["#dc3545" if v > 0 else "#28a745" for v in shap_df["SHAP Value"]],
            text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]],
            textposition="auto",
        ))
        fig3.update_layout(
            title=f"SHAP Values — What's Driving the {risk_level} Risk Prediction?",
            xaxis_title="SHAP Value (positive = increases risk, negative = decreases risk)",
            height=max(400, len(feature_cols) * 30),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "**How to read:** Red bars push the prediction toward higher risk. "
            "Green bars push toward lower risk. Try adjusting sliders above to see "
            "how the bars change."
        )
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

except FileNotFoundError:
    st.error(
        "Models not trained yet. Please run the training notebooks first:\n\n"
        "1. `notebooks/02_preprocessing.ipynb`\n"
        "2. `notebooks/03_model_training.ipynb`"
    )
except Exception as e:
    st.error(f"Error: {e}")
