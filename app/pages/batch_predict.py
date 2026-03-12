"""Batch Prediction Page — Upload CSV and get predictions for all patients."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Batch Prediction", page_icon="📊", layout="wide")
st.title("Batch Prediction")
st.markdown("Upload a CSV file with patient data to predict risk levels for all patients.")

dataset = st.sidebar.selectbox("Disease Type", ["Diabetes", "Heart Disease"])
model_name = st.sidebar.selectbox("Model", ["xgboost", "lightgbm", "random_forest", "logistic_regression"])

uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{len(df)} patients loaded")

    if st.button("Run Predictions", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_diabetes_features, add_heart_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            model = load_model(model_name, dataset.lower().replace(" ", "_").replace("disease", "").strip("_") if dataset == "Heart Disease" else "diabetes")
            scaler = load_scaler("heart" if dataset == "Heart Disease" else "diabetes")

            if dataset == "Diabetes":
                df_feat = add_diabetes_features(df.copy())
                feature_cols = [
                    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                    "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory",
                ]
            else:
                df_feat = add_heart_features(df.copy())
                feature_cols = [
                    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                    "BPxHR", "CholAge", "HeartStress",
                ]

            X = df_feat[feature_cols]
            X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)

            results = df.copy()
            results["Risk_Level"] = [RISK_LABEL_MAP[p] for p in predictions]
            results["Risk_Class"] = predictions
            results["Confidence_Low"] = probabilities[:, 0].round(4)
            results["Confidence_Medium"] = probabilities[:, 1].round(4)
            results["Confidence_High"] = probabilities[:, 2].round(4)

            st.subheader("Prediction Results")

            # Summary
            col1, col2, col3 = st.columns(3)
            low_count = (predictions == 0).sum()
            med_count = (predictions == 1).sum()
            high_count = (predictions == 2).sum()

            col1.metric("Low Risk", low_count, delta=None)
            col2.metric("Medium Risk", med_count, delta=None)
            col3.metric("High Risk", high_count, delta=None)

            # Color-coded results table
            st.dataframe(results, use_container_width=True)

            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                f"banglahealth_predictions_{dataset.lower()}.csv",
                "text/csv",
                use_container_width=True,
            )

        except FileNotFoundError:
            st.error("Models not trained yet. Please run the training notebooks first.")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
else:
    st.info("Please upload a CSV file with patient data to begin.")

    if dataset == "Diabetes":
        st.markdown("""
        **Expected columns:** Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
        """)
    else:
        st.markdown("""
        **Expected columns:** age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
        """)
