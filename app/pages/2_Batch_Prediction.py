"""Batch Prediction — Upload CSV and get predictions for all patients."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Batch Prediction", page_icon="📊", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .page-header {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        border-radius: 12px; padding: 28px 32px; color: white; margin-bottom: 24px;
    }
    .page-header h2 { margin: 0; color: white; font-size: 28px; }
    .page-header p { margin: 4px 0 0 0; color: #90CAF9; font-size: 15px; }
    .summary-card {
        background: white; border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .summary-card h3 { margin: 0; font-size: 13px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .summary-card .val { font-size: 36px; font-weight: 700; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>📊 Batch Prediction</h2>
    <p>Upload a CSV file with patient data to predict risk levels for all patients at once</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    dataset = st.selectbox("Disease Type", ["Diabetes", "Heart Disease"])
    model_name = st.selectbox("ML Model", ["xgboost", "random_forest", "lightgbm", "logistic_regression"])

uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown(f"**{len(df)} patients loaded** — Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Run Predictions", type="primary", use_container_width=True):
        try:
            from src.feature_engineering import add_diabetes_features, add_heart_features
            from src.model import load_model, load_scaler
            from src.preprocessing import RISK_LABEL_MAP

            ds_key = "heart" if dataset == "Heart Disease" else "diabetes"
            model = load_model(model_name, ds_key)
            scaler = load_scaler(ds_key)

            if dataset == "Diabetes":
                df_feat = add_diabetes_features(df.copy())
                feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                                "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory"]
            else:
                df_feat = add_heart_features(df.copy())
                feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                                "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                                "BPxHR", "CholAge", "HeartStress"]

            X_scaled = pd.DataFrame(scaler.transform(df_feat[feature_cols]), columns=feature_cols)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)

            results = df.copy()
            results["Risk_Level"] = [RISK_LABEL_MAP[p] for p in predictions]
            results["Confidence_Low"] = probabilities[:, 0].round(4)
            results["Confidence_Medium"] = probabilities[:, 1].round(4)
            results["Confidence_High"] = probabilities[:, 2].round(4)

            low_n = int((predictions == 0).sum())
            med_n = int((predictions == 1).sum())
            high_n = int((predictions == 2).sum())

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="summary-card"><h3>Total</h3><div class="val" style="color:#0077B6;">{len(df)}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="summary-card"><h3>Low Risk</h3><div class="val" style="color:#059669;">{low_n}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="summary-card"><h3>Medium Risk</h3><div class="val" style="color:#D97706;">{med_n}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="summary-card"><h3>High Risk</h3><div class="val" style="color:#DC2626;">{high_n}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig = go.Figure(go.Pie(
                labels=["Low", "Medium", "High"], values=[low_n, med_n, high_n],
                marker_colors=["#059669", "#D97706", "#DC2626"],
                hole=0.4, textinfo="label+percent+value",
            ))
            fig.update_layout(title="Risk Distribution", height=350,
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detailed Results")
            st.dataframe(results, use_container_width=True)

            st.download_button("📥 Download Results CSV", results.to_csv(index=False),
                               f"banglahealth_batch_{ds_key}.csv", "text/csv",
                               use_container_width=True)

        except FileNotFoundError:
            st.error("Models not trained yet. Please run the training notebooks first.")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
else:
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Diabetes CSV Format")
        st.code("Pregnancies,Glucose,BloodPressure,SkinThickness,\nInsulin,BMI,DiabetesPedigreeFunction,Age", language="text")
    with col_r:
        st.markdown("#### Heart Disease CSV Format")
        st.code("age,sex,cp,trestbps,chol,fbs,restecg,\nthalach,exang,oldpeak,slope,ca,thal", language="text")
