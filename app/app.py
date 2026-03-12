"""BanglaHealth-AI — Dashboard Home."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="BanglaHealth-AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.header("BanglaHealth-AI")
st.caption("Clinical Risk Stratification System")

# ── Status ───────────────────────────────────────────────────────────────────
models_dir = PROJECT_ROOT / "models"
models_exist = models_dir.exists() and any(models_dir.glob("*.joblib"))
model_count = len(list(models_dir.glob("*.joblib"))) - 2 if models_exist else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Physicians/1K", "0.58", help="Bangladesh (WHO 2022)")
c2.metric("Total Patients", "1,071", help="768 Diabetes + 303 Heart")
c3.metric("ML Models", "4", help="LR, RF, XGBoost, LightGBM")
c4.metric("Risk Tiers", "3", help="Low / Medium / High")
c5.metric("Models Ready", model_count, help="Trained .joblib files")

st.divider()

# ── Model Performance ────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Model Performance")
    if models_exist:
        tab_d, tab_h = st.tabs(["Diabetes", "Heart Disease"])
        for tab, ds_key in [(tab_d, "diabetes"), (tab_h, "heart")]:
            with tab:
                comp_file = PROJECT_ROOT / "outputs" / f"{ds_key}_model_comparison.csv"
                if comp_file.exists():
                    df = pd.read_csv(comp_file)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    best = df.loc[df["F1-Macro"].idxmax()]
                    st.success(f"Best: **{best['Model']}** — F1: {best['F1-Macro']:.4f} | AUC: {best['ROC-AUC']:.4f}")
                else:
                    st.info("Run training notebooks to generate results.")
    else:
        st.warning("No trained models found. Run notebooks first.")

with col_right:
    st.subheader("Pipeline")
    st.markdown("""
| Step | Process |
|------|---------|
| 1 | Data Cleaning — median imputation |
| 2 | Feature Engineering — GlucoseBMI, AgeRisk |
| 3 | StandardScaler normalization |
| 4 | SMOTE class balancing (train only) |
| 5 | RandomizedSearchCV, 5-fold CV |
| 6 | SHAP explainability |
""")

    st.subheader("System Info")
    st.markdown("""
| Component | Technology |
|-----------|-----------|
| ML | scikit-learn, XGBoost, LightGBM |
| Explain | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Reports | fpdf2 |
| DevOps | Docker, GitHub Actions |
""")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**BanglaHealth-AI** v1.0")
    st.markdown("---")
    if models_exist:
        st.success(f"{model_count} models ready")
    else:
        st.warning("Models not trained")
    st.markdown("---")
    st.caption("Md. Rakib — Combosoft Ltd.")
