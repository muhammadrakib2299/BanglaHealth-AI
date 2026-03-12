"""About — System information and methodology."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

st.header("About BanglaHealth-AI")
st.caption("Methodology, technology stack, and references")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Motivation")
    st.markdown("""
    Bangladesh has only **0.58 physicians per 1,000 people** (WHO, 2022).
    Clinical data is collected but rarely analyzed predictively. This system
    transforms routine clinical measurements into actionable 3-tier risk scores
    using explainable AI.
    """)

    st.subheader("Processing Pipeline")
    st.markdown("""
| Step | Process |
|------|---------|
| 1. Data Collection | Pima Diabetes (768) + UCI Heart (303) |
| 2. Cleaning | Median imputation for zero values |
| 3. Feature Engineering | GlucoseBMI, AgeRisk, InsulinLog, BPCategory |
| 4. Normalization | StandardScaler (mean=0, std=1) |
| 5. Class Balancing | SMOTE on training set only |
| 6. Training | RandomizedSearchCV, 5-fold StratifiedKFold |
| 7. Evaluation | F1-Macro, Precision, Recall, ROC-AUC |
| 8. Explainability | SHAP TreeExplainer / LinearExplainer |
""")

    st.subheader("Risk Stratification")
    st.markdown("""
| Level | Description | Action |
|-------|-------------|--------|
| **Low** | Healthy indicators | Routine monitoring |
| **Medium** | Borderline values | Enhanced screening |
| **High** | Disease markers | Immediate intervention |
""")

with col2:
    st.subheader("ML Models")
    st.markdown("""
| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline, interpretable |
| Random Forest | Ensemble bagging |
| XGBoost | Gradient boosting |
| LightGBM | Leaf-wise gradient boosting |
""")

    st.subheader("Technology Stack")
    st.markdown("""
| Component | Technology |
|-----------|-----------|
| ML | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Reports | fpdf2 |
| Testing | pytest, ruff |
| DevOps | Docker, GitHub Actions |
""")

    st.subheader("References")
    st.markdown("""
1. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
2. WHO Bangladesh Health Workforce Data (2022).
3. Chawla et al. (2002). "SMOTE." *JAIR*.
4. Chen & Guestrin (2016). "XGBoost." *KDD*.
5. Ke et al. (2017). "LightGBM." *NeurIPS*.
""")

    st.subheader("Developer")
    st.markdown("""
| | |
|---|---|
| **Name** | Md. Rakib |
| **Position** | Junior Software Developer, Combosoft Ltd. |
| **Education** | BSc Computer Science, Daffodil International University |
""")
