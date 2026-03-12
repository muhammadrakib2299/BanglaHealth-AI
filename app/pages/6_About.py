"""About — System information and methodology."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

from app.styles import ERP_CSS
st.markdown(ERP_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="toolbar">
    <div class="tb-title">ℹ️ System Information</div>
    <div class="tb-right">Methodology, stack, and references</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">Project Motivation</div>
        <div class="panel-body" style="font-size: 13px; color: #475569; line-height: 1.6;">
            Bangladesh has only <strong>0.58 physicians per 1,000 people</strong> (WHO, 2022).
            Clinical data is collected but rarely analyzed predictively. This system transforms
            routine clinical measurements into actionable 3-tier risk scores using explainable AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-header">Processing Pipeline</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label">1. Data Collection</span><span class="col-value">Pima Diabetes (768) + UCI Heart (303)</span></div>
            <div class="grid-row"><span class="col-label">2. Cleaning</span><span class="col-value">Median imputation for zero values</span></div>
            <div class="grid-row"><span class="col-label">3. Feature Engineering</span><span class="col-value">GlucoseBMI, AgeRisk, InsulinLog, BPCategory</span></div>
            <div class="grid-row"><span class="col-label">4. Normalization</span><span class="col-value">StandardScaler (mean=0, std=1)</span></div>
            <div class="grid-row"><span class="col-label">5. Class Balancing</span><span class="col-value">SMOTE on training set only</span></div>
            <div class="grid-row"><span class="col-label">6. Training</span><span class="col-value">RandomizedSearchCV, 5-fold StratifiedKFold</span></div>
            <div class="grid-row"><span class="col-label">7. Evaluation</span><span class="col-value">F1-Macro, Precision, Recall, ROC-AUC</span></div>
            <div class="grid-row"><span class="col-label">8. Explainability</span><span class="col-value">SHAP TreeExplainer / LinearExplainer</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-header">Risk Stratification</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label" style="color:#059669; font-weight:700;">LOW</span><span class="col-value">Healthy indicators &rarr; routine monitoring</span></div>
            <div class="grid-row"><span class="col-label" style="color:#D97706; font-weight:700;">MEDIUM</span><span class="col-value">Borderline values &rarr; enhanced screening</span></div>
            <div class="grid-row"><span class="col-label" style="color:#DC2626; font-weight:700;">HIGH</span><span class="col-value">Disease markers &rarr; immediate intervention</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">ML Models</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label">Logistic Regression</span><span class="col-value">Linear baseline, interpretable coefficients</span></div>
            <div class="grid-row"><span class="col-label">Random Forest</span><span class="col-value">Ensemble bagging, feature importance</span></div>
            <div class="grid-row"><span class="col-label">XGBoost</span><span class="col-value">Gradient boosting, best tabular performance</span></div>
            <div class="grid-row"><span class="col-label">LightGBM</span><span class="col-value">Leaf-wise gradient boosting, fast training</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-header">Technology Stack</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label">ML Framework</span><span class="col-value">scikit-learn, XGBoost, LightGBM</span></div>
            <div class="grid-row"><span class="col-label">Explainability</span><span class="col-value">SHAP (SHapley Additive exPlanations)</span></div>
            <div class="grid-row"><span class="col-label">API Server</span><span class="col-value">FastAPI + Uvicorn</span></div>
            <div class="grid-row"><span class="col-label">Dashboard</span><span class="col-value">Streamlit + Plotly</span></div>
            <div class="grid-row"><span class="col-label">Reports</span><span class="col-value">fpdf2 PDF generation</span></div>
            <div class="grid-row"><span class="col-label">Testing</span><span class="col-value">pytest, ruff</span></div>
            <div class="grid-row"><span class="col-label">DevOps</span><span class="col-value">Docker, GitHub Actions CI</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-header">References</div>
        <div class="panel-body" style="font-size: 12px; color: #475569; line-height: 1.6;">
            1. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.<br>
            2. WHO Bangladesh Health Workforce Data (2022).<br>
            3. Chawla et al. (2002). "SMOTE." JAIR.<br>
            4. Chen & Guestrin (2016). "XGBoost." KDD.<br>
            5. Ke et al. (2017). "LightGBM." NeurIPS.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-header">Developer</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label">Name</span><span class="col-value">Md. Rakib</span></div>
            <div class="grid-row"><span class="col-label">Position</span><span class="col-value">Junior Software Developer, Combosoft Ltd.</span></div>
            <div class="grid-row"><span class="col-label">Education</span><span class="col-value">BSc Computer Science, Daffodil International University</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
