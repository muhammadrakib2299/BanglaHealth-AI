"""About — Project methodology, team, and references."""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .page-header {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%);
        border-radius: 12px; padding: 28px 32px; color: white; margin-bottom: 24px;
    }
    .page-header h2 { margin: 0; color: white; font-size: 28px; }
    .page-header p { margin: 4px 0 0 0; color: #90CAF9; font-size: 15px; }
    .info-card {
        background: white; border-radius: 12px; padding: 24px; border: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .info-card h4 { margin: 0 0 12px 0; color: #0077B6; font-size: 16px; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }
    .info-card p, .info-card li { color: #475569; font-size: 14px; line-height: 1.7; }
    .author-card {
        background: linear-gradient(135deg, #0A1628 0%, #0D2137 100%);
        border-radius: 12px; padding: 28px; color: white; text-align: center;
    }
    .author-card h3 { color: white; margin: 0 0 4px 0; font-size: 22px; }
    .author-card .role { color: #90CAF9; font-size: 14px; }
    .author-card .edu { color: #64748B; font-size: 13px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h2>ℹ️ About BanglaHealth-AI</h2>
    <p>Project methodology, technology stack, and research references</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>Motivation</h4>
        <p>Bangladesh faces a critical healthcare gap: with only <strong>0.58 physicians per 1,000 people</strong>
        (WHO, 2022), clinical data collected in hospitals and rural clinics is rarely analyzed predictively.
        BanglaHealth-AI addresses this by building an explainable AI system that transforms routine
        clinical measurements into actionable risk scores.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Data Pipeline</h4>
        <p>
        <strong>1. Data Collection</strong> — Pima Diabetes (768 samples) + UCI Heart Disease (303 samples)<br>
        <strong>2. Cleaning</strong> — Replace biologically impossible zeros with median imputation<br>
        <strong>3. Feature Engineering</strong> — GlucoseBMI, AgeRisk, InsulinLog, BPCategory, BPxHR, CholAge<br>
        <strong>4. Preprocessing</strong> — StandardScaler normalization, stratified 80/20 split<br>
        <strong>5. Class Balancing</strong> — SMOTE applied to training set only<br>
        <strong>6. Training</strong> — RandomizedSearchCV, 5-fold StratifiedKFold, F1-Macro optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>3-Class Risk Stratification</h4>
        <p>Instead of binary prediction, we use clinically meaningful risk tiers:</p>
        <ul>
            <li><strong style="color: #059669;">Low</strong> — Healthy indicators, routine monitoring</li>
            <li><strong style="color: #D97706;">Medium</strong> — Borderline values, enhanced screening</li>
            <li><strong style="color: #DC2626;">High</strong> — Disease markers present, immediate intervention</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>Models</h4>
        <p>
        <strong>Logistic Regression</strong> — Linear baseline with interpretable coefficients<br>
        <strong>Random Forest</strong> — Ensemble bagging with feature importance<br>
        <strong>XGBoost</strong> — Gradient boosting, often best for tabular data<br>
        <strong>LightGBM</strong> — Fast leaf-wise gradient boosting
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Explainability</h4>
        <p><strong>SHAP (SHapley Additive exPlanations)</strong> — Game-theory based method that explains
        individual predictions by computing each feature's contribution to the outcome.
        Clinical alerts translate SHAP values into human-readable medical language.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Tech Stack</h4>
        <p>
        <strong>ML/Data:</strong> Python, scikit-learn, XGBoost, LightGBM, SHAP, pandas<br>
        <strong>API:</strong> FastAPI, Pydantic, Uvicorn<br>
        <strong>Dashboard:</strong> Streamlit, Plotly<br>
        <strong>Reports:</strong> fpdf2 (PDF generation)<br>
        <strong>Testing:</strong> pytest, ruff<br>
        <strong>DevOps:</strong> Docker, GitHub Actions CI
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>References</h4>
        <p>
        1. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." <em>NeurIPS</em>.<br>
        2. WHO Bangladesh Health Workforce Data (2022).<br>
        3. Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." <em>JAIR</em>.<br>
        4. Chen & Guestrin (2016). "XGBoost." <em>KDD</em>.<br>
        5. Ke et al. (2017). "LightGBM." <em>NeurIPS</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Author card
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="author-card">
    <h3>Md. Rakib</h3>
    <div class="role">Junior Software Developer, Combosoft Ltd.</div>
    <div class="edu">BSc Computer Science, Daffodil International University</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("BanglaHealth-AI is a research portfolio project demonstrating explainable ML for healthcare in low-resource settings.")
