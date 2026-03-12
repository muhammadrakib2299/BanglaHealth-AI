"""BanglaHealth-AI — Desktop Dashboard."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="BanglaHealth-AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ERP / Desktop Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove website feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tighten main content padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }

    /* ── Sidebar: dark nav panel ── */
    [data-testid="stSidebar"] {
        background: #1E293B;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        border-radius: 4px;
        margin: 1px 8px;
        padding: 6px 12px;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {
        color: #94A3B8 !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] {
        background: #2563EB !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] span {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #334155; }

    /* ── Toolbar header bar ── */
    .toolbar {
        background: #1E293B;
        color: #F8FAFC;
        padding: 10px 20px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
        font-size: 13px;
        border: 1px solid #334155;
    }
    .toolbar .app-title {
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.3px;
    }
    .toolbar .status {
        display: flex;
        gap: 16px;
        align-items: center;
    }
    .toolbar .status-item {
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 12px;
        color: #94A3B8;
    }
    .toolbar .dot {
        width: 7px; height: 7px; border-radius: 50%; display: inline-block;
    }
    .dot-green { background: #22C55E; }
    .dot-yellow { background: #EAB308; }
    .dot-red { background: #EF4444; }

    /* ── Panel / Section containers ── */
    .panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .panel-header {
        background: #F8FAFC;
        border-bottom: 1px solid #E2E8F0;
        padding: 8px 14px;
        font-size: 12px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .panel-body {
        padding: 12px 14px;
    }

    /* ── KPI row ── */
    .kpi {
        text-align: center;
        padding: 10px 8px;
    }
    .kpi .kpi-value {
        font-size: 26px;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi .kpi-label {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    .kpi .kpi-sub {
        font-size: 11px;
        color: #94A3B8;
    }

    /* ── Data grid row ── */
    .grid-row {
        display: flex;
        border-bottom: 1px solid #F1F5F9;
        padding: 6px 14px;
        font-size: 13px;
    }
    .grid-row:hover { background: #F8FAFC; }
    .grid-row .col-label {
        width: 180px;
        color: #64748B;
        font-weight: 500;
        flex-shrink: 0;
    }
    .grid-row .col-value {
        color: #1E293B;
        font-weight: 600;
    }

    /* ── Module list items ── */
    .module-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 14px;
        border-bottom: 1px solid #F1F5F9;
        font-size: 13px;
        color: #334155;
    }
    .module-item:last-child { border-bottom: none; }
    .module-item .mod-icon {
        width: 28px; height: 28px; border-radius: 4px;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; background: #EFF6FF; flex-shrink: 0;
    }
    .module-item .mod-name { font-weight: 600; }
    .module-item .mod-desc { color: #64748B; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
models_dir = PROJECT_ROOT / "models"
models_exist = models_dir.exists() and any(models_dir.glob("*.joblib"))
model_count = len(list(models_dir.glob("*.joblib"))) - 2 if models_exist else 0

# ── Toolbar ──────────────────────────────────────────────────────────────────
dot = "dot-green" if models_exist else "dot-red"
status_text = f"{model_count} models ready" if models_exist else "No models"

st.markdown(f"""
<div class="toolbar">
    <div class="app-title">BanglaHealth-AI &mdash; Clinical Risk Stratification System</div>
    <div class="status">
        <div class="status-item"><span class="dot {dot}"></span> {status_text}</div>
        <div class="status-item"><span class="dot dot-green"></span> 2 datasets loaded</div>
        <div class="status-item"><span class="dot dot-green"></span> System online</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown("""
    <div class="panel"><div class="kpi">
        <div class="kpi-value" style="color:#DC2626;">0.58</div>
        <div class="kpi-label">Physicians / 1K</div>
        <div class="kpi-sub">Bangladesh WHO 2022</div>
    </div></div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="panel"><div class="kpi">
        <div class="kpi-value" style="color:#2563EB;">1,071</div>
        <div class="kpi-label">Total Patients</div>
        <div class="kpi-sub">768 + 303 samples</div>
    </div></div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="panel"><div class="kpi">
        <div class="kpi-value" style="color:#2563EB;">4</div>
        <div class="kpi-label">ML Models</div>
        <div class="kpi-sub">LR / RF / XGB / LGBM</div>
    </div></div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="panel"><div class="kpi">
        <div class="kpi-value" style="color:#2563EB;">3</div>
        <div class="kpi-label">Risk Tiers</div>
        <div class="kpi-sub">Low / Medium / High</div>
    </div></div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown("""
    <div class="panel"><div class="kpi">
        <div class="kpi-value" style="color:#059669;">SHAP</div>
        <div class="kpi-label">Explainability</div>
        <div class="kpi-sub">Per-prediction</div>
    </div></div>
    """, unsafe_allow_html=True)

# ── Two-column layout ───────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    # Model Performance Panel
    st.markdown('<div class="panel"><div class="panel-header">Model Performance Summary</div><div class="panel-body">', unsafe_allow_html=True)

    if models_exist:
        tab_d, tab_h = st.tabs(["Diabetes", "Heart Disease"])
        for tab, ds_key, ds_label in [(tab_d, "diabetes", "Diabetes"), (tab_h, "heart", "Heart Disease")]:
            with tab:
                comp_file = PROJECT_ROOT / "outputs" / f"{ds_key}_model_comparison.csv"
                if comp_file.exists():
                    df = pd.read_csv(comp_file)
                    st.dataframe(
                        df.style.highlight_max(
                            subset=["F1-Macro", "ROC-AUC", "Precision (macro)", "Recall (macro)"],
                            color="#DBEAFE"
                        ),
                        use_container_width=True, hide_index=True, height=180,
                    )
                    best = df.loc[df["F1-Macro"].idxmax()]
                    st.caption(f"Best: **{best['Model']}** — F1-Macro: {best['F1-Macro']:.4f} | ROC-AUC: {best['ROC-AUC']:.4f}")
                else:
                    st.caption("Run training notebooks to generate results.")
    else:
        st.caption("No trained models. Run notebooks/03_model_training.ipynb first.")

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Pipeline Panel
    st.markdown("""
    <div class="panel">
        <div class="panel-header">Processing Pipeline</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label">1. Data Cleaning</span><span class="col-value">Median imputation for zero values</span></div>
            <div class="grid-row"><span class="col-label">2. Feature Engineering</span><span class="col-value">GlucoseBMI, AgeRisk, InsulinLog, BPCategory</span></div>
            <div class="grid-row"><span class="col-label">3. Normalization</span><span class="col-value">StandardScaler (mean=0, std=1)</span></div>
            <div class="grid-row"><span class="col-label">4. Class Balancing</span><span class="col-value">SMOTE on training set only</span></div>
            <div class="grid-row"><span class="col-label">5. Training</span><span class="col-value">RandomizedSearchCV, 5-fold StratifiedKFold</span></div>
            <div class="grid-row"><span class="col-label">6. Explainability</span><span class="col-value">SHAP TreeExplainer / LinearExplainer</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    # Modules Panel
    st.markdown("""
    <div class="panel">
        <div class="panel-header">System Modules</div>
        <div class="panel-body" style="padding: 0;">
            <div class="module-item">
                <div class="mod-icon">🩺</div>
                <div><div class="mod-name">Patient Prediction</div><div class="mod-desc">Single patient risk assessment + PDF report</div></div>
            </div>
            <div class="module-item">
                <div class="mod-icon">📊</div>
                <div><div class="mod-name">Batch Prediction</div><div class="mod-desc">CSV upload, bulk risk scoring</div></div>
            </div>
            <div class="module-item">
                <div class="mod-icon">📈</div>
                <div><div class="mod-name">Model Comparison</div><div class="mod-desc">Side-by-side performance metrics</div></div>
            </div>
            <div class="module-item">
                <div class="mod-icon">🔍</div>
                <div><div class="mod-name">EDA Dashboard</div><div class="mod-desc">Interactive dataset exploration</div></div>
            </div>
            <div class="module-item">
                <div class="mod-icon">🔬</div>
                <div><div class="mod-name">What-If Analysis</div><div class="mod-desc">Real-time parameter sensitivity</div></div>
            </div>
            <div class="module-item">
                <div class="mod-icon">ℹ️</div>
                <div><div class="mod-name">About</div><div class="mod-desc">Methodology and references</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System Info Panel
    st.markdown("""
    <div class="panel">
        <div class="panel-header">System Information</div>
        <div class="panel-body">
            <div class="grid-row"><span class="col-label">Application</span><span class="col-value">BanglaHealth-AI v1.0</span></div>
            <div class="grid-row"><span class="col-label">ML Framework</span><span class="col-value">scikit-learn, XGBoost, LightGBM</span></div>
            <div class="grid-row"><span class="col-label">Explainability</span><span class="col-value">SHAP v0.51</span></div>
            <div class="grid-row"><span class="col-label">API</span><span class="col-value">FastAPI + Uvicorn</span></div>
            <div class="grid-row"><span class="col-label">Dashboard</span><span class="col-value">Streamlit + Plotly</span></div>
            <div class="grid-row"><span class="col-label">Reports</span><span class="col-value">fpdf2 PDF generation</span></div>
            <div class="grid-row"><span class="col-label">Developer</span><span class="col-value">Md. Rakib</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 4px 4px 4px; border-bottom: 1px solid #334155; margin-bottom: 8px;">
        <div style="font-size: 14px; font-weight: 700; color: #F8FAFC; letter-spacing: 0.3px;">
            🏥 BanglaHealth-AI
        </div>
        <div style="font-size: 11px; color: #64748B; margin-top: 2px;">
            Clinical Risk Stratification
        </div>
    </div>
    """, unsafe_allow_html=True)
