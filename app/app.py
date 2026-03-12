"""BanglaHealth-AI — Streamlit Dashboard."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="BanglaHealth-AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0D2137 100%);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] span {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #CBD5E1;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #94A3B8 !important;
    }

    /* Card containers */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0 0 8px 0;
        font-size: 14px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #0077B6 0%, #023E8A 50%, #03045E 100%);
        border-radius: 16px;
        padding: 48px 40px;
        color: white;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero h1 {
        font-size: 40px;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: white;
    }
    .hero .tagline {
        font-size: 18px;
        color: #90CAF9;
        margin-bottom: 16px;
        font-weight: 400;
    }
    .hero .stat {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 8px 16px;
        margin-right: 12px;
        margin-top: 8px;
        font-size: 14px;
        color: #E3F2FD;
        backdrop-filter: blur(4px);
    }

    /* Feature grid */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        height: 100%;
        transition: all 0.2s ease;
    }
    .feature-card:hover {
        border-color: #0077B6;
        box-shadow: 0 4px 12px rgba(0,119,182,0.1);
    }
    .feature-card .icon {
        font-size: 28px;
        margin-bottom: 12px;
    }
    .feature-card h4 {
        margin: 0 0 8px 0;
        color: #1B2A4A;
        font-size: 16px;
    }
    .feature-card p {
        margin: 0;
        color: #64748B;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Section headers */
    .section-header {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #0077B6;
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-ready { background: #D1FAE5; color: #065F46; }
    .status-pending { background: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>BanglaHealth-AI</h1>
    <div class="tagline">Explainable AI for Patient Risk Stratification in Low-Resource Clinical Settings</div>
    <div>
        <span class="stat">4 ML Models</span>
        <span class="stat">SHAP Explanations</span>
        <span class="stat">3-Tier Risk Scoring</span>
        <span class="stat">Real-Time Predictions</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Key Metrics ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Project Overview</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Physicians / 1K People</h3>
        <p class="value" style="color: #DC2626;">0.58</p>
        <p class="subtitle">Bangladesh (WHO, 2022)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Datasets</h3>
        <p class="value" style="color: #0077B6;">2</p>
        <p class="subtitle">Diabetes + Heart Disease</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Total Patients</h3>
        <p class="value" style="color: #0077B6;">1,071</p>
        <p class="subtitle">768 Diabetes + 303 Heart</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Risk Levels</h3>
        <p class="value" style="color: #059669;">3</p>
        <p class="subtitle">Low / Medium / High</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Model Status ─────────────────────────────────────────────────────────────
models_dir = PROJECT_ROOT / "models"
models_exist = models_dir.exists() and any(models_dir.glob("*.joblib"))

if models_exist:
    # Load comparison results if available
    import pandas as pd
    st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
    col_d, col_h = st.columns(2)

    with col_d:
        comp_file = PROJECT_ROOT / "outputs" / "diabetes_model_comparison.csv"
        if comp_file.exists():
            df = pd.read_csv(comp_file)
            best = df.loc[df["F1-Macro"].idxmax()]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Diabetes — Best Model</h3>
                <p class="value" style="color: #059669;">{best['F1-Macro']:.2%}</p>
                <p class="subtitle">{best['Model']} (F1-Macro)</p>
            </div>
            """, unsafe_allow_html=True)

    with col_h:
        comp_file = PROJECT_ROOT / "outputs" / "heart_model_comparison.csv"
        if comp_file.exists():
            df = pd.read_csv(comp_file)
            best = df.loc[df["F1-Macro"].idxmax()]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Heart Disease — Best Model</h3>
                <p class="value" style="color: #059669;">{best['F1-Macro']:.2%}</p>
                <p class="subtitle">{best['Model']} (F1-Macro)</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Features Grid ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Features</p>', unsafe_allow_html=True)

features = [
    ("🩺", "Patient Prediction", "Enter patient data and get instant risk assessment with SHAP explanations and PDF reports."),
    ("📊", "Batch Prediction", "Upload a CSV file to predict risk levels for hundreds of patients at once."),
    ("📈", "Model Comparison", "Compare performance metrics across all 4 trained models side by side."),
    ("🔍", "EDA Dashboard", "Explore the training datasets with interactive charts, distributions, and correlations."),
    ("🔬", "What-If Analysis", "Adjust patient parameters with sliders and watch risk predictions change in real-time."),
    ("📄", "PDF Reports", "Download professional patient reports with risk level, SHAP values, and clinical alerts."),
]

cols = st.columns(3)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="icon">{icon}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ── How It Works ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">How It Works</p>', unsafe_allow_html=True)

steps = [
    ("1", "Data Input", "Patient clinical measurements are entered or uploaded"),
    ("2", "Preprocessing", "Feature engineering + StandardScaler normalization"),
    ("3", "ML Prediction", "Trained model predicts Low / Medium / High risk"),
    ("4", "SHAP Explain", "Every prediction is explained with feature-level impact"),
    ("5", "Clinical Alert", "Human-readable insights for healthcare workers"),
]

cols = st.columns(5)
for i, (num, title, desc) in enumerate(steps):
    with cols[i]:
        st.markdown(f"""
        <div style="text-align: center; padding: 16px;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: #0077B6;
                        color: white; display: inline-flex; align-items: center; justify-content: center;
                        font-weight: 700; font-size: 18px; margin-bottom: 8px;">{num}</div>
            <h4 style="margin: 4px 0; font-size: 14px; color: #1B2A4A;">{title}</h4>
            <p style="font-size: 12px; color: #64748B; margin: 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0 8px 0;">
        <span style="font-size: 32px;">🏥</span>
        <h2 style="margin: 4px 0; font-size: 18px; color: white !important;">BanglaHealth-AI</h2>
        <p style="font-size: 12px; color: #64748B !important; margin: 0;">v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if models_exist:
        model_count = len(list(models_dir.glob("*.joblib"))) - 2  # exclude scalers
        st.success(f"{model_count} models trained and ready")
    else:
        st.warning("Models not trained yet")

    st.markdown("---")
    st.caption("Developed by Md. Rakib")
    st.caption("BSc CS, Daffodil International University")
