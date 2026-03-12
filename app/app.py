"""BanglaHealth-AI — Streamlit Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="BanglaHealth-AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("BanglaHealth-AI")
st.subheader("Explainable AI for Patient Risk Stratification")

st.markdown("""
Bangladesh has only **0.58 physicians per 1,000 people** (WHO, 2022).
This tool predicts patient risk levels and explains every prediction
using SHAP — empowering health workers to make informed clinical decisions.

---

### Navigate

Use the **sidebar** to access different pages:

| Page | Description |
|------|-------------|
| **Patient Prediction** | Enter patient data and get a risk prediction with explanations |
| **Batch Prediction** | Upload a CSV file to predict risk for multiple patients |
| **Model Comparison** | Compare performance metrics across all trained models |
| **EDA Dashboard** | Explore the training datasets interactively |
| **About** | Learn about the methodology and team |

---

### Quick Start

1. **Train models first** by running the Jupyter notebooks in `notebooks/`
2. Then use this dashboard to make predictions and explore results

### Models Used

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble (bagging) |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting (leaf-wise) |
""")

# Show model status in sidebar
st.sidebar.markdown("---")
st.sidebar.caption("BanglaHealth-AI v1.0.0")
