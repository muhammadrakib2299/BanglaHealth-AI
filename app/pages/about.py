"""About Page — Project methodology, team, and references."""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("About BanglaHealth-AI")

st.markdown("""
## Motivation

Bangladesh faces a critical healthcare gap: with only **0.58 physicians per 1,000 people**
(WHO, 2022), clinical data collected in hospitals and rural clinics is rarely analyzed
predictively. BanglaHealth-AI addresses this by building an explainable AI system that
transforms routine clinical measurements into actionable risk scores.

## Methodology

### Data Pipeline
1. **Data Collection**: Two public clinical datasets (Pima Diabetes + UCI Heart Disease)
2. **Cleaning**: Replace biologically impossible zero values with median imputation
3. **Feature Engineering**: Create clinically meaningful derived features
   - GlucoseBMI (metabolic syndrome interaction)
   - AgeRisk (age-adjusted risk factor)
   - BPCategory (AHA blood pressure guidelines)
4. **Preprocessing**: StandardScaler normalization, stratified 80/20 split
5. **Class Balancing**: SMOTE applied to training set only

### Models
| Model | Type | Key Property |
|-------|------|-------------|
| Logistic Regression | Linear | Interpretable coefficients |
| Random Forest | Ensemble (Bagging) | Feature importance |
| XGBoost | Gradient Boosting | Best tabular performance |
| LightGBM | Gradient Boosting | Fast leaf-wise growth |

All models use **RandomizedSearchCV** with **5-fold StratifiedKFold** cross-validation,
optimizing for **F1-Macro** (not accuracy — critical for imbalanced medical data).

### Explainability
- **SHAP (SHapley Additive exPlanations)**: Game-theory based method that explains
  individual predictions by computing each feature's contribution.
- **Clinical Alerts**: SHAP values translated into human-readable medical language.

### Evaluation Metrics
- **F1-Macro**: Primary metric — equal weight to all risk classes
- **Precision**: Avoid false alarms
- **Recall**: Don't miss high-risk patients
- **ROC-AUC**: Overall model discrimination ability

## 3-Class Risk Stratification

Instead of binary yes/no prediction, we use clinically meaningful risk tiers:

| Level | Description | Action |
|-------|-------------|--------|
| **Low** | Healthy indicators, no disease markers | Routine monitoring |
| **Medium** | Borderline values, elevated risk factors | Enhanced screening |
| **High** | Disease present or critical risk factors | Immediate intervention |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core ML | Python 3.12, scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| API | FastAPI |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Testing | pytest |

## References

1. Lundberg, S.M. and Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
2. WHO Bangladesh Health Workforce Data (2022).
3. Chawla, N.V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*.
4. Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
5. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*.

## Author

**Md. Rakib**
- Junior Software Developer, Combosoft Ltd.
- BSc Computer Science, Daffodil International University

---

*BanglaHealth-AI is a research portfolio project demonstrating explainable ML
for healthcare in low-resource settings.*
""")
