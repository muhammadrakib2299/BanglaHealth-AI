# Model Card — BanglaHealth-AI

Following [Google's Model Card framework](https://arxiv.org/abs/1810.03993) for transparent ML documentation.

## Model Details

- **Developed by:** Md. Rakib
- **Model type:** Supervised classification (3-class: Low / Medium / High risk)
- **Models used:** Logistic Regression, Random Forest, XGBoost, LightGBM
- **Primary model:** XGBoost (best F1-Macro performance)
- **Language:** Python 3.12
- **Framework:** scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **License:** MIT

## Intended Use

### Primary Use
- **Research and education:** Demonstrating explainable AI for healthcare risk stratification
- **Portfolio project:** Showcasing ML pipeline design for academic applications
- **Proof of concept:** Showing how clinical data can be used for patient risk scoring in low-resource settings

### Out-of-Scope Uses
- **NOT for clinical diagnosis** — This model has not been validated for medical use
- **NOT for treatment decisions** — Always consult qualified healthcare professionals
- **NOT for deployment in hospitals** without proper clinical trials and regulatory approval

## Training Data

| Dataset | Source | Samples | Features | License |
|---------|--------|---------|----------|---------|
| Pima Indians Diabetes | UCI / Kaggle | 768 | 8 clinical measurements | CC0 Public Domain |
| UCI Heart Disease (Cleveland) | UCI Repository | 303 | 13 clinical measurements | CC BY 4.0 |

### Data Limitations
- **Small sample sizes** — 768 and 303 patients respectively
- **Single demographics** — Pima dataset is exclusively from Pima Indian women aged 21+
- **Geographic bias** — Data not from Bangladesh; used as a proof-of-concept
- **Historical data** — Collected decades ago; clinical standards may have changed
- **Missing values** — Original data uses 0 for missing values (handled via median imputation)

## Preprocessing

1. **Missing value handling:** Biologically impossible zeros replaced with column medians
2. **Risk label engineering:** Binary outcomes converted to 3-tier risk (Low/Medium/High)
3. **Feature engineering:** 4 new clinical features (GlucoseBMI, AgeRisk, InsulinLog, BPCategory)
4. **Normalization:** StandardScaler (mean=0, std=1)
5. **Class balancing:** SMOTE applied to training set only
6. **Split:** 80/20 stratified train/test

## Evaluation Metrics

| Metric | Why This Metric |
|--------|----------------|
| **F1-Macro** (primary) | Equal weight to all classes — prevents ignoring minority risk groups |
| **Precision** | Minimizes false alarms (unnecessary patient anxiety) |
| **Recall** | Maximizes catching actual high-risk patients (safety critical) |
| **ROC-AUC** | Overall discrimination ability across all thresholds |

### Performance

> Results populated after model training. Run notebooks to generate.

| Model | F1-Macro | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |

## Ethical Considerations

### Bias and Fairness
- The Pima Diabetes dataset only includes women of Pima Indian heritage — the model may not generalize to other populations
- The Heart Disease dataset skews male (approx. 68% male patients)
- **Recommendation:** Before any real-world deployment, the model MUST be retrained on representative Bangladeshi population data

### Privacy
- All training data is from public, anonymized datasets
- No personally identifiable information (PII) is used or stored
- Patient data entered in the dashboard is processed locally and not saved

### Clinical Safety
- Model predictions include confidence scores — low-confidence predictions should be treated with caution
- Clinical alerts are generated to provide context, not replace medical judgment
- A disclaimer is included in all PDF reports

### Potential Harms
- **Over-reliance:** Users may trust model predictions without seeking professional medical advice
- **Misclassification:** Medium-risk patients classified as Low could miss early intervention
- **Demographic bias:** Model trained on non-Bangladeshi data may perform differently on target population

## Recommendations

1. **Do not use for clinical decisions** without proper validation studies
2. **Retrain on local data** before any deployment in Bangladesh
3. **Always pair with clinical expertise** — this is a decision-support tool, not a replacement
4. **Monitor for bias** across demographic groups when deployed
5. **Regular retraining** as new clinical data becomes available

## Citation

If you use this model or methodology, please cite:

```
@software{banglahealth_ai,
  author = {Md. Rakib},
  title = {BanglaHealth-AI: Explainable AI for Patient Risk Stratification},
  year = {2026},
  url = {https://github.com/muhammadrakib2299/BanglaHealth-AI}
}
```
