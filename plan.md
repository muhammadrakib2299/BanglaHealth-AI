# BanglaHealth-AI — Project Plan

## Vision

An **Explainable AI system** for patient risk stratification in Bangladesh's low-resource healthcare settings. The system predicts patient risk levels (Low / Medium / High) using clinical data and provides **human-readable explanations** for every prediction — empowering doctors, nurses, and health workers to make informed decisions even without specialist support.

> Bangladesh has only **0.58 physicians per 1,000 people** (WHO, 2022). This project bridges the gap between collected clinical data and actionable medical insight.

---

## Project Architecture

```
BanglaHealth-AI/
├── data/
│   ├── raw/                    # Original downloaded datasets
│   ├── processed/              # Cleaned, engineered datasets
│   └── README.md               # Dataset documentation & sources
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Cleaning & Feature Engineering
│   ├── 03_model_training.ipynb # Train & compare 4 models
│   ├── 04_evaluation.ipynb     # Metrics, confusion matrices, comparison
│   └── 05_explainability.ipynb # SHAP analysis & clinical insights
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load & validate datasets
│   ├── preprocessing.py        # Cleaning, encoding, scaling, SMOTE
│   ├── feature_engineering.py  # Create derived clinical features
│   ├── model.py                # Train, evaluate, save models
│   ├── explainer.py            # SHAP explanations & clinical alerts
│   └── utils.py                # Shared helpers
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── schemas.py              # Pydantic request/response models
│   └── routes/
│       ├── __init__.py
│       ├── predict.py          # Single & batch prediction endpoints
│       └── explain.py          # Explanation endpoints
│
├── app/
│   ├── app.py                  # Streamlit main dashboard
│   ├── pages/
│   │   ├── patient_predict.py  # Individual patient prediction
│   │   ├── batch_predict.py    # CSV batch prediction
│   │   ├── model_compare.py    # Model comparison dashboard
│   │   ├── eda_dashboard.py    # Interactive EDA visualizations
│   │   └── about.py            # Project info & methodology
│   └── components/
│       ├── risk_card.py        # Risk level display component
│       ├── shap_plots.py       # SHAP visualization wrappers
│       └── clinical_alert.py   # Human-readable alert generator
│
├── models/                     # Saved trained models (.joblib)
├── outputs/                    # Generated plots, reports, CSVs
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
└── plan.md
```

---

## Technology Stack

### Core Language

| Tech | Version | Why |
|------|---------|-----|
| **Python** | 3.12 | Latest stable release. Better error messages, performance improvements over 3.10. Full compatibility with all ML libraries. |

### Data & ML

| Tech | Purpose | Details |
|------|---------|---------|
| **pandas** | Data manipulation | The standard for tabular data. Read CSV/Excel, filter, group, merge, pivot. Everything starts here. |
| **NumPy** | Numerical computing | Underlies pandas and scikit-learn. Fast array operations, linear algebra, statistical functions. |
| **scikit-learn** | Classical ML models | Provides Logistic Regression, Random Forest, cross-validation, metrics, preprocessing (StandardScaler, train_test_split, StratifiedKFold). The backbone of the ML pipeline. |
| **XGBoost** | Gradient boosting | Our primary model. Handles missing values natively, regularization prevents overfitting, consistently wins on tabular data. Faster than Random Forest with better accuracy. |
| **LightGBM** | Light gradient boosting | Microsoft's alternative to XGBoost. Faster training on larger datasets, leaf-wise tree growth, lower memory usage. We compare both to pick the best. |
| **imbalanced-learn** | Class balancing | Provides SMOTE (Synthetic Minority Oversampling Technique). Medical data is always imbalanced — most patients are "Low risk". Without SMOTE, models ignore High-risk patients. |
| **joblib** | Model serialization | Save trained models to disk (.joblib files). Faster than pickle for NumPy arrays. Models are loaded once at API startup. |

### Explainability

| Tech | Purpose | Details |
|------|---------|---------|
| **SHAP** | Model explanations | SHapley Additive exPlanations — the gold standard for ML interpretability. Based on game theory (Shapley values). Tells you exactly how much each feature pushed a prediction toward or away from risk. Works with any model. |
| **matplotlib** | Static plots | Base plotting library. Used for confusion matrices, bar charts, heatmaps. Publication-quality figures for the research paper. |
| **seaborn** | Statistical plots | Built on matplotlib. Better defaults, cleaner syntax. Used for distribution plots, correlation heatmaps, pair plots in EDA. |
| **Plotly** | Interactive plots | Used in Streamlit dashboard. Hover tooltips, zoom, pan. Makes SHAP waterfall plots and feature importance charts interactive. |

### Web — API (Backend)

| Tech | Purpose | Details |
|------|---------|---------|
| **FastAPI** | REST API framework | Modern, async Python web framework. Auto-generates OpenAPI/Swagger docs. Type-safe with Pydantic. 10x faster than Flask. Endpoints for single prediction, batch prediction, and explanation retrieval. |
| **Pydantic** | Data validation | Validates incoming patient data against expected types and ranges. Example: Blood Pressure must be 0-300, Age must be 0-120. Prevents garbage predictions. |
| **Uvicorn** | ASGI server | Production-grade async server for FastAPI. Handles concurrent requests efficiently. |

### Web — Dashboard (Frontend)

| Tech | Purpose | Details |
|------|---------|---------|
| **Streamlit** | Interactive dashboard | Python-only web framework for data apps. No HTML/CSS/JS needed. Perfect for ML demos — input widgets, charts, and SHAP plots in pure Python. Multi-page app with sidebar navigation. |

### Testing & Quality

| Tech | Purpose | Details |
|------|---------|---------|
| **pytest** | Testing framework | Unit tests for preprocessing, feature engineering, model predictions, and API endpoints. Ensures nothing breaks when you change code. |
| **ruff** | Linting & formatting | Extremely fast Python linter (written in Rust). Replaces flake8 + black + isort. Keeps code clean and consistent. |

### DevOps

| Tech | Purpose | Details |
|------|---------|---------|
| **Git + GitHub** | Version control | Track all changes. Public repo showcases your work. Clean commit history shows research methodology. |
| **GitHub Actions** | CI/CD | Auto-run tests on every push. Ensures code quality. Impressive on a portfolio project — shows engineering maturity. |

---

## Datasets

### Primary Datasets (Public, Free)

| Dataset | Source | Samples | Features | Target |
|---------|--------|---------|----------|--------|
| **Pima Indians Diabetes** | UCI / Kaggle | 768 | 8 (Glucose, BP, BMI, Insulin, Age, etc.) | Diabetes (binary → converted to 3-class risk) |
| **UCI Heart Disease (Cleveland)** | UCI Repository | 303 | 13 (Age, Sex, Cholesterol, RestBP, MaxHR, etc.) | Heart Disease (binary → converted to 3-class risk) |

### Risk Label Engineering

We convert binary labels into **3-tier clinical risk scores**:

```
Low Risk    → Negative outcome + low-severity features
Medium Risk → Negative outcome + elevated features (borderline patients)
High Risk   → Positive outcome
```

For diabetes (example):
- **Low:** No diabetes + Glucose < 120 + BMI < 30
- **Medium:** No diabetes + (Glucose ≥ 120 OR BMI ≥ 30) — borderline
- **High:** Diabetic

This is more clinically useful than binary yes/no prediction.

---

## Features & Functionality

### Core Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **3-Class Risk Prediction** | Predict Low / Medium / High risk with confidence scores for each class |
| 2 | **Multi-Disease Support** | Separate models for diabetes risk and heart disease risk |
| 3 | **SHAP Explanations** | Every prediction includes feature-level explanations: "Glucose=180 increased risk by +0.23" |
| 4 | **Clinical Alerts** | Human-readable alerts: "⚠ High glucose (180 mg/dL) significantly increases diabetes risk. Consider HbA1c test." |
| 5 | **Model Comparison** | Side-by-side comparison of 4 models (LR, RF, XGBoost, LightGBM) with F1-Macro, Precision, Recall |
| 6 | **Interactive Dashboard** | Streamlit multi-page app with input forms, charts, and real-time predictions |
| 7 | **REST API** | FastAPI endpoints for integration with hospital systems or mobile apps |
| 8 | **Batch Prediction** | Upload a CSV of patients → get risk scores + explanations for all |

### Advanced Features

| # | Feature | Description |
|---|---------|-------------|
| 9 | **Feature Importance Dashboard** | Global view: which features matter most across all patients (bar + beeswarm plots) |
| 10 | **Individual Patient Report** | SHAP waterfall plot for a single patient — shows exactly why this patient got this risk level |
| 11 | **What-If Analysis** | Adjust patient values with sliders and see how risk changes in real-time |
| 12 | **Confidence Indicator** | Show prediction confidence — flag cases where the model is uncertain (close probabilities between classes) |
| 13 | **PDF Report Generation** | Generate a downloadable PDF report for a patient with risk score, explanations, and recommendations |
| 14 | **Dataset Explorer** | Interactive EDA page — distributions, correlations, class balance visualizations |
| 15 | **Hyperparameter Tuning Results** | Show how model performance varies with different hyperparameters (RandomizedSearchCV) |

---

## ML Pipeline — Technical Detail

### Step 1: Data Loading & Validation

```
Raw CSV → Validate columns & types → Check for impossible values → Load into DataFrame
```

- Reject rows where Glucose = 0, BloodPressure = 0 (biologically impossible — these are missing values coded as 0)
- Replace with median values (not mean — robust to outliers)

### Step 2: Feature Engineering

| New Feature | Formula | Clinical Reasoning |
|-------------|---------|-------------------|
| `GlucoseBMI` | Glucose × BMI | Captures metabolic syndrome interaction — high glucose AND high BMI together is more dangerous than either alone |
| `AgeRisk` | Age / BMI | Older patients with lower BMI may still have age-related risk |
| `BPCategory` | Binned BloodPressure | Normal (<120), Elevated (120-129), High (130+) per AHA guidelines |
| `InsulinLog` | log(1 + Insulin) | Insulin is heavily right-skewed. Log transform normalizes distribution for better model performance |

### Step 3: Preprocessing

```
Feature Engineering → StandardScaler → Train/Test Split (80/20, stratified) → SMOTE on training set only
```

- **StandardScaler**: Mean=0, StdDev=1. Required for Logistic Regression. Good practice for all models.
- **Stratified split**: Maintains class ratios in both train and test sets.
- **SMOTE on training only**: Never leak synthetic data into test set. This is a common mistake — we avoid it.

### Step 4: Model Training

| Model | Key Hyperparameters | Strengths |
|-------|-------------------|-----------|
| **Logistic Regression** | C=tuned, max_iter=1000 | Baseline. Fast. Inherently interpretable. Coefficients directly show feature importance. |
| **Random Forest** | n_estimators=tuned, max_depth=tuned | Ensemble of decision trees. Handles non-linear relationships. Built-in feature importance. |
| **XGBoost** | learning_rate=tuned, max_depth=tuned, n_estimators=tuned | Gradient boosting. Best performance on tabular data. Regularization (L1+L2) prevents overfitting. |
| **LightGBM** | learning_rate=tuned, num_leaves=tuned, n_estimators=tuned | Leaf-wise growth (vs XGBoost's level-wise). Faster training. Better on larger datasets. |

**Hyperparameter Tuning**: RandomizedSearchCV with 5-fold StratifiedKFold, scoring=F1-Macro.

### Step 5: Evaluation

| Metric | Why This Metric |
|--------|----------------|
| **F1-Macro** | Primary metric. Balances precision and recall equally across all 3 classes. Doesn't let the model ignore minority classes. |
| **Precision (per class)** | How many patients flagged as "High Risk" actually are? Important to avoid unnecessary panic. |
| **Recall (per class)** | How many actual "High Risk" patients did we catch? Missing a high-risk patient is dangerous. |
| **Confusion Matrix** | Shows exactly where misclassifications happen — e.g., "Medium" predicted as "Low" is concerning. |
| **ROC-AUC (One-vs-Rest)** | Overall discriminative ability of the model across all thresholds. |

### Step 6: Explainability (SHAP)

```
Trained Model → SHAP TreeExplainer → Shapley Values → Visualizations + Clinical Alerts
```

- **Global**: Feature importance bar chart — which features matter most overall
- **Global**: Beeswarm plot — how each feature value pushes predictions
- **Local**: Waterfall plot — why THIS specific patient got THIS risk score
- **Clinical Alert**: Translate SHAP values into doctor-friendly language

---

## Implementation Phases

### Phase 1: Foundation (Day 1-2)

- [x] Create project plan
- [ ] Initialize project structure (folders, files)
- [ ] Set up Python 3.12 virtual environment
- [ ] Install dependencies (requirements.txt)
- [ ] Initialize Git repository
- [ ] Create .gitignore
- [ ] Download and document datasets
- [ ] Create data/README.md with dataset sources and licenses

### Phase 2: EDA & Preprocessing (Day 3-4)

- [ ] Notebook 01_eda.ipynb
  - Load both datasets
  - Statistical summaries (describe, info, value_counts)
  - Distribution plots for all features
  - Correlation heatmap
  - Class distribution visualization
  - Identify missing values coded as zeros
  - Document clinical observations
- [ ] Notebook 02_preprocessing.ipynb
  - Handle impossible zero values (replace with median)
  - Create 3-class risk labels
  - Feature engineering (GlucoseBMI, AgeRisk, BPCategory, InsulinLog)
  - StandardScaler normalization
  - Stratified train/test split
  - SMOTE on training set
  - Save processed data to data/processed/
- [ ] Write src/data_loader.py
- [ ] Write src/preprocessing.py
- [ ] Write src/feature_engineering.py

### Phase 3: Model Training & Evaluation (Day 5-7)

- [ ] Notebook 03_model_training.ipynb
  - Train Logistic Regression
  - Train Random Forest
  - Train XGBoost
  - Train LightGBM
  - RandomizedSearchCV for each model
  - 5-fold StratifiedKFold cross-validation
  - Save all models to models/
- [ ] Notebook 04_evaluation.ipynb
  - Classification reports for all models
  - Confusion matrices (heatmap format)
  - Model comparison table (F1-Macro, Precision, Recall, ROC-AUC)
  - Best model selection with justification
  - Save comparison results to outputs/
- [ ] Write src/model.py

### Phase 4: Explainability (Day 8-9)

- [ ] Notebook 05_explainability.ipynb
  - SHAP TreeExplainer for XGBoost/LightGBM
  - Global feature importance (bar plot)
  - Beeswarm plot
  - Waterfall plots for sample patients (Low, Medium, High risk examples)
  - Clinical alert generation
  - Save SHAP plots to outputs/
- [ ] Write src/explainer.py
- [ ] Write clinical alert templates

### Phase 5: API Development (Day 10-11)

- [ ] FastAPI application setup
- [ ] Pydantic schemas for patient data input/output
- [ ] POST /predict — single patient prediction
- [ ] POST /predict/batch — CSV batch prediction
- [ ] GET /explain/{prediction_id} — SHAP explanation
- [ ] GET /models/compare — model comparison data
- [ ] Auto-generated Swagger docs
- [ ] API tests with pytest

### Phase 6: Dashboard (Day 12-14)

- [ ] Streamlit multi-page app
- [ ] Page: Patient Prediction — input form → risk score + SHAP waterfall
- [ ] Page: Batch Prediction — CSV upload → results table + download
- [ ] Page: Model Comparison — charts comparing all 4 models
- [ ] Page: EDA Dashboard — interactive dataset exploration
- [ ] Page: What-If Analysis — sliders to adjust features, see risk change
- [ ] Page: About — methodology, team, references
- [ ] Clinical alert cards with color coding
- [ ] PDF report generation

### Phase 7: Polish & Deploy (Day 15-17)

- [ ] Write comprehensive README.md
- [ ] Add GitHub Actions CI (run tests on push)
- [ ] Code cleanup with ruff
- [ ] Final testing
- [ ] Screenshots for README
- [ ] Record demo GIF/video

---

## Additional Suggestions

### 1. Add a "Model Card"
Include a model card (Google's standard) documenting: what the model does, training data, intended use, limitations, ethical considerations. Admissions committees love this — shows you think about responsible AI.

### 2. Clinical Validation Section
Add a section in your paper/README discussing how this would be validated clinically before deployment. Shows you understand the gap between ML research and real healthcare deployment.

### 3. Fairness Analysis
Add basic fairness analysis — does the model perform equally across age groups? Gender? This is cutting-edge in health AI research and will stand out.

### 4. Dockerize the Project
Add a Dockerfile so anyone can run the entire project (API + Dashboard) with one command. Shows DevOps awareness.

### 5. Consider Bangla Language Support (Future Scope)
Even though the interface is English, mention in the paper that clinical alerts could be translated to Bangla for rural health workers. This frames the project in real-world impact.

### 6. Use DVC (Data Version Control)
Track your datasets and models with DVC alongside Git. Research projects need reproducibility — DVC ensures anyone can replicate your exact data pipeline.

---

## Key Metrics to Showcase

For your MSc applications, highlight these numbers in your README and paper:

| Metric | Target |
|--------|--------|
| F1-Macro (best model) | > 0.80 |
| Number of models compared | 4 |
| Explainability method | SHAP (Shapley values) |
| Datasets used | 2 (1,071 total patients) |
| Engineered features | 4 new clinical features |
| API endpoints | 4+ |
| Test coverage | > 80% |
| Research paper | 4-page arXiv preprint |

---

## References

1. Lundberg, S.M. and Lee, S.I., 2017. "A Unified Approach to Interpreting Model Predictions." NeurIPS.
2. WHO Bangladesh Health Workforce Data, 2022.
3. Chawla, N.V., et al., 2002. "SMOTE: Synthetic Minority Over-sampling Technique." JAIR.
4. Chen, T. and Guestrin, C., 2016. "XGBoost: A Scalable Tree Boosting System." KDD.
5. Ke, G., et al., 2017. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS.
