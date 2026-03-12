# BanglaHealth-AI

**Explainable AI for Patient Risk Stratification in Low-Resource Clinical Settings**

> Bangladesh has only **0.58 physicians per 1,000 people** (WHO, 2022). Healthcare data is collected but rarely analyzed predictively. BanglaHealth-AI bridges this gap by transforming clinical tabular data into explainable, three-tier risk scores — empowering health workers to make informed decisions.

## What It Does

- Predicts patient risk levels (**Low / Medium / High**) for diabetes and heart disease
- Explains every prediction using **SHAP** (SHapley Additive exPlanations)
- Generates **human-readable clinical alerts** (e.g., "High glucose significantly increases diabetes risk")
- Compares **4 ML models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluates **fairness** across demographic groups (age, sex, BMI)
- Provides **interactive dashboard** with What-If Analysis

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML/Data | Python 3.12, scikit-learn, XGBoost, LightGBM, SHAP, pandas |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Reports | fpdf2 (PDF generation) |
| Testing | pytest, ruff |
| CI/CD | GitHub Actions |
| DevOps | Docker, docker-compose |

## Quick Start

### Option 1: Local Setup

```bash
# Clone
git clone https://github.com/muhammadrakib2299/BanglaHealth-AI.git
cd BanglaHealth-AI

# Automated setup
bash setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run notebooks (train models)
jupyter notebook notebooks/

# Launch dashboard
streamlit run app/app.py

# Launch API
uvicorn api.main:app --reload

# Run tests
pytest tests/ -v
```

### Option 2: Docker

```bash
# Run everything with one command
docker-compose up --build

# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## Project Structure

```
BanglaHealth-AI/
├── data/
│   ├── raw/                    # Original datasets (diabetes.csv, heart.csv)
│   └── processed/              # Cleaned, engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Cleaning & Feature Engineering
│   ├── 03_model_training.ipynb # Train 4 models with hyperparameter tuning
│   ├── 04_evaluation.ipynb     # Metrics, confusion matrices, comparison
│   ├── 05_explainability.ipynb # SHAP analysis & clinical insights
│   └── 06_fairness_analysis.ipynb  # Demographic fairness evaluation
├── src/
│   ├── data_loader.py          # Load & validate datasets
│   ├── preprocessing.py        # Cleaning, scaling, SMOTE
│   ├── feature_engineering.py  # Derived clinical features
│   ├── model.py                # Train, evaluate, save models
│   ├── explainer.py            # SHAP explanations & clinical alerts
│   ├── fairness.py             # Fairness analysis across demographics
│   └── utils.py                # Plotting helpers
├── api/                        # FastAPI REST endpoints
│   ├── main.py                 # App entry point
│   ├── schemas.py              # Pydantic validation
│   └── routes/                 # Predict & explain endpoints
├── app/                        # Streamlit dashboard
│   ├── app.py                  # Main page
│   ├── pages/
│   │   ├── patient_predict.py  # Single patient prediction + PDF download
│   │   ├── batch_predict.py    # CSV batch prediction
│   │   ├── model_compare.py    # Model comparison charts
│   │   ├── eda_dashboard.py    # Interactive EDA
│   │   ├── what_if.py          # What-If Analysis with live SHAP
│   │   └── about.py            # Methodology & references
│   └── components/             # Reusable UI components
├── models/                     # Saved trained models (.joblib)
├── outputs/                    # Generated plots, reports, CSVs
├── tests/                      # pytest test suite
├── Dockerfile                  # Container setup
├── docker-compose.yml          # Multi-service orchestration
├── MODEL_CARD.md               # Model documentation (Google standard)
└── plan.md                     # Detailed project plan
```

## Datasets

| Dataset | Samples | Features | Target | Source |
|---------|---------|----------|--------|--------|
| Pima Indians Diabetes | 768 | 8 | Diabetes (binary → 3-class risk) | UCI / Kaggle |
| UCI Heart Disease (Cleveland) | 303 | 13 | Heart Disease (binary → 3-class risk) | UCI Repository |

### 3-Class Risk Stratification

| Level | Description | Clinical Action |
|-------|-------------|----------------|
| **Low** | Healthy indicators | Routine monitoring |
| **Medium** | Borderline values | Enhanced screening |
| **High** | Disease markers present | Immediate intervention |

## Model Performance

| Model | F1-Macro | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |

> Results will be populated after running the training notebooks.

## Key Features

| Feature | Description |
|---------|-------------|
| **3-Class Risk Prediction** | Low / Medium / High with confidence scores |
| **SHAP Explanations** | Feature-level impact for every prediction |
| **Clinical Alerts** | Human-readable medical insights |
| **What-If Analysis** | Adjust patient values with sliders, see risk change in real-time |
| **Batch Prediction** | Upload CSV, get predictions for all patients |
| **PDF Reports** | Downloadable patient report with risk, alerts, and SHAP values |
| **Model Comparison** | Side-by-side metrics for all 4 models |
| **Fairness Analysis** | Performance evaluation across age, sex, and BMI groups |
| **REST API** | FastAPI endpoints with auto-generated Swagger docs |
| **Interactive EDA** | Explore datasets with interactive Plotly charts |

## Methodology

1. **Data Cleaning** — Replace biologically impossible zeros with median imputation
2. **Feature Engineering** — GlucoseBMI (metabolic interaction), AgeRisk, InsulinLog, BPCategory
3. **Preprocessing** — StandardScaler, stratified 80/20 split, SMOTE on training set only
4. **Training** — 4 models with RandomizedSearchCV, 5-fold StratifiedKFold, F1-Macro optimization
5. **Explainability** — SHAP TreeExplainer for global and local feature importance
6. **Fairness** — Evaluate performance gaps across demographic subgroups

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/diabetes` | Single diabetes prediction |
| POST | `/predict/heart` | Single heart disease prediction |
| POST | `/predict/diabetes/batch` | Batch diabetes prediction (CSV) |
| POST | `/predict/heart/batch` | Batch heart disease prediction (CSV) |
| POST | `/explain/diabetes` | SHAP explanation for diabetes |
| POST | `/explain/heart` | SHAP explanation for heart disease |
| GET | `/models` | List available trained models |
| GET | `/docs` | Interactive Swagger documentation |

## Documentation

- [**PROJECT PLAN**](plan.md) — Detailed implementation roadmap
- [**MODEL CARD**](MODEL_CARD.md) — Model documentation following Google's standard

## Author

**Md. Rakib**
- Junior Software Developer, Combosoft Ltd.
- BSc Computer Science, Daffodil International University

## License

MIT License
