# BanglaHealth-AI

**Explainable AI for Patient Risk Stratification in Low-Resource Clinical Settings**

> Bangladesh has only **0.58 physicians per 1,000 people** (WHO, 2022). Healthcare data is collected but rarely analyzed predictively. BanglaHealth-AI bridges this gap by transforming clinical tabular data into explainable, three-tier risk scores — empowering health workers to make informed decisions.

## What It Does

- Predicts patient risk levels (**Low / Medium / High**) for diabetes and heart disease
- Explains every prediction using **SHAP** (SHapley Additive exPlanations)
- Generates **human-readable clinical alerts** (e.g., "High glucose significantly increases diabetes risk")
- Compares **4 ML models**: Logistic Regression, Random Forest, XGBoost, LightGBM

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML/Data | Python 3.12, scikit-learn, XGBoost, LightGBM, SHAP, pandas |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, ruff |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/BanglaHealth-AI.git
cd BanglaHealth-AI

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run notebooks (training)
jupyter notebook notebooks/

# Run API
uvicorn api.main:app --reload

# Run Dashboard
streamlit run app/app.py
```

## Project Structure

```
BanglaHealth-AI/
├── data/               # Raw and processed datasets
├── notebooks/          # 5 Jupyter notebooks (EDA → Explainability)
├── src/                # Core ML pipeline modules
├── api/                # FastAPI REST endpoints
├── app/                # Streamlit interactive dashboard
├── models/             # Saved trained models (.joblib)
├── outputs/            # Generated plots, reports, CSVs
└── tests/              # pytest test suite
```

## Datasets

| Dataset | Samples | Features | Source |
|---------|---------|----------|--------|
| Pima Indians Diabetes | 768 | 8 | UCI / Kaggle |
| UCI Heart Disease (Cleveland) | 303 | 13 | UCI Repository |

## Model Performance

| Model | F1-Macro | Precision | Recall |
|-------|----------|-----------|--------|
| Logistic Regression | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |
| LightGBM | - | - | - |

> Results will be updated after model training.

## Key Features

- **3-Class Risk Prediction** with confidence scores
- **SHAP Explanations** for every prediction
- **What-If Analysis** — adjust patient values, see risk change in real-time
- **Batch Prediction** — upload CSV, get results for all patients
- **PDF Report Generation** for clinical use
- **Model Comparison Dashboard**

## Author

**Md. Rakib**
Junior Software Developer, Combosoft Ltd.
BSc Computer Science, Daffodil International University

## License

MIT License
