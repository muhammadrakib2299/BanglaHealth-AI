import { useState } from "react";

const phases = [
  {
    id: 1,
    week: "Week 1",
    title: "Setup & Data",
    subtitle: "Foundation",
    color: "#00C896",
    icon: "⚙️",
    steps: [
      {
        title: "Create Project Folder Structure",
        description: "Set up the entire repository layout before writing a single line of ML code. Professional structure signals research maturity to admissions committees.",
        code: `banglahealth-ai/
├── data/
│   ├── raw/               # Original downloaded datasets
│   └── processed/         # Cleaned, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Reusable preprocessing functions
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Metrics & visualization
│   └── explain.py          # SHAP explanation functions
├── models/
│   └── saved/              # Saved .pkl model files
├── outputs/
│   ├── figures/            # All charts, SHAP plots
│   └── reports/            # Model performance reports
├── paper/                  # Your arXiv writeup (Week 6-8)
├── requirements.txt
├── README.md
└── .gitignore`,
        tip: "Create this on GitHub first, then clone locally. Your GitHub link goes in your SOP."
      },
      {
        title: "Install Dependencies",
        description: "Set up a clean virtual environment. This is non-negotiable for reproducible research — any reviewer must be able to clone and run your code.",
        code: `# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\\Scripts\\activate  # Windows

# Install all required packages
pip install pandas numpy scikit-learn xgboost shap
pip install matplotlib seaborn plotly
pip install jupyter notebook ipykernel
pip install imbalanced-learn  # For handling class imbalance
pip install joblib             # For saving models

# Freeze dependencies
pip freeze > requirements.txt`,
        tip: "Always use a virtual environment. It shows you understand reproducible science."
      },
      {
        title: "Download Datasets",
        description: "You will use TWO public datasets and merge them into one narrative: chronic disease risk in low-resource settings. This mirrors Bangladesh's actual disease burden.",
        code: `# Dataset 1: Pima Indians Diabetes (UCI)
# URL: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# 768 patients, 8 clinical features, binary outcome
# Features: Glucose, BloodPressure, BMI, Age, Insulin, etc.

# Dataset 2: UCI Heart Disease (Cleveland)
# URL: https://archive.ics.uci.edu/dataset/45/heart+disease
# 303 patients, 13 clinical features, binary outcome
# Features: Age, ChestPain, Cholesterol, MaxHR, etc.

# Place raw CSVs in:
# data/raw/diabetes.csv
# data/raw/heart_disease.csv`,
        tip: "In your README, write one sentence explaining why you chose these datasets: they represent the top two non-communicable disease burdens in Bangladesh."
      }
    ]
  },
  {
    id: 2,
    week: "Week 1",
    title: "EDA & Preprocessing",
    subtitle: "Understanding Data",
    color: "#0099FF",
    icon: "🔍",
    steps: [
      {
        title: "Notebook 01: Exploratory Data Analysis",
        description: "EDA is where you demonstrate scientific thinking. This is not just plotting charts — it is asking questions of the data and documenting what you find.",
        code: `import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/diabetes.csv')

# 1. Basic overview
print(df.shape)          # How many patients, features?
print(df.dtypes)         # What type is each feature?
print(df.isnull().sum()) # Any missing values?
print(df.describe())     # Statistical summary

# 2. Target distribution (class imbalance check!)
df['Outcome'].value_counts().plot(kind='bar')
plt.title('Risk Distribution: Diabetic vs Non-Diabetic')
plt.savefig('outputs/figures/class_distribution.png')

# 3. Feature distributions by outcome
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(df.columns[:-1]):
    sns.histplot(data=df, x=col, hue='Outcome', ax=axes[i//4][i%4])
plt.savefig('outputs/figures/feature_distributions.png')

# 4. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.savefig('outputs/figures/correlation_heatmap.png')

# KEY INSIGHT TO DOCUMENT:
# Glucose is the strongest predictor (correlation ~0.47)
# BMI and Age also strongly correlated with outcome
# Some features have 0 values that are biologically impossible (missing data encoded as 0)`,
        tip: "Write markdown cells in your notebook explaining WHAT you see and WHY it matters clinically. This shows domain awareness."
      },
      {
        title: "Notebook 02: Preprocessing Pipeline",
        description: "Real clinical data is messy. Your HMS experience taught you this. Show that you can clean data professionally.",
        code: `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/raw/diabetes.csv')

# STEP 1: Handle impossible zero values
# Glucose=0, BloodPressure=0 is biologically impossible → these are missing values
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_not_accepted:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())  # Impute with median

# STEP 2: Create a 3-class risk label (Low / Medium / High)
# This makes clinical output more meaningful than binary 0/1
def assign_risk(row):
    score = 0
    if row['Glucose'] > 140: score += 2
    elif row['Glucose'] > 100: score += 1
    if row['BMI'] > 30: score += 1
    if row['Age'] > 45: score += 1
    if row['Outcome'] == 1: score += 2
    if score >= 4: return 'High'
    elif score >= 2: return 'Medium'
    else: return 'Low'

df['RiskLevel'] = df.apply(assign_risk, axis=1)

# STEP 3: Feature engineering
df['GlucoseBMI'] = df['Glucose'] * df['BMI']       # Interaction feature
df['AgeRisk'] = df['Age'] / df['BMI']               # Age-adjusted BMI risk

# STEP 4: Split and scale
X = df.drop(['Outcome', 'RiskLevel'], axis=1)
y = df['RiskLevel']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 5: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Save processed data
pd.DataFrame(X_train_balanced).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(y_train_balanced).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

import joblib
joblib.dump(scaler, 'models/saved/scaler.pkl')
print("Preprocessing complete. Data saved.")`,
        tip: "Document every decision in markdown: WHY median imputation? WHY SMOTE? This is what separates a research project from a Kaggle notebook."
      }
    ]
  },
  {
    id: 3,
    week: "Week 2",
    title: "Model Training",
    subtitle: "The Core ML Work",
    color: "#FF6B35",
    icon: "🤖",
    steps: [
      {
        title: "Notebook 03: Train 3 Models & Compare",
        description: "Training three models and comparing them scientifically is the heart of the project. This directly mirrors real research methodology.",
        code: `import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv').values
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_test = pd.read_csv('data/processed/X_test.csv').values
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Define all 3 models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, C=0.1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric='mlogloss',
        use_label_encoder=False
    )
}

# Train with 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    
    # Fit on full training set
    model.fit(X_train, y_train)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }
    
    print(f"  CV F1-Macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Save model
    safe_name = name.replace(' ', '_').lower()
    joblib.dump(model, f'models/saved/{safe_name}.pkl')

print("\\nAll models trained and saved!")`,
        tip: "Always use cross-validation, never just a single train/test split. This is what reviewers at Paris-Saclay and Lille specifically look for."
      },
      {
        title: "Notebook 04: Evaluation & Visualization",
        description: "Rigorous evaluation is what separates your project from a tutorial. Every metric must be justified clinically, not just computationally.",
        code: `from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib, pandas as pd

X_test = pd.read_csv('data/processed/X_test.csv').values
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

model_names = ['logistic_regression', 'random_forest', 'xgboost']
display_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
colors = ['#0099FF', '#00C896', '#FF6B35']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (mname, dname, color) in enumerate(zip(model_names, display_names, colors)):
    model = joblib.load(f'models/saved/{mname}.pkl')
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[i], xticklabels=['Low','Med','High'],
                yticklabels=['Low','Med','High'])
    axes[i].set_title(f'{dname}', fontsize=14, fontweight='bold', color=color)
    axes[i].set_ylabel('Actual Risk')
    axes[i].set_xlabel('Predicted Risk')
    
    # Print classification report
    print(f"\\n{'='*40}")
    print(f"  {dname}")
    print('='*40)
    print(classification_report(y_test, y_pred, target_names=['Low','Medium','High']))

plt.suptitle('Confusion Matrices: Patient Risk Stratification', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Summary Table ──────────────────────────────────────
summary_data = []
for mname, dname in zip(model_names, display_names):
    model = joblib.load(f'models/saved/{mname}.pkl')
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    summary_data.append({
        'Model': dname,
        'Accuracy': f"{report['accuracy']:.3f}",
        'F1-Macro': f"{report['macro avg']['f1-score']:.3f}",
        'Precision': f"{report['macro avg']['precision']:.3f}",
        'Recall': f"{report['macro avg']['recall']:.3f}",
    })

summary_df = pd.DataFrame(summary_data)
print("\\n" + summary_df.to_string(index=False))
summary_df.to_csv('outputs/reports/model_comparison.csv', index=False)`,
        tip: "In your paper and SOP, reference the F1-Macro score, not accuracy. In imbalanced clinical data, accuracy is misleading — knowing this shows you think like a researcher."
      }
    ]
  },
  {
    id: 4,
    week: "Week 2",
    title: "Explainability (SHAP)",
    subtitle: "Why the Model Decides",
    color: "#9B59B6",
    icon: "💡",
    steps: [
      {
        title: "Notebook 05: SHAP Explainability Analysis",
        description: "SHAP is the most important part of this project for your target programs. DKAI, MIND, Health AI all care about explainability. This is where your project earns its research credibility.",
        code: `import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load best model (XGBoost typically performs best)
model = joblib.load('models/saved/xgboost.pkl')
X_test = pd.read_csv('data/processed/X_test.csv')
feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                  'Insulin','BMI','DiabetesPedigree','Age',
                  'GlucoseBMI','AgeRisk']
X_test.columns = feature_names

# ── 1. Global SHAP: What features matter MOST overall? ──
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  class_names=['Low Risk', 'Medium Risk', 'High Risk'],
                  show=False)
plt.title('Global Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/shap_global_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 2. SHAP Beeswarm: Feature impact direction ──
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[2], X_test,  # index 2 = High Risk class
                  feature_names=feature_names, show=False)
plt.title('SHAP Beeswarm: High Risk Prediction Drivers', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/shap_beeswarm_highrisk.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 3. Individual Patient Explanation (Waterfall Plot) ──
# Pick a high-risk patient and explain their specific prediction
high_risk_indices = [i for i, v in enumerate(model.predict(X_test)) if v == 'High']
patient_idx = high_risk_indices[0]

explanation = shap.Explanation(
    values=shap_values[2][patient_idx],
    base_values=explainer.expected_value[2],
    data=X_test.iloc[patient_idx].values,
    feature_names=feature_names
)

plt.figure(figsize=(10, 5))
shap.waterfall_plot(explanation, show=False)
plt.title(f'Patient #{patient_idx}: Why HIGH RISK?', fontsize=13, fontweight='bold', color='#C0392B')
plt.tight_layout()
plt.savefig('outputs/figures/shap_patient_explanation.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 4. Generate Text Explanation for each patient ──
def explain_patient(patient_row, shap_vals, feature_names, top_n=3):
    """Generate human-readable explanation for clinical staff"""
    abs_shap = np.abs(shap_vals)
    top_indices = abs_shap.argsort()[-top_n:][::-1]
    
    reasons = []
    for idx in top_indices:
        feat = feature_names[idx]
        val = patient_row[idx]
        impact = "↑ increases" if shap_vals[idx] > 0 else "↓ decreases"
        reasons.append(f"  • {feat} = {val:.1f} {impact} risk")
    
    return "\\n".join(reasons)

# Print explanations for first 3 high-risk patients
print("\\n🔴 HIGH-RISK PATIENT EXPLANATIONS")
print("="*50)
for i, idx in enumerate(high_risk_indices[:3]):
    patient = X_test.iloc[idx].values
    explanation_text = explain_patient(patient, shap_values[2][idx], feature_names)
    print(f"\\nPatient #{idx}:")
    print(explanation_text)`,
        tip: "The text explanation function at the bottom is what makes this clinically usable. It translates ML output into something a nurse in a rural clinic can act on. Mention this in your SOP."
      }
    ]
  },
  {
    id: 5,
    week: "Week 2",
    title: "GitHub & README",
    subtitle: "Your Public Face",
    color: "#E74C3C",
    icon: "📦",
    steps: [
      {
        title: "Write a Research-Quality README",
        description: "Your README is the first thing an admissions committee member sees when they click your GitHub link. It must read like a mini research paper, not a tutorial.",
        code: `# BanglaHealth-AI: Predictive Patient Risk Stratification
## For Low-Resource Clinical Settings in Bangladesh

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Motivation

Bangladesh has **0.58 physicians per 1,000 people** (WHO, 2022), among the 
lowest in South Asia. Hospital management systems collect vast amounts of 
patient data — but this data sits static, recording history without predicting 
the future.

This project builds a machine learning pipeline that transforms clinical 
tabular data into **explainable, three-tier risk scores** (Low / Medium / High), 
designed for integration into existing hospital software via API.

## 📊 Dataset

- **Source**: UCI Pima Indians Diabetes Dataset (768 patients, 8 features)
- **Task**: 3-class patient risk stratification
- **Preprocessing**: Median imputation for biologically impossible zeros, 
  SMOTE oversampling for class balance, StandardScaler normalization

## 🤖 Models Compared

| Model | CV F1-Macro | Test Accuracy | 
|-------|-------------|---------------|
| Logistic Regression | 0.XXX ± 0.0XX | XX.X% |
| Random Forest | 0.XXX ± 0.0XX | XX.X% |
| **XGBoost** | **0.XXX ± 0.0XX** | **XX.X%** |

XGBoost achieved the highest performance. Full results: 
\`outputs/reports/model_comparison.csv\`

## 💡 Explainability (SHAP)

We use SHAP (SHapley Additive exPlanations) to generate:
- **Global feature importance**: Which clinical variables drive risk most?
- **Patient-level explanations**: Why is THIS patient flagged as high risk?
- **Human-readable alerts**: Text summaries for non-specialist clinical staff

Top risk predictors: Glucose level, BMI, Age (consistent with clinical literature)

## 📁 Project Structure
\`\`\`
banglahealth-ai/
├── notebooks/         # Step-by-step Jupyter analysis
├── src/               # Reusable Python modules  
├── models/saved/      # Trained model files (.pkl)
├── outputs/figures/   # All visualizations
└── paper/             # Research writeup (arXiv preprint)
\`\`\`

## 🚀 Quickstart
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/banglahealth-ai
cd banglahealth-ai
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
\`\`\`

## 📄 Paper
[arXiv preprint link — coming Week 8]

## 👤 Author
Md. Rakib | Junior Software Developer, Combosoft Ltd.  
BSc Computer Science, Daffodil International University  
*Applying for MSc Data Science / AI programs in France, 2025*`,
        tip: "Fill in the actual accuracy numbers after training. Fake placeholder numbers look unprofessional. Real numbers, even if lower than expected, are fine."
      }
    ]
  },
  {
    id: 6,
    week: "Week 6-8",
    title: "Research Paper",
    subtitle: "arXiv Preprint",
    color: "#F39C12",
    icon: "📝",
    steps: [
      {
        title: "4-Page Paper Structure (LaTeX / Overleaf)",
        description: "Submit this to arXiv. It takes 1-3 days to appear. This single link on your application transforms your profile from 'aspiring researcher' to 'active researcher'.",
        code: `PAPER TITLE:
"Towards Explainable AI-Powered Triage in Bangladesh: 
Comparative Analysis of Machine Learning Approaches 
for Patient Risk Stratification"

STRUCTURE (4 pages, IEEE two-column format):

1. ABSTRACT (150 words)
   - Problem: Low doctor-to-patient ratio in Bangladesh
   - Method: 3-model comparison + SHAP explainability
   - Key result: XGBoost achieves F1-macro of X.XX
   - Significance: Deployable as API for existing HMS

2. INTRODUCTION (0.5 page)
   - Bangladesh healthcare statistics (cite WHO)
   - The gap between data collection and intelligence
   - Contribution: first ML triage study framed for BD context

3. RELATED WORK (0.5 page)
   - Clinical risk stratification ML papers (3-4 citations)
   - SHAP in healthcare (1-2 citations)
   - Existing work on low-resource health AI

4. METHODOLOGY (1 page)
   - Dataset description + preprocessing decisions
   - Feature engineering rationale
   - Model selection justification
   - Evaluation metrics (why F1-macro, not accuracy)

5. RESULTS (1 page)
   - Model comparison table (cross-validation + test)
   - Confusion matrices
   - SHAP global importance figure
   - Patient-level explanation example

6. DISCUSSION & LIMITATIONS (0.5 page)
   - What the results mean clinically
   - Limitations: single dataset, no prospective validation
   - Future work: integrate with real HMS, multi-clinic study

7. CONCLUSION (0.25 page)
   - Summary + vision for Bangladesh health AI

Use Overleaf (free): overleaf.com
Template: IEEE Conference Paper Template`,
        tip: "You do NOT need journal acceptance to post on arXiv. Just submit the PDF. An arXiv link is permanent, free, and instantly credible."
      }
    ]
  }
];

const techStack = [
  { name: "Python 3.10", purpose: "Core language", color: "#3776AB" },
  { name: "pandas", purpose: "Data manipulation", color: "#150458" },
  { name: "scikit-learn", purpose: "LR + RF models", color: "#F7931E" },
  { name: "XGBoost", purpose: "Gradient boosting", color: "#189B3A" },
  { name: "SHAP", purpose: "Explainability", color: "#FF0052" },
  { name: "matplotlib", purpose: "Visualization", color: "#11557C" },
  { name: "seaborn", purpose: "Statistical plots", color: "#4C72B0" },
  { name: "imbalanced-learn", purpose: "SMOTE balancing", color: "#E67E22" },
  { name: "Jupyter", purpose: "Notebooks", color: "#F37626" },
  { name: "joblib", purpose: "Model saving", color: "#27AE60" },
];

export default function App() {
  const [activePhase, setActivePhase] = useState(0);
  const [activeStep, setActiveStep] = useState(0);
  const [showCode, setShowCode] = useState(true);

  const phase = phases[activePhase];
  const step = phase.steps[activeStep];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0A0E1A",
      color: "#E8EAF0",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0D1B2A 0%, #1A2744 50%, #0D1B2A 100%)",
        borderBottom: "1px solid #1E3A5F",
        padding: "24px 32px",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 10,
            background: "linear-gradient(135deg, #00C896, #0099FF)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20, flexShrink: 0
          }}>🏥</div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#FFFFFF", letterSpacing: "-0.3px" }}>
              BanglaHealth-AI Core
            </div>
            <div style={{ fontSize: 12, color: "#6B8CAE", marginTop: 2 }}>
              Predictive Patient Risk Stratification · Complete Build Guide
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            {["Week 1–2", "ML Project", "arXiv Ready"].map((tag, i) => (
              <span key={i} style={{
                background: "rgba(0,200,150,0.12)",
                border: "1px solid rgba(0,200,150,0.3)",
                color: "#00C896", borderRadius: 20,
                padding: "3px 10px", fontSize: 11
              }}>{tag}</span>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 32px" }}>
        {/* The Story */}
        <div style={{
          background: "linear-gradient(135deg, rgba(0,153,255,0.08), rgba(0,200,150,0.08))",
          border: "1px solid rgba(0,153,255,0.2)",
          borderRadius: 14, padding: "20px 24px", marginBottom: 28
        }}>
          <div style={{ fontSize: 11, color: "#0099FF", letterSpacing: 2, marginBottom: 8, fontWeight: 600 }}>
            THE PROJECT STORY
          </div>
          <p style={{ margin: 0, lineHeight: 1.7, color: "#C8D8E8", fontSize: 14 }}>
            One evening in Dhaka, you saw thousands of patient records scrolling by — and realized that data was being
            <span style={{ color: "#00C896", fontWeight: 600 }}> collected but never interrogated</span>.
            BanglaHealth-AI Core is the answer to that frustration: a rigorous, explainable ML pipeline that transforms
            raw clinical data into <span style={{ color: "#FF6B35", fontWeight: 600 }}>actionable risk scores</span>,
            complete with SHAP explanations that a rural clinic nurse can understand and act on.
            This project proves every claim in your SOP — with <span style={{ color: "#9B59B6", fontWeight: 600 }}>code, results, and a research paper</span>.
          </p>
        </div>

        {/* Phase Navigation */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          {phases.map((p, i) => (
            <button
              key={i}
              onClick={() => { setActivePhase(i); setActiveStep(0); }}
              style={{
                background: activePhase === i
                  ? `linear-gradient(135deg, ${p.color}22, ${p.color}44)`
                  : "rgba(255,255,255,0.04)",
                border: `1px solid ${activePhase === i ? p.color : "rgba(255,255,255,0.1)"}`,
                borderRadius: 10, padding: "10px 16px", cursor: "pointer",
                color: activePhase === i ? p.color : "#6B8CAE",
                fontSize: 12, fontWeight: activePhase === i ? 700 : 400,
                transition: "all 0.2s", fontFamily: "inherit",
                display: "flex", alignItems: "center", gap: 8
              }}
            >
              <span>{p.icon}</span>
              <span>
                <div style={{ fontSize: 10, opacity: 0.7 }}>{p.week}</div>
                <div>{p.title}</div>
              </span>
            </button>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 20 }}>
          {/* Step Sidebar */}
          <div>
            <div style={{
              background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12, overflow: "hidden"
            }}>
              <div style={{
                padding: "12px 16px",
                background: `linear-gradient(135deg, ${phase.color}18, ${phase.color}08)`,
                borderBottom: "1px solid rgba(255,255,255,0.06)",
                fontSize: 11, color: phase.color, fontWeight: 600, letterSpacing: 1.5
              }}>
                {phase.subtitle.toUpperCase()}
              </div>
              {phase.steps.map((s, i) => (
                <button
                  key={i}
                  onClick={() => setActiveStep(i)}
                  style={{
                    width: "100%", padding: "12px 16px",
                    background: activeStep === i ? `${phase.color}18` : "transparent",
                    borderLeft: `3px solid ${activeStep === i ? phase.color : "transparent"}`,
                    border: "none", borderBottom: "1px solid rgba(255,255,255,0.05)",
                    cursor: "pointer", textAlign: "left",
                    color: activeStep === i ? "#FFFFFF" : "#7A90A8",
                    fontSize: 12, lineHeight: 1.4, fontFamily: "inherit",
                    transition: "all 0.15s"
                  }}
                >
                  <div style={{
                    width: 20, height: 20, borderRadius: "50%",
                    background: activeStep === i ? phase.color : "rgba(255,255,255,0.1)",
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    fontSize: 10, color: "#fff", marginRight: 8, fontWeight: 700,
                    verticalAlign: "middle"
                  }}>{i + 1}</div>
                  {s.title.length > 35 ? s.title.substring(0, 35) + "..." : s.title}
                </button>
              ))}
            </div>
          </div>

          {/* Main Content */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Step Header */}
            <div style={{
              background: `linear-gradient(135deg, ${phase.color}12, ${phase.color}06)`,
              border: `1px solid ${phase.color}30`, borderRadius: 12, padding: "18px 22px"
            }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#FFFFFF", marginBottom: 6 }}>
                    {step.title}
                  </div>
                  <p style={{ margin: 0, fontSize: 13, color: "#8BA8C8", lineHeight: 1.6 }}>
                    {step.description}
                  </p>
                </div>
                <button
                  onClick={() => setShowCode(!showCode)}
                  style={{
                    background: showCode ? `${phase.color}22` : "rgba(255,255,255,0.06)",
                    border: `1px solid ${showCode ? phase.color : "rgba(255,255,255,0.1)"}`,
                    borderRadius: 8, padding: "6px 12px",
                    color: showCode ? phase.color : "#6B8CAE",
                    fontSize: 11, cursor: "pointer", fontFamily: "inherit",
                    flexShrink: 0, marginLeft: 16
                  }}
                >{showCode ? "Hide Code" : "Show Code"}</button>
              </div>
            </div>

            {/* Code Block */}
            {showCode && (
              <div style={{
                background: "#080C14", border: "1px solid #1A2A3D",
                borderRadius: 12, overflow: "hidden"
              }}>
                <div style={{
                  padding: "10px 18px", borderBottom: "1px solid #1A2A3D",
                  background: "#0D1520", display: "flex", alignItems: "center", gap: 8
                }}>
                  {["#FF5F57","#FFBD2E","#28C840"].map((c, i) => (
                    <div key={i} style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />
                  ))}
                  <span style={{ fontSize: 11, color: "#3D5A7A", marginLeft: 8 }}>
                    {activePhase <= 1 ? "terminal / folder structure" :
                     activePhase >= 4 ? "README.md / LaTeX" : "python · jupyter notebook"}
                  </span>
                </div>
                <pre style={{
                  margin: 0, padding: "18px", fontSize: 11.5, lineHeight: 1.65,
                  color: "#C8D8E8", overflowX: "auto", whiteSpace: "pre",
                  maxHeight: 380
                }}>
                  {step.code.split('\n').map((line, i) => {
                    const isComment = line.trim().startsWith('#') || line.trim().startsWith('//') || line.trim().startsWith('*') || line.trim().startsWith('STEP') || line.trim().startsWith('──');
                    const isKeyword = /^(import|from|def|class|if|for|else|return|print|plt|pd|np|df)\b/.test(line.trim());
                    const isString = line.includes('"""') || line.includes("'''");
                    return (
                      <span key={i} style={{
                        color: isComment ? "#4A7A6A" : isString ? "#C8A060" : "#C8D8E8",
                        display: "block"
                      }}>{line}</span>
                    );
                  })}
                </pre>
              </div>
            )}

            {/* Pro Tip */}
            <div style={{
              background: "rgba(243,156,18,0.08)", border: "1px solid rgba(243,156,18,0.25)",
              borderRadius: 10, padding: "12px 16px",
              display: "flex", gap: 10, alignItems: "flex-start"
            }}>
              <span style={{ fontSize: 16, flexShrink: 0 }}>💡</span>
              <div>
                <div style={{ fontSize: 10, color: "#F39C12", fontWeight: 700, letterSpacing: 1.5, marginBottom: 4 }}>
                  SOP / ADMISSION TIP
                </div>
                <div style={{ fontSize: 12, color: "#C8A860", lineHeight: 1.5 }}>{step.tip}</div>
              </div>
            </div>

            {/* Navigation */}
            <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
              <button
                onClick={() => {
                  if (activeStep > 0) setActiveStep(activeStep - 1);
                  else if (activePhase > 0) { setActivePhase(activePhase - 1); setActiveStep(phases[activePhase - 1].steps.length - 1); }
                }}
                disabled={activePhase === 0 && activeStep === 0}
                style={{
                  background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8, padding: "8px 16px", cursor: "pointer",
                  color: "#6B8CAE", fontSize: 12, fontFamily: "inherit",
                  opacity: (activePhase === 0 && activeStep === 0) ? 0.3 : 1
                }}>← Previous</button>
              <div style={{ fontSize: 11, color: "#3D5A7A", alignSelf: "center" }}>
                Phase {activePhase + 1}/{phases.length} · Step {activeStep + 1}/{phase.steps.length}
              </div>
              <button
                onClick={() => {
                  if (activeStep < phase.steps.length - 1) setActiveStep(activeStep + 1);
                  else if (activePhase < phases.length - 1) { setActivePhase(activePhase + 1); setActiveStep(0); }
                }}
                disabled={activePhase === phases.length - 1 && activeStep === phase.steps.length - 1}
                style={{
                  background: `linear-gradient(135deg, ${phase.color}30, ${phase.color}20)`,
                  border: `1px solid ${phase.color}50`,
                  borderRadius: 8, padding: "8px 16px", cursor: "pointer",
                  color: phase.color, fontSize: 12, fontFamily: "inherit",
                  opacity: (activePhase === phases.length - 1 && activeStep === phase.steps.length - 1) ? 0.3 : 1
                }}>Next →</button>
            </div>
          </div>
        </div>

        {/* Tech Stack */}
        <div style={{ marginTop: 28 }}>
          <div style={{ fontSize: 11, color: "#3D5A7A", letterSpacing: 2, marginBottom: 14, fontWeight: 600 }}>
            TECHNOLOGY STACK
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {techStack.map((t, i) => (
              <div key={i} style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8, padding: "8px 14px",
                display: "flex", alignItems: "center", gap: 8
              }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: t.color }} />
                <div>
                  <div style={{ fontSize: 12, color: "#E8EAF0", fontWeight: 600 }}>{t.name}</div>
                  <div style={{ fontSize: 10, color: "#4A6A8A" }}>{t.purpose}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Deliverables */}
        <div style={{
          marginTop: 24,
          display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12
        }}>
          {[
            { icon: "🐙", title: "GitHub Repo", desc: "Public, research-quality README with results + figures", color: "#0099FF" },
            { icon: "📊", title: "5 Notebooks", desc: "EDA → Preprocessing → Training → Evaluation → SHAP", color: "#00C896" },
            { icon: "📄", title: "arXiv Paper", desc: "4-page preprint — permanent proof of research activity", color: "#F39C12" },
          ].map((d, i) => (
            <div key={i} style={{
              background: `${d.color}10`, border: `1px solid ${d.color}25`,
              borderRadius: 12, padding: "16px",
            }}>
              <div style={{ fontSize: 22, marginBottom: 8 }}>{d.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#FFFFFF", marginBottom: 4 }}>{d.title}</div>
              <div style={{ fontSize: 11, color: "#6B8CAE", lineHeight: 1.5 }}>{d.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
