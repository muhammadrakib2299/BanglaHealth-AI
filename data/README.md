# Datasets

## Sources

### 1. Pima Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **URL:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Samples:** 768
- **Features:** 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
- **Target:** Outcome (0 = No Diabetes, 1 = Diabetes)
- **License:** CC0 Public Domain

### 2. UCI Heart Disease Dataset (Cleveland)
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Samples:** 303
- **Features:** 13 (Age, Sex, ChestPainType, RestBP, Cholesterol, FastingBS, RestECG, MaxHR, ExerciseAngina, Oldpeak, Slope, CA, Thal)
- **Target:** num (0 = No Disease, 1-4 = Disease severity)
- **License:** CC BY 4.0

## Directory Structure

```
data/
├── raw/          # Original downloaded CSVs (do not modify)
├── processed/    # Cleaned, engineered, split datasets
└── README.md     # This file
```

## Risk Label Mapping

Binary targets are converted to 3-class risk labels:

| Risk Level | Diabetes Criteria | Heart Disease Criteria |
|------------|-------------------|----------------------|
| Low        | No diabetes + Glucose < 120 + BMI < 30 | No disease + RestBP < 130 + Cholesterol < 240 |
| Medium     | No diabetes + elevated features | No disease + elevated features |
| High       | Diabetic | Heart disease present |
