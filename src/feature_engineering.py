"""Create derived clinical features from raw patient data."""

import numpy as np
import pandas as pd


def add_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features for the diabetes dataset.

    New features:
        GlucoseBMI:  Glucose * BMI — captures metabolic syndrome interaction.
                     High glucose AND high BMI together is more dangerous.
        AgeRisk:     Age / BMI — older patients with lower BMI may still
                     carry age-related risk.
        InsulinLog:  log(1 + Insulin) — Insulin is heavily right-skewed,
                     log transform normalizes the distribution.
        BPCategory:  Binned BloodPressure per AHA guidelines.
                     0 = Normal (<120), 1 = Elevated (120-129), 2 = High (130+).
    """
    df = df.copy()
    df["GlucoseBMI"] = df["Glucose"] * df["BMI"]
    df["AgeRisk"] = df["Age"] / df["BMI"].replace(0, np.nan)
    df["InsulinLog"] = np.log1p(df["Insulin"])
    df["BPCategory"] = pd.cut(
        df["BloodPressure"],
        bins=[0, 120, 130, 300],
        labels=[0, 1, 2],
        right=False,
    ).astype(float)
    return df


def add_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features for the heart disease dataset.

    New features:
        BPxHR:       RestBP * MaxHR — cardiovascular stress indicator.
                     High resting BP combined with high max HR suggests strain.
        CholHDL:     Cholesterol / Age — age-adjusted cholesterol risk.
        HeartStress: MaxHR / Age — exercise tolerance relative to age.
                     Lower values indicate reduced cardiac capacity.
    """
    df = df.copy()
    df["BPxHR"] = df["trestbps"] * df["thalach"]
    df["CholAge"] = df["chol"] / df["age"].replace(0, np.nan)
    df["HeartStress"] = df["thalach"] / df["age"].replace(0, np.nan)
    return df


def get_feature_columns_diabetes() -> list[str]:
    """Return feature column names for diabetes prediction (excluding target)."""
    return [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory",
    ]


def get_feature_columns_heart() -> list[str]:
    """Return feature column names for heart disease prediction (excluding target)."""
    return [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        "BPxHR", "CholAge", "HeartStress",
    ]
