"""Data cleaning, scaling, splitting, and SMOTE resampling."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Columns where 0 is biologically impossible (missing value indicator)
DIABETES_ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def replace_zero_with_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace biologically impossible zero values with column medians.

    In the Pima Diabetes dataset, 0 in Glucose, BloodPressure, BMI etc.
    represents missing data, not actual measurements.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median_val)
    return df


def create_risk_labels_diabetes(df: pd.DataFrame) -> pd.Series:
    """Convert binary diabetes outcome to 3-class risk labels.

    Risk levels:
        0 (Low):    No diabetes + Glucose < 120 + BMI < 30
        1 (Medium): No diabetes + elevated features (borderline)
        2 (High):   Diabetic
    """
    conditions = [
        (df["Outcome"] == 1),
        (df["Outcome"] == 0) & ((df["Glucose"] >= 120) | (df["BMI"] >= 30)),
        (df["Outcome"] == 0) & (df["Glucose"] < 120) & (df["BMI"] < 30),
    ]
    labels = [2, 1, 0]  # High, Medium, Low
    return pd.Series(np.select(conditions, labels, default=0), name="RiskLevel")


def create_risk_labels_heart(df: pd.DataFrame) -> pd.Series:
    """Convert binary heart disease target to 3-class risk labels.

    Risk levels:
        0 (Low):    No disease + RestBP < 130 + Cholesterol < 240
        1 (Medium): No disease + elevated features
        2 (High):   Heart disease present
    """
    conditions = [
        (df["target"] == 1),
        (df["target"] == 0) & ((df["trestbps"] >= 130) | (df["chol"] >= 240)),
        (df["target"] == 0) & (df["trestbps"] < 130) & (df["chol"] < 240),
    ]
    labels = [2, 1, 0]
    return pd.Series(np.select(conditions, labels, default=0), name="RiskLevel")


RISK_LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Apply StandardScaler (fit on train, transform both).

    Returns:
        Tuple of (scaled_X_train, scaled_X_test, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance classes in training data only.

    Never apply SMOTE to test data — that would leak synthetic samples
    into evaluation.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_resampled, columns=X_train.columns),
        pd.Series(y_resampled, name=y_train.name),
    )
