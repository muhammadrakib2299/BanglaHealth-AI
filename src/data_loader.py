"""Load and validate clinical datasets."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Expected columns for each dataset
DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_diabetes(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset.

    Args:
        filepath: Path to CSV. Defaults to data/raw/diabetes.csv.

    Returns:
        DataFrame with validated columns.
    """
    if filepath is None:
        filepath = RAW_DIR / "diabetes.csv"
    df = pd.read_csv(filepath)
    _validate_columns(df, DIABETES_COLUMNS, "Diabetes")
    return df


def load_heart(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load the UCI Heart Disease (Cleveland) dataset.

    Args:
        filepath: Path to CSV. Defaults to data/raw/heart.csv.

    Returns:
        DataFrame with validated columns.
    """
    if filepath is None:
        filepath = RAW_DIR / "heart.csv"
    df = pd.read_csv(filepath)
    _validate_columns(df, HEART_COLUMNS, "Heart Disease")
    return df


def _validate_columns(
    df: pd.DataFrame, expected: list[str], dataset_name: str
) -> None:
    """Check that the DataFrame contains expected columns."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} dataset missing columns: {missing}. "
            f"Got: {list(df.columns)}"
        )


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return summary statistics for a dataset."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "describe": df.describe().to_dict(),
    }
