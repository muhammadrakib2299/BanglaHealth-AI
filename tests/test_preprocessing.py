"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    RISK_LABEL_MAP,
    apply_smote,
    create_risk_labels_diabetes,
    create_risk_labels_heart,
    replace_zero_with_median,
    scale_features,
    split_data,
)


@pytest.fixture
def sample_diabetes_df():
    """Create a sample diabetes-like DataFrame."""
    return pd.DataFrame({
        "Pregnancies": [6, 1, 8, 0, 5],
        "Glucose": [148, 0, 183, 90, 137],
        "BloodPressure": [72, 66, 0, 68, 40],
        "SkinThickness": [35, 29, 0, 23, 35],
        "Insulin": [0, 0, 0, 94, 168],
        "BMI": [33.6, 26.6, 0, 28.1, 43.1],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
        "Age": [50, 31, 32, 21, 33],
        "Outcome": [1, 0, 1, 0, 1],
    })


@pytest.fixture
def sample_heart_df():
    """Create a sample heart disease DataFrame."""
    return pd.DataFrame({
        "age": [63, 37, 41, 56, 57],
        "sex": [1, 1, 0, 1, 0],
        "cp": [3, 2, 1, 1, 0],
        "trestbps": [145, 130, 130, 120, 120],
        "chol": [233, 250, 204, 236, 354],
        "fbs": [1, 0, 0, 0, 0],
        "restecg": [0, 1, 0, 1, 1],
        "thalach": [150, 187, 172, 178, 163],
        "exang": [0, 0, 0, 0, 1],
        "oldpeak": [2.3, 3.5, 1.4, 0.8, 0.6],
        "slope": [0, 0, 2, 2, 2],
        "ca": [0, 0, 0, 0, 0],
        "thal": [1, 2, 2, 2, 2],
        "target": [1, 1, 0, 0, 0],
    })


class TestReplaceZeroWithMedian:
    def test_replaces_zeros(self, sample_diabetes_df):
        result = replace_zero_with_median(sample_diabetes_df, ["Glucose", "BMI"])
        assert (result["Glucose"] == 0).sum() == 0
        assert (result["BMI"] == 0).sum() == 0

    def test_preserves_nonzero(self, sample_diabetes_df):
        original_glucose_148 = sample_diabetes_df.loc[0, "Glucose"]
        result = replace_zero_with_median(sample_diabetes_df, ["Glucose"])
        assert result.loc[0, "Glucose"] == original_glucose_148

    def test_does_not_modify_original(self, sample_diabetes_df):
        original = sample_diabetes_df.copy()
        replace_zero_with_median(sample_diabetes_df, ["Glucose"])
        pd.testing.assert_frame_equal(sample_diabetes_df, original)


class TestRiskLabels:
    def test_diabetes_risk_labels_range(self, sample_diabetes_df):
        labels = create_risk_labels_diabetes(sample_diabetes_df)
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_diabetes_high_risk_for_diabetic(self, sample_diabetes_df):
        labels = create_risk_labels_diabetes(sample_diabetes_df)
        diabetic_mask = sample_diabetes_df["Outcome"] == 1
        assert (labels[diabetic_mask] == 2).all()

    def test_heart_risk_labels_range(self, sample_heart_df):
        labels = create_risk_labels_heart(sample_heart_df)
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_heart_high_risk_for_disease(self, sample_heart_df):
        labels = create_risk_labels_heart(sample_heart_df)
        disease_mask = sample_heart_df["target"] == 1
        assert (labels[disease_mask] == 2).all()

    def test_risk_label_map(self):
        assert RISK_LABEL_MAP == {0: "Low", 1: "Medium", 2: "High"}


class TestScaleFeatures:
    def test_output_shape(self, sample_diabetes_df):
        X = sample_diabetes_df[["Glucose", "BMI", "Age"]]
        X_train, X_test = X.iloc[:3], X.iloc[3:]
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

    def test_train_mean_near_zero(self, sample_diabetes_df):
        X = sample_diabetes_df[["Glucose", "BMI", "Age"]]
        X_train, X_test = X.iloc[:4], X.iloc[4:]
        X_train_s, _, _ = scale_features(X_train, X_test)
        assert np.abs(X_train_s.mean()).max() < 1e-10


class TestSplitData:
    def test_split_sizes(self, sample_diabetes_df):
        X = sample_diabetes_df[["Glucose", "BMI"]]
        y = pd.Series([0, 1, 0, 1, 0])
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4)
        assert len(X_train) == 3
        assert len(X_test) == 2


class TestSmote:
    def test_balances_classes(self):
        X = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        y = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        X_res, y_res = apply_smote(X, y)
        assert y_res.value_counts()[0] == y_res.value_counts()[1]
