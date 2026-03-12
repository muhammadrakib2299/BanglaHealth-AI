"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_diabetes_features,
    add_heart_features,
    get_feature_columns_diabetes,
    get_feature_columns_heart,
)


@pytest.fixture
def diabetes_df():
    return pd.DataFrame({
        "Pregnancies": [6],
        "Glucose": [148],
        "BloodPressure": [72],
        "SkinThickness": [35],
        "Insulin": [100],
        "BMI": [33.6],
        "DiabetesPedigreeFunction": [0.627],
        "Age": [50],
    })


@pytest.fixture
def heart_df():
    return pd.DataFrame({
        "age": [63],
        "sex": [1],
        "cp": [3],
        "trestbps": [145],
        "chol": [233],
        "fbs": [1],
        "restecg": [0],
        "thalach": [150],
        "exang": [0],
        "oldpeak": [2.3],
        "slope": [0],
        "ca": [0],
        "thal": [1],
    })


class TestDiabetesFeatures:
    def test_adds_glucose_bmi(self, diabetes_df):
        result = add_diabetes_features(diabetes_df)
        assert "GlucoseBMI" in result.columns
        expected = 148 * 33.6
        assert result["GlucoseBMI"].iloc[0] == pytest.approx(expected)

    def test_adds_age_risk(self, diabetes_df):
        result = add_diabetes_features(diabetes_df)
        assert "AgeRisk" in result.columns
        expected = 50 / 33.6
        assert result["AgeRisk"].iloc[0] == pytest.approx(expected)

    def test_adds_insulin_log(self, diabetes_df):
        result = add_diabetes_features(diabetes_df)
        assert "InsulinLog" in result.columns
        expected = np.log1p(100)
        assert result["InsulinLog"].iloc[0] == pytest.approx(expected)

    def test_adds_bp_category(self, diabetes_df):
        result = add_diabetes_features(diabetes_df)
        assert "BPCategory" in result.columns
        # BP=72 should be Normal (0)
        assert result["BPCategory"].iloc[0] == 0

    def test_does_not_modify_original(self, diabetes_df):
        original = diabetes_df.copy()
        add_diabetes_features(diabetes_df)
        pd.testing.assert_frame_equal(diabetes_df, original)

    def test_feature_columns_list(self):
        cols = get_feature_columns_diabetes()
        assert "GlucoseBMI" in cols
        assert "AgeRisk" in cols
        assert len(cols) == 12


class TestHeartFeatures:
    def test_adds_bpxhr(self, heart_df):
        result = add_heart_features(heart_df)
        assert "BPxHR" in result.columns
        expected = 145 * 150
        assert result["BPxHR"].iloc[0] == expected

    def test_adds_chol_age(self, heart_df):
        result = add_heart_features(heart_df)
        assert "CholAge" in result.columns
        expected = 233 / 63
        assert result["CholAge"].iloc[0] == pytest.approx(expected)

    def test_adds_heart_stress(self, heart_df):
        result = add_heart_features(heart_df)
        assert "HeartStress" in result.columns
        expected = 150 / 63
        assert result["HeartStress"].iloc[0] == pytest.approx(expected)

    def test_feature_columns_list(self):
        cols = get_feature_columns_heart()
        assert "BPxHR" in cols
        assert len(cols) == 16
