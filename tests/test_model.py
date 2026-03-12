"""Tests for model training and persistence."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.model import (
    evaluate_model,
    save_model,
    load_model,
    train_model,
)


@pytest.fixture
def classification_data():
    """Generate synthetic 3-class classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_classes=3,
        n_informative=6,
        n_redundant=1,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(8)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


class TestTrainModel:
    def test_logistic_regression(self, classification_data):
        X, y = classification_data
        model, params, cv_results = train_model(
            "logistic_regression", X, y, n_iter=3, cv=3
        )
        assert model is not None
        assert hasattr(model, "predict")
        assert isinstance(params, dict)

    def test_random_forest(self, classification_data):
        X, y = classification_data
        model, params, _ = train_model(
            "random_forest", X, y, n_iter=3, cv=3
        )
        assert model is not None
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_xgboost(self, classification_data):
        X, y = classification_data
        model, params, _ = train_model(
            "xgboost", X, y, n_iter=3, cv=3
        )
        assert model is not None

    def test_unknown_model_raises(self, classification_data):
        X, y = classification_data
        with pytest.raises(ValueError, match="Unknown model"):
            train_model("invalid_model", X, y)


class TestEvaluateModel:
    def test_returns_expected_keys(self, classification_data):
        X, y = classification_data
        model, _, _ = train_model("logistic_regression", X, y, n_iter=2, cv=2)
        result = evaluate_model(model, X, y)

        assert "y_pred" in result
        assert "y_proba" in result
        assert "classification_report" in result
        assert "confusion_matrix" in result
        assert "f1_macro" in result
        assert "roc_auc" in result

    def test_f1_macro_range(self, classification_data):
        X, y = classification_data
        model, _, _ = train_model("xgboost", X, y, n_iter=2, cv=2)
        result = evaluate_model(model, X, y)
        assert 0 <= result["f1_macro"] <= 1


class TestModelPersistence:
    def test_save_and_load(self, classification_data, tmp_path, monkeypatch):
        X, y = classification_data
        model, _, _ = train_model("logistic_regression", X, y, n_iter=2, cv=2)

        monkeypatch.setattr("src.model.MODELS_DIR", tmp_path)
        save_model(model, "logistic_regression", "test")
        loaded = load_model("logistic_regression", "test")

        original_preds = model.predict(X[:5])
        loaded_preds = loaded.predict(X[:5])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.model.MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent", "test")
