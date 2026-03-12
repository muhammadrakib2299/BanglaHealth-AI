"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["project"] == "BanglaHealth-AI"

    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()


class TestPredictEndpoints:
    """These tests require trained models.
    They will return 503 if models are not yet trained.
    """

    sample_diabetes = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }

    sample_heart = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    def test_diabetes_predict_returns_valid_response_or_503(self):
        response = client.post("/predict/diabetes", json=self.sample_diabetes)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert data["risk_level"] in ("Low", "Medium", "High")
            assert "confidence" in data

    def test_heart_predict_returns_valid_response_or_503(self):
        response = client.post("/predict/heart", json=self.sample_heart)
        assert response.status_code in (200, 503)

    def test_diabetes_invalid_input(self):
        response = client.post("/predict/diabetes", json={"Glucose": 100})
        assert response.status_code == 422  # Validation error
