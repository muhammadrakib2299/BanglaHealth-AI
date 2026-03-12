"""Prediction endpoints."""

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from api.schemas import DiabetesInput, HeartInput, RiskPrediction
from src.explainer import generate_clinical_alerts
from src.feature_engineering import add_diabetes_features, add_heart_features
from src.model import load_model, load_scaler
from src.preprocessing import RISK_LABEL_MAP

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _predict_diabetes(data: DiabetesInput, model_name: str = "xgboost") -> RiskPrediction:
    """Run diabetes risk prediction pipeline."""
    # Load model and scaler
    model = load_model(model_name, "diabetes")
    scaler = load_scaler("diabetes")

    # Create DataFrame from input
    df = pd.DataFrame([data.model_dump()])

    # Feature engineering
    df = add_diabetes_features(df)

    # Select feature columns (must match training order)
    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory",
    ]
    X = df[feature_cols]

    # Scale
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    # Predict
    risk_class = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    # Generate clinical alerts
    alerts = generate_clinical_alerts(
        patient_data=data.model_dump(),
        shap_values_patient=probabilities,  # placeholder until SHAP is computed
        feature_names=feature_cols,
        dataset_type="diabetes",
    )

    return RiskPrediction(
        risk_level=RISK_LABEL_MAP[risk_class],
        risk_class=risk_class,
        confidence={
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4),
        },
        alerts=alerts,
    )


def _predict_heart(data: HeartInput, model_name: str = "xgboost") -> RiskPrediction:
    """Run heart disease risk prediction pipeline."""
    model = load_model(model_name, "heart")
    scaler = load_scaler("heart")

    df = pd.DataFrame([data.model_dump()])
    df = add_heart_features(df)

    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        "BPxHR", "CholAge", "HeartStress",
    ]
    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    risk_class = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    alerts = generate_clinical_alerts(
        patient_data=data.model_dump(),
        shap_values_patient=probabilities,
        feature_names=feature_cols,
        dataset_type="heart",
    )

    return RiskPrediction(
        risk_level=RISK_LABEL_MAP[risk_class],
        risk_class=risk_class,
        confidence={
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4),
        },
        alerts=alerts,
    )


@router.post("/diabetes", response_model=RiskPrediction)
def predict_diabetes(
    data: DiabetesInput,
    model_name: str = Query("xgboost", description="Model to use"),
):
    """Predict diabetes risk for a single patient."""
    try:
        return _predict_diabetes(data, model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not trained yet: {e}")


@router.post("/heart", response_model=RiskPrediction)
def predict_heart(
    data: HeartInput,
    model_name: str = Query("xgboost", description="Model to use"),
):
    """Predict heart disease risk for a single patient."""
    try:
        return _predict_heart(data, model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not trained yet: {e}")


@router.post("/diabetes/batch")
def predict_diabetes_batch(file: UploadFile = File(...)):
    """Predict diabetes risk for multiple patients from CSV."""
    try:
        df = pd.read_csv(file.file)
        results = []
        for _, row in df.iterrows():
            input_data = DiabetesInput(**row.to_dict())
            pred = _predict_diabetes(input_data)
            results.append(pred.model_dump())
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/heart/batch")
def predict_heart_batch(file: UploadFile = File(...)):
    """Predict heart disease risk for multiple patients from CSV."""
    try:
        df = pd.read_csv(file.file)
        results = []
        for _, row in df.iterrows():
            input_data = HeartInput(**row.to_dict())
            pred = _predict_heart(input_data)
            results.append(pred.model_dump())
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
