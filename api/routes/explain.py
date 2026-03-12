"""Explanation endpoints — SHAP values and feature importance."""

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas import DiabetesInput, HeartInput
from src.explainer import (
    compute_shap_values,
    generate_clinical_alerts,
    get_feature_importance,
    get_shap_explainer,
)
from src.feature_engineering import add_diabetes_features, add_heart_features
from src.model import load_model, load_scaler
from src.preprocessing import RISK_LABEL_MAP

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post("/diabetes")
def explain_diabetes(
    data: DiabetesInput,
    model_name: str = Query("xgboost", description="Model to use"),
):
    """Get SHAP explanation for a diabetes prediction."""
    try:
        model = load_model(model_name, "diabetes")
        scaler = load_scaler("diabetes")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not trained yet: {e}")

    df = pd.DataFrame([data.model_dump()])
    df = add_diabetes_features(df)

    feature_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "GlucoseBMI", "AgeRisk", "InsulinLog", "BPCategory",
    ]
    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    # SHAP
    explainer = get_shap_explainer(model)
    shap_values = compute_shap_values(explainer, X_scaled)

    # Prediction
    risk_class = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    # Get SHAP values for the predicted class
    if len(shap_values.shape) == 3:
        shap_for_class = shap_values.values[0, :, risk_class]
    else:
        shap_for_class = shap_values.values[0]

    # Clinical alerts with real SHAP values
    alerts = generate_clinical_alerts(
        patient_data=data.model_dump(),
        shap_values_patient=shap_for_class,
        feature_names=feature_cols,
        dataset_type="diabetes",
    )

    return {
        "risk_level": RISK_LABEL_MAP[risk_class],
        "risk_class": risk_class,
        "confidence": {
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4),
        },
        "shap_values": {
            feat: round(float(val), 4)
            for feat, val in zip(feature_cols, shap_for_class)
        },
        "alerts": alerts,
    }


@router.post("/heart")
def explain_heart(
    data: HeartInput,
    model_name: str = Query("xgboost", description="Model to use"),
):
    """Get SHAP explanation for a heart disease prediction."""
    try:
        model = load_model(model_name, "heart")
        scaler = load_scaler("heart")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not trained yet: {e}")

    df = pd.DataFrame([data.model_dump()])
    df = add_heart_features(df)

    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        "BPxHR", "CholAge", "HeartStress",
    ]
    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    explainer = get_shap_explainer(model)
    shap_values = compute_shap_values(explainer, X_scaled)

    risk_class = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]

    if len(shap_values.shape) == 3:
        shap_for_class = shap_values.values[0, :, risk_class]
    else:
        shap_for_class = shap_values.values[0]

    alerts = generate_clinical_alerts(
        patient_data=data.model_dump(),
        shap_values_patient=shap_for_class,
        feature_names=feature_cols,
        dataset_type="heart",
    )

    return {
        "risk_level": RISK_LABEL_MAP[risk_class],
        "risk_class": risk_class,
        "confidence": {
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4),
        },
        "shap_values": {
            feat: round(float(val), 4)
            for feat, val in zip(feature_cols, shap_for_class)
        },
        "alerts": alerts,
    }
