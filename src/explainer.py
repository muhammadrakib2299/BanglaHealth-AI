"""SHAP-based model explanations and clinical alert generation."""

import numpy as np
import pandas as pd
import shap

from .preprocessing import RISK_LABEL_MAP


# ---------------------------------------------------------------------------
# SHAP Explanations
# ---------------------------------------------------------------------------

def get_shap_explainer(model, X_background: pd.DataFrame | None = None):
    """Create the appropriate SHAP explainer for a model.

    Uses TreeExplainer for tree-based models (XGBoost, LightGBM, RandomForest)
    and LinearExplainer for Logistic Regression.
    """
    model_type = type(model).__name__

    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    elif model_type == "LogisticRegression":
        if X_background is None:
            raise ValueError("LinearExplainer requires background data (X_background)")
        return shap.LinearExplainer(model, X_background)
    else:
        if X_background is None:
            raise ValueError("KernelExplainer requires background data (X_background)")
        return shap.KernelExplainer(model.predict_proba, X_background)


def compute_shap_values(explainer, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for given data."""
    return explainer(X)


def get_feature_importance(shap_values: shap.Explanation) -> pd.DataFrame:
    """Get global feature importance from SHAP values.

    Returns:
        DataFrame with feature names and mean absolute SHAP values,
        sorted by importance.
    """
    if len(shap_values.shape) == 3:
        # Multi-class: average across classes
        vals = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        vals = np.abs(shap_values.values).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": shap_values.feature_names,
        "Importance": vals,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return importance


# ---------------------------------------------------------------------------
# Clinical Alerts
# ---------------------------------------------------------------------------

# Thresholds and clinical context for diabetes features
DIABETES_ALERTS = {
    "Glucose": {
        "high_threshold": 140,
        "high_message": "High glucose ({value:.0f} mg/dL) significantly increases diabetes risk. Consider HbA1c testing.",
        "normal_message": "Glucose level ({value:.0f} mg/dL) is within normal range.",
    },
    "BMI": {
        "high_threshold": 30,
        "high_message": "BMI of {value:.1f} indicates obesity, a major diabetes risk factor. Lifestyle intervention recommended.",
        "normal_message": "BMI ({value:.1f}) is within healthy range.",
    },
    "BloodPressure": {
        "high_threshold": 130,
        "high_message": "Elevated blood pressure ({value:.0f} mmHg) detected. Monitor for hypertension.",
        "normal_message": "Blood pressure ({value:.0f} mmHg) is within normal range.",
    },
    "Age": {
        "high_threshold": 45,
        "high_message": "Patient age ({value:.0f}) is a risk factor. Regular screening recommended.",
        "normal_message": "Patient age ({value:.0f}) — continue routine monitoring.",
    },
    "Insulin": {
        "high_threshold": 166,
        "high_message": "Elevated insulin ({value:.0f} mu U/ml) may indicate insulin resistance.",
        "normal_message": "Insulin level ({value:.0f} mu U/ml) is within normal range.",
    },
}

HEART_ALERTS = {
    "trestbps": {
        "high_threshold": 140,
        "high_message": "High resting blood pressure ({value:.0f} mmHg). Hypertension is a major cardiovascular risk factor.",
        "normal_message": "Resting blood pressure ({value:.0f} mmHg) is within normal range.",
    },
    "chol": {
        "high_threshold": 240,
        "high_message": "High cholesterol ({value:.0f} mg/dL). Consider lipid panel and statin therapy evaluation.",
        "normal_message": "Cholesterol ({value:.0f} mg/dL) is within acceptable range.",
    },
    "thalach": {
        "low_threshold": 120,
        "low_message": "Low maximum heart rate ({value:.0f} bpm) may indicate reduced cardiac capacity.",
        "normal_message": "Maximum heart rate ({value:.0f} bpm) is within expected range.",
    },
    "oldpeak": {
        "high_threshold": 2.0,
        "high_message": "ST depression of {value:.1f} suggests exercise-induced ischemia. Cardiology referral recommended.",
        "normal_message": "ST depression ({value:.1f}) is within normal limits.",
    },
}


def generate_clinical_alerts(
    patient_data: dict,
    shap_values_patient: np.ndarray,
    feature_names: list[str],
    dataset_type: str = "diabetes",
) -> list[dict]:
    """Generate human-readable clinical alerts for a patient prediction.

    Args:
        patient_data: Dict of feature_name -> value for one patient.
        shap_values_patient: SHAP values for this patient (1D array).
        feature_names: List of feature names matching shap_values order.
        dataset_type: 'diabetes' or 'heart'.

    Returns:
        List of alert dicts with keys: feature, value, shap_impact, severity, message.
    """
    alerts_config = DIABETES_ALERTS if dataset_type == "diabetes" else HEART_ALERTS
    alerts = []

    for i, feat in enumerate(feature_names):
        if feat not in alerts_config:
            continue

        config = alerts_config[feat]
        value = patient_data.get(feat, 0)
        shap_impact = float(shap_values_patient[i]) if i < len(shap_values_patient) else 0

        # Determine severity
        if "high_threshold" in config and value >= config["high_threshold"]:
            severity = "high"
            message = config["high_message"].format(value=value)
        elif "low_threshold" in config and value <= config["low_threshold"]:
            severity = "high"
            message = config["low_message"].format(value=value)
        else:
            severity = "normal"
            message = config["normal_message"].format(value=value)

        alerts.append({
            "feature": feat,
            "value": value,
            "shap_impact": round(shap_impact, 4),
            "severity": severity,
            "message": message,
        })

    # Sort: high severity first, then by absolute SHAP impact
    alerts.sort(key=lambda a: (a["severity"] != "high", -abs(a["shap_impact"])))
    return alerts


def format_prediction_summary(
    risk_level: int,
    probabilities: np.ndarray,
    alerts: list[dict],
) -> str:
    """Format a complete prediction summary as readable text.

    Args:
        risk_level: Predicted risk class (0, 1, or 2).
        probabilities: Class probabilities [low, medium, high].
        alerts: List of clinical alert dicts.

    Returns:
        Formatted string summary.
    """
    risk_name = RISK_LABEL_MAP[risk_level]
    lines = [
        f"Risk Level: {risk_name}",
        f"Confidence: Low={probabilities[0]:.1%}, Medium={probabilities[1]:.1%}, High={probabilities[2]:.1%}",
        "",
        "Clinical Findings:",
    ]

    for alert in alerts:
        icon = "[!]" if alert["severity"] == "high" else "[ok]"
        lines.append(f"  {icon} {alert['message']}")
        lines.append(f"      SHAP impact: {alert['shap_impact']:+.4f}")

    return "\n".join(lines)
