"""Fairness analysis — evaluate model performance across demographic groups."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_by_group(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    group_column: str,
    group_labels: dict | None = None,
) -> pd.DataFrame:
    """Evaluate model performance separately for each demographic group.

    Args:
        model: Trained model with predict method.
        X: Test features.
        y: True labels.
        group_column: Column name to group by (e.g., 'Age_Group', 'sex').
        group_labels: Optional mapping from group values to readable names.

    Returns:
        DataFrame with per-group metrics.
    """
    y_pred = model.predict(X)
    groups = X[group_column].unique()

    rows = []
    for group in sorted(groups):
        mask = X[group_column] == group
        n = mask.sum()

        if n < 5:
            continue

        y_true_g = y[mask]
        y_pred_g = y_pred[mask]

        label = group_labels.get(group, str(group)) if group_labels else str(group)

        rows.append({
            "Group": label,
            "N": n,
            "F1-Macro": round(f1_score(y_true_g, y_pred_g, average="macro", zero_division=0), 4),
            "Precision": round(precision_score(y_true_g, y_pred_g, average="macro", zero_division=0), 4),
            "Recall": round(recall_score(y_true_g, y_pred_g, average="macro", zero_division=0), 4),
        })

    return pd.DataFrame(rows)


def create_age_groups(ages: pd.Series) -> pd.Series:
    """Bin ages into clinical age groups."""
    return pd.cut(
        ages,
        bins=[0, 30, 45, 60, 120],
        labels=["Young (<=30)", "Middle (31-45)", "Senior (46-60)", "Elderly (60+)"],
    )


def create_bmi_groups(bmi: pd.Series) -> pd.Series:
    """Bin BMI into WHO categories."""
    return pd.cut(
        bmi,
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )


def compute_fairness_gap(group_metrics: pd.DataFrame, metric: str = "F1-Macro") -> dict:
    """Compute the fairness gap — difference between best and worst performing groups.

    A gap > 0.10 is generally considered concerning.
    """
    best = group_metrics[metric].max()
    worst = group_metrics[metric].min()
    gap = best - worst

    return {
        "metric": metric,
        "best_group": group_metrics.loc[group_metrics[metric].idxmax(), "Group"],
        "best_score": best,
        "worst_group": group_metrics.loc[group_metrics[metric].idxmin(), "Group"],
        "worst_score": worst,
        "gap": round(gap, 4),
        "is_fair": gap <= 0.10,
    }


def generate_fairness_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_type: str = "diabetes",
) -> dict:
    """Generate a complete fairness report for a model.

    Evaluates performance across age groups and (for heart disease) sex.

    Returns:
        Dict with group metrics and fairness gaps.
    """
    report = {}

    # Age-based fairness
    if dataset_type == "diabetes":
        X_eval = X.copy()
        X_eval["Age_Group"] = create_age_groups(X_eval["Age"] if "Age" in X_eval.columns else X_eval.iloc[:, 7])
    else:
        X_eval = X.copy()
        X_eval["Age_Group"] = create_age_groups(X_eval["age"] if "age" in X_eval.columns else X_eval.iloc[:, 0])

    age_metrics = evaluate_by_group(model, X_eval, y, "Age_Group")
    report["age_groups"] = age_metrics
    report["age_fairness"] = compute_fairness_gap(age_metrics)

    # Sex-based fairness (heart disease only — diabetes dataset is all female)
    if dataset_type == "heart":
        sex_col = "sex" if "sex" in X.columns else X.columns[1]
        sex_metrics = evaluate_by_group(
            model, X, y, sex_col,
            group_labels={0: "Female", 1: "Male"},
        )
        report["sex_groups"] = sex_metrics
        report["sex_fairness"] = compute_fairness_gap(sex_metrics)

    # BMI-based fairness (diabetes only)
    if dataset_type == "diabetes" and "BMI" in X.columns:
        X_eval2 = X.copy()
        X_eval2["BMI_Group"] = create_bmi_groups(X_eval2["BMI"])
        bmi_metrics = evaluate_by_group(model, X_eval2, y, "BMI_Group")
        report["bmi_groups"] = bmi_metrics
        report["bmi_fairness"] = compute_fairness_gap(bmi_metrics)

    return report
