"""Train, evaluate, compare, and persist ML models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Model definitions with hyperparameter search spaces
# ---------------------------------------------------------------------------

def _get_model_configs() -> dict:
    """Return model constructors and their hyperparameter search spaces."""
    configs = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "saga"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                random_state=42,
                eval_metric="mlogloss",
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        },
    }

    if LGBMClassifier is not None:
        configs["lightgbm"] = {
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9, -1],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "subsample": [0.8, 1.0],
            },
        }

    return configs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 42,
) -> tuple:
    """Train a model with RandomizedSearchCV.

    Args:
        model_name: One of 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'.
        X_train: Training features.
        y_train: Training labels.
        n_iter: Number of random search iterations.
        cv: Number of cross-validation folds.
        random_state: Random seed.

    Returns:
        Tuple of (best_model, best_params, cv_results).
    """
    configs = _get_model_configs()
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs)}")

    config = configs[model_name]
    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="f1_macro",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.cv_results_


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
) -> dict:
    """Train all available models and return results.

    Returns:
        Dict mapping model_name -> {model, best_params, cv_results}.
    """
    results = {}
    for name in _get_model_configs():
        model, params, cv_results = train_model(name, X_train, y_train, n_iter=n_iter)
        results[name] = {
            "model": model,
            "best_params": params,
            "cv_results": cv_results,
        }
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Evaluate a trained model on test data.

    Returns:
        Dict with predictions, classification_report, confusion_matrix,
        f1_macro, and roc_auc.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = None

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": report,
        "confusion_matrix": cm,
        "f1_macro": f1,
        "roc_auc": auc,
    }


def compare_models(results: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compare all trained models and return a summary DataFrame."""
    rows = []
    for name, res in results.items():
        eval_result = evaluate_model(res["model"], X_test, y_test)
        rows.append({
            "Model": name,
            "F1-Macro": round(eval_result["f1_macro"], 4),
            "ROC-AUC": round(eval_result["roc_auc"], 4) if eval_result["roc_auc"] else None,
            "Precision (macro)": round(eval_result["classification_report"]["macro avg"]["precision"], 4),
            "Recall (macro)": round(eval_result["classification_report"]["macro avg"]["recall"], 4),
        })
    return pd.DataFrame(rows).sort_values("F1-Macro", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, name: str, dataset: str) -> Path:
    """Save a trained model to disk.

    Args:
        model: Trained sklearn/xgboost/lightgbm model.
        name: Model name (e.g., 'xgboost').
        dataset: Dataset name (e.g., 'diabetes').

    Returns:
        Path to saved model file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"{dataset}_{name}.joblib"
    joblib.dump(model, filepath)
    return filepath


def load_model(name: str, dataset: str):
    """Load a saved model from disk."""
    filepath = MODELS_DIR / f"{dataset}_{name}.joblib"
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)


def save_scaler(scaler, dataset: str) -> Path:
    """Save a fitted StandardScaler."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"{dataset}_scaler.joblib"
    joblib.dump(scaler, filepath)
    return filepath


def load_scaler(dataset: str):
    """Load a saved StandardScaler."""
    filepath = MODELS_DIR / f"{dataset}_scaler.joblib"
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found: {filepath}")
    return joblib.load(filepath)
