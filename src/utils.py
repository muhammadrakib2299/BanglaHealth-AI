"""Shared utility functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def save_plot(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib figure to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUTS_DIR / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Create a confusion matrix heatmap."""
    if labels is None:
        labels = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def plot_model_comparison(comparison_df) -> plt.Figure:
    """Create a bar chart comparing model metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["F1-Macro", "Precision (macro)", "Recall (macro)"]
    x = np.arange(len(comparison_df))
    width = 0.25

    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            ax.bar(x + i * width, comparison_df[metric], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(comparison_df["Model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    return fig


def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs = [
        Path(__file__).resolve().parent.parent / d
        for d in ["data/raw", "data/processed", "models", "outputs", "notebooks"]
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
