#!/usr/bin/env python3
"""Evaluation utilities for binary classification.

Provides functions for computing metrics, generating classification reports,
plotting ROC/confusion-matrix curves, and performing error analysis.

All functions are designed to be imported and used standalone.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary-classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    threshold : float, optional
        Decision threshold (default 0.5).

    Returns
    -------
    dict[str, float]
        Dictionary with ROC-AUC, PR-AUC, Precision, Recall, F1, Accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Print a sklearn classification report plus ROC-AUC / PR-AUC.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float, optional
        Decision threshold (default 0.5).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_prob, threshold)
    print("=" * 60)
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC  : {metrics['pr_auc']:.4f}")
    print("=" * 60)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Jailbreak"],
            digits=4,
        )
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot the Receiver Operating Characteristic curve.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    save_path : str or None
        If provided, save the figure to this path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot_roc_curve.")
        return

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    save_path : str or None
        If provided, save the figure to this path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping plot_confusion_matrix.")
        return

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Jailbreak"],
        yticklabels=["Benign", "Jailbreak"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(
    texts: list[str] | np.ndarray | pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify the worst false-positive and false-negative predictions.

    Parameters
    ----------
    texts : sequence of str
        Original text inputs.
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float, optional
        Decision threshold (default 0.5).
    top_n : int, optional
        Number of worst errors to return per category.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(false_positives_df, false_negatives_df)`` sorted by confidence
        of the erroneous prediction (descending).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "text": list(texts),
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    # False positives: predicted 1 but true 0 — sort by highest prob
    fp = (
        df[(df["y_pred"] == 1) & (df["y_true"] == 0)]
        .sort_values("y_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # False negatives: predicted 0 but true 1 — sort by lowest prob
    fn = (
        df[(df["y_pred"] == 0) & (df["y_true"] == 1)]
        .sort_values("y_prob", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    return fp, fn
