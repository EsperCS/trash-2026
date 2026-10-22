#!/usr/bin/env python3
"""Train a TF-IDF + Logistic Regression baseline for jailbreak detection.

Pipeline
--------
1. Load ``train.csv`` and auto-detect text / label columns.
2. Encode string labels (``Benign`` / ``Jailbreak``) to 0 / 1 if needed.
3. Stratified 80/20 train-validation split.
4. Fit TF-IDF (word + char n-grams, max 50 000 features) → Logistic Regression.
5. Evaluate on the validation set and print metrics.
6. Generate predictions on ``test.csv`` → ``submissions/baseline_submission.csv``.
7. Persist the trained pipeline to ``models/baseline_tfidf_lr.pkl``.
8. Append results to ``reports/experiment_results.csv``.

Usage
-----
    python src/train_baseline.py --data_dir data --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.evaluate import compute_metrics, print_classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

TEXT_CANDIDATES: list[str] = ["prompt", "text", "content", "query", "message"]
LABEL_CANDIDATES: list[str] = ["label", "target", "is_jailbreak", "class"]

LABEL_MAP: dict[str, int] = {
    "benign": 0,
    "jailbreak": 1,
}


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name from *candidates* found in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _encode_labels(series: pd.Series) -> np.ndarray:
    """Convert label series to integer 0/1.

    Handles numeric (0/1), categorical, and string (``Benign`` / ``Jailbreak``) formats.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).values

    mapped = series.astype(str).str.lower().str.strip().map(LABEL_MAP)
    if mapped.isna().any():
        unknown = set(series[mapped.isna()].unique())
        raise ValueError(
            f"Unknown label values encountered: {unknown}. "
            f"Expected one of {list(LABEL_MAP.keys())}."
        )
    return mapped.astype(int).values


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline(
    max_features: int = 50_000,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> Pipeline:
    """Build a TF-IDF (word + char n-grams) → Logistic Regression pipeline.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size shared across word and char vectorizers.
    C : float
        Inverse regularisation strength for Logistic Regression.
    max_iter : int
        Maximum solver iterations.
    seed : int
        Random state.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        ("word", word_tfidf),
                        ("char", char_tfidf),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def train(
    data_dir: str = "data",
    seed: int = 42,
    C: float = 1.0,
    exp_name: str = "baseline-01",
) -> None:
    """End-to-end training, evaluation, and submission generation.

    Parameters
    ----------
    data_dir : str
        Directory containing ``train.csv`` and ``test.csv``.
    seed : int
        Random seed for reproducibility.
    C : float
        Regularization parameter for Logistic Regression.
    exp_name : str
        Identifier for the experiment.
    """
    t0 = time.perf_counter()
    data_path = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. Load training data
    # ------------------------------------------------------------------
    train_csv = data_path / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Training file not found: {train_csv}")

    logger.info("Loading training data from %s", train_csv)
    df_train = pd.read_csv(train_csv)
    logger.info("Training data shape: %s", df_train.shape)

    text_col = _detect_column(df_train, TEXT_CANDIDATES)
    label_col = _detect_column(df_train, LABEL_CANDIDATES)

    if text_col is None:
        raise ValueError(
            f"Could not auto-detect a text column. "
            f"Expected one of {TEXT_CANDIDATES}, got columns: {list(df_train.columns)}"
        )
    if label_col is None:
        raise ValueError(
            f"Could not auto-detect a label column. "
            f"Expected one of {LABEL_CANDIDATES}, got columns: {list(df_train.columns)}"
        )

    logger.info("Text column : %s", text_col)
    logger.info("Label column: %s", label_col)

    X = df_train[text_col].astype(str).values
    y = _encode_labels(df_train[label_col])

    # ------------------------------------------------------------------
    # 2. Stratified train / validation split
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    logger.info(
        "Split — train: %d, val: %d  (pos rate: train=%.3f, val=%.3f)",
        len(X_train),
        len(X_val),
        y_train.mean(),
        y_val.mean(),
    )

    # ------------------------------------------------------------------
    # 3. Build and fit pipeline
    # ------------------------------------------------------------------
    pipeline = build_pipeline(C=C, seed=seed)
    logger.info("Fitting TF-IDF + Logistic Regression pipeline (C=%.1f) …", C)
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline fitted.")

    # ------------------------------------------------------------------
    # 4. Evaluate on validation set
    # ------------------------------------------------------------------
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_val_prob)
    print_classification_report(y_val, y_val_prob)

    logger.info("Validation metrics: %s", metrics)

    # Confusion matrix (printed to console)
    from sklearn.metrics import confusion_matrix as cm_fn

    y_val_pred = (y_val_prob >= 0.5).astype(int)
    cm = cm_fn(y_val, y_val_pred)
    logger.info("Confusion matrix:\n%s", cm)

    # ------------------------------------------------------------------
    # 5. Generate test predictions
    # ------------------------------------------------------------------
    test_csv = data_path / "test.csv"
    if test_csv.exists():
        logger.info("Loading test data from %s", test_csv)
        df_test = pd.read_csv(test_csv)

        test_text_col = _detect_column(df_test, TEXT_CANDIDATES)
        if test_text_col is None:
            raise ValueError(
                f"Could not auto-detect text column in test set. "
                f"Columns: {list(df_test.columns)}"
            )

        X_test = df_test[test_text_col].astype(str).values
        test_probs = pipeline.predict_proba(X_test)[:, 1]

        # Determine id column in test set
        id_col = "Id" if "Id" in df_test.columns else ("id" if "id" in df_test.columns else df_test.columns[0])

        # Match columns with sample_submission.csv if present
        sample_sub_csv = data_path / "sample_submission.csv"
        if sample_sub_csv.exists():
            sample_cols = pd.read_csv(sample_sub_csv, nrows=1).columns.tolist()
            id_out, target_out = sample_cols[0], sample_cols[1]
        else:
            id_out, target_out = id_col, "TARGET"

        sub_dir = Path("submissions")
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_path = sub_dir / "baseline_submission.csv"

        submission = pd.DataFrame(
            {id_out: df_test[id_col].values, target_out: test_probs}
        )
        submission.to_csv(sub_path, index=False)
        logger.info("Submission saved to %s  (%d rows)", sub_path, len(submission))
    else:
        logger.warning("Test file not found (%s) — skipping submission.", test_csv)

    # ------------------------------------------------------------------
    # 6. Save model
    # ------------------------------------------------------------------
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_tfidf_lr.pkl"
    joblib.dump(pipeline, model_path)
    logger.info("Model saved to %s", model_path)

    # ------------------------------------------------------------------
    # 7. Append results to experiment log
    # ------------------------------------------------------------------
    results_dir = Path("reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "experiment_results.csv"

    result_row = {
        "experiment": exp_name,
        "model": f"tfidf_lr_C{C}",
        "seed": seed,
        **{f"val_{k}": round(v, 5) for k, v in metrics.items()},
    }
    result_df = pd.DataFrame([result_row])

    if results_path.exists():
        result_df.to_csv(results_path, mode="a", header=False, index=False)
    else:
        result_df.to_csv(results_path, index=False)

    logger.info("Results appended to %s", results_path)

    elapsed = time.perf_counter() - t0
    logger.info("Total runtime: %.2f seconds", elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing train.csv and test.csv (default: data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization strength for Logistic Regression (default: 1.0).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline-01",
        help="Experiment identifier (default: baseline-01).",
    )
    args = parser.parse_args()
    train(data_dir=args.data_dir, seed=args.seed, C=args.C, exp_name=args.exp_name)


if __name__ == "__main__":
    main()
