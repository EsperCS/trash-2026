#!/usr/bin/env python3
"""Fine-tune a DistilBERT model for binary text classification (Jailbreak detection).

This script provides a complete training pipeline including:
- Auto-detection of text/label columns from CSV data
- Stratified train/validation splitting
- HuggingFace Trainer-based fine-tuning with early stopping
- Comprehensive evaluation (ROC-AUC, PR-AUC, Precision, Recall, F1)
- Error analysis of worst false positives/negatives
- Test set prediction and submission CSV generation
- Experiment result logging

Usage:
    python src/train_transformer.py --data_dir data --epochs 5 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEXT_COLUMN_CANDIDATES = ("prompt", "text", "content")
LABEL_COLUMN_CANDIDATES = ("label", "target")
LABEL_MAP = {"jailbreak": 1, "benign": 0}


# ============================================================================
# Utility helpers
# ============================================================================


def set_all_seeds(seed: int) -> None:
    """Set seeds for full reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("All random seeds set to %d", seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve the compute device with auto-detection for MPS / CUDA / CPU.

    Args:
        requested: One of ``'auto'``, ``'mps'``, ``'cuda'``, or ``'cpu'``.

    Returns:
        A :class:`torch.device` instance.
    """
    if requested == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
    logger.info("Using device: %s", device)
    return device


def _detect_column(df: pd.DataFrame, candidates: Tuple[str, ...], role: str) -> str:
    """Auto-detect a column name from a DataFrame by trying candidates.

    Args:
        df: Input DataFrame.
        candidates: Ordered tuple of candidate column names.
        role: Human-readable role name for logging (e.g. ``'text'``).

    Returns:
        The first matching column name.

    Raises:
        ValueError: If none of the candidates are found.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_cols:
            col = lower_cols[candidate.lower()]
            logger.info("Auto-detected %s column: '%s'", role, col)
            return col
    raise ValueError(
        f"Could not auto-detect {role} column. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def encode_labels(series: pd.Series) -> pd.Series:
    """Convert string labels to integers using ``LABEL_MAP`` if needed.

    If labels are already numeric (0/1), they are returned as-is.

    Args:
        series: Pandas Series of labels.

    Returns:
        Integer-encoded label Series.
    """
    if series.dtype == object or series.dtype.name == "category":
        mapped = series.str.strip().str.lower().map(LABEL_MAP)
        unmapped = mapped.isna()
        if unmapped.any():
            bad_vals = series[unmapped].unique().tolist()
            raise ValueError(
                f"Unknown label values: {bad_vals}. Expected one of {list(LABEL_MAP.keys())}"
            )
        logger.info("Converted string labels to integers using map: %s", LABEL_MAP)
        return mapped.astype(int)
    return series.astype(int)


# ============================================================================
# Data pipeline
# ============================================================================


def load_and_split_data(
    data_dir: str,
    val_split: float,
    seed: int,
) -> Tuple[Dataset, Dataset, str, str]:
    """Load ``train.csv``, detect columns, encode labels, and stratified-split.

    Args:
        data_dir: Directory containing ``train.csv``.
        val_split: Fraction of data to use for validation.
        seed: Random seed for the split.

    Returns:
        A tuple of ``(train_dataset, val_dataset, text_col, label_col)``.
    """
    csv_path = os.path.join(data_dir, "train.csv")
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    text_col = _detect_column(df, TEXT_COLUMN_CANDIDATES, "text")
    label_col = _detect_column(df, LABEL_COLUMN_CANDIDATES, "label")

    # Drop rows with missing text or labels
    before = len(df)
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing text/label values", before - len(df))

    # Encode labels
    df["label"] = encode_labels(df[label_col])
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info("Train size: %d | Val size: %d", len(train_df), len(val_df))

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]])
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]])

    return train_ds, val_ds, text_col, label_col


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    text_col: str,
    max_length: int,
) -> Dataset:
    """Tokenize a HuggingFace Dataset using batch ``map()``.

    Args:
        dataset: HuggingFace Dataset with a text column.
        tokenizer: Pre-trained tokenizer.
        text_col: Name of the text column to tokenize.
        max_length: Maximum sequence length.

    Returns:
        Tokenized Dataset with ``input_ids``, ``attention_mask``, and ``label``.
    """

    def _tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize_fn, batched=True, desc="Tokenizing")
    # Keep only the columns the model needs
    keep_cols = {"input_ids", "attention_mask", "label"}
    remove_cols = [c for c in tokenized.column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)
    tokenized.set_format("torch")
    return tokenized


# ============================================================================
# Metrics
# ============================================================================


def compute_metrics_fn(eval_pred: Any) -> Dict[str, float]:
    """Compute metrics for HuggingFace Trainer evaluation.

    Computes accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.

    Args:
        eval_pred: ``EvalPrediction`` namedtuple from the Trainer.

    Returns:
        Dictionary of metric name → value.
    """
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    # ROC-AUC (requires both classes present)
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
        metrics["pr_auc"] = float(auc(rec_curve, prec_curve))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


# ============================================================================
# Evaluation & error analysis
# ============================================================================


def evaluate_model(
    trainer: Trainer,
    val_dataset: Dataset,
    raw_val_texts: list[str],
) -> Dict[str, float]:
    """Run full evaluation on the validation set and print detailed metrics.

    Args:
        trainer: Trained HuggingFace Trainer instance.
        val_dataset: Tokenized validation dataset.
        raw_val_texts: Raw text strings for error analysis.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION ON VALIDATION SET")
    logger.info("=" * 60)

    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)

    # Detailed classification report
    report = classification_report(
        labels, preds, target_names=["Benign (0)", "Jailbreak (1)"], digits=4
    )
    logger.info("\nClassification Report:\n%s", report)

    # Compute core metrics
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        prec_curve, rec_curve, _ = precision_recall_curve(labels, probs)
        metrics["pr_auc"] = float(auc(rec_curve, prec_curve))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    logger.info("Metrics summary:")
    for k, v in metrics.items():
        logger.info("  %-12s : %.4f", k, v)

    # ----- Error analysis -----
    _error_analysis(labels, preds, probs, raw_val_texts)

    return metrics


def _error_analysis(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    texts: list[str],
    top_k: int = 5,
) -> None:
    """Print the worst false-positive and false-negative examples.

    Args:
        labels: Ground-truth labels.
        preds: Predicted labels.
        probs: Predicted probabilities for the positive class.
        texts: Raw text inputs.
        top_k: Number of examples to show per category.
    """
    fp_mask = (preds == 1) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)

    logger.info("-" * 60)
    logger.info("ERROR ANALYSIS (top %d per category)", top_k)
    logger.info("-" * 60)

    # Worst false positives: benign classified as jailbreak (highest prob)
    if fp_mask.any():
        fp_indices = np.where(fp_mask)[0]
        fp_sorted = fp_indices[np.argsort(-probs[fp_indices])][:top_k]
        logger.info("\n>>> Worst FALSE POSITIVES (benign predicted as jailbreak):")
        for rank, idx in enumerate(fp_sorted, 1):
            snippet = texts[idx][:120].replace("\n", " ")
            logger.info("  %d. [prob=%.4f] %s...", rank, probs[idx], snippet)
    else:
        logger.info("No false positives found.")

    # Worst false negatives: jailbreak classified as benign (lowest prob)
    if fn_mask.any():
        fn_indices = np.where(fn_mask)[0]
        fn_sorted = fn_indices[np.argsort(probs[fn_indices])][:top_k]
        logger.info("\n>>> Worst FALSE NEGATIVES (jailbreak predicted as benign):")
        for rank, idx in enumerate(fn_sorted, 1):
            snippet = texts[idx][:120].replace("\n", " ")
            logger.info("  %d. [prob=%.4f] %s...", rank, probs[idx], snippet)
    else:
        logger.info("No false negatives found.")


# ============================================================================
# Test prediction & submission
# ============================================================================


def generate_test_predictions(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    data_dir: str,
    text_col: str,
    max_length: int,
    output_dir: str,
) -> Optional[str]:
    """Load test.csv, generate predictions, and save submission CSV.

    Args:
        trainer: Trained Trainer instance.
        tokenizer: Tokenizer used during training.
        data_dir: Directory containing ``test.csv``.
        text_col: Name of the text column.
        max_length: Maximum sequence length.
        output_dir: Root project output directory (submissions saved under ``submissions/``).

    Returns:
        Path to the saved submission CSV, or ``None`` if test.csv not found.
    """
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        logger.warning("test.csv not found at %s – skipping test predictions.", test_path)
        return None

    logger.info("Loading test data from %s", test_path)
    test_df = pd.read_csv(test_path)
    logger.info("Test set size: %d", len(test_df))

    # Auto-detect text column in test set
    test_text_col = text_col
    if test_text_col not in test_df.columns:
        test_text_col = _detect_column(test_df, TEXT_COLUMN_CANDIDATES, "text (test)")

    # Detect or create ID column
    if "id" in test_df.columns:
        ids = test_df["id"].tolist()
    elif "Id" in test_df.columns:
        ids = test_df["Id"].tolist()
    else:
        ids = list(range(len(test_df)))
        logger.info("No 'id' column found in test set; using row indices.")

    # Fill any missing text with empty string
    test_df[test_text_col] = test_df[test_text_col].fillna("")

    test_ds = Dataset.from_pandas(test_df[[test_text_col]])

    def _tok(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples[test_text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    test_ds = test_ds.map(_tok, batched=True, desc="Tokenizing test set")
    remove_cols = [c for c in test_ds.column_names if c not in {"input_ids", "attention_mask"}]
    test_ds = test_ds.remove_columns(remove_cols)
    test_ds.set_format("torch")

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Map predictions back to string labels
    inv_label_map = {v: k.capitalize() for k, v in LABEL_MAP.items()}
    pred_labels = [inv_label_map[p] for p in preds]

    # Save submission
    submissions_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")), "submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    submission_path = os.path.join(submissions_dir, "transformer_submission.csv")

    sub_df = pd.DataFrame({"id": ids, "target": pred_labels})
    sub_df.to_csv(submission_path, index=False)
    logger.info("Submission saved to %s (%d rows)", submission_path, len(sub_df))

    return submission_path


# ============================================================================
# Results logging
# ============================================================================


def log_experiment_results(
    metrics: Dict[str, float],
    args: argparse.Namespace,
    elapsed: float,
    output_dir: str,
) -> None:
    """Append experiment results to ``reports/experiment_results.csv`` and save run config JSON.

    Args:
        metrics: Evaluation metrics dictionary.
        args: Parsed command-line arguments.
        elapsed: Training wall-clock time in seconds.
        output_dir: Model output directory.
    """
    # ---- Save run config JSON ----
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "run_config.json")
    run_config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "val_split": args.val_split,
        "seed": args.seed,
        "device": args.device,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    logger.info("Run config saved to %s", config_path)

    # ---- Append to experiment_results.csv ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    results_path = os.path.join(reports_dir, "experiment_results.csv")

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "max_length": args.max_length,
        "elapsed_s": round(elapsed, 2),
        **{f"val_{k}": round(v, 4) for k, v in metrics.items()},
    }
    results_df = pd.DataFrame([row])

    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        results_df = pd.concat([existing, results_df], ignore_index=True)

    results_df.to_csv(results_path, index=False)
    logger.info("Experiment results appended to %s", results_path)


# ============================================================================
# Main training pipeline
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for Jailbreak detection (binary classification)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name or path (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing train.csv and test.csv (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/transformer",
        help="Directory to save model checkpoints (default: models/transformer)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for train and eval (default: 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token sequence length (default: 256)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Proportion of training for LR warmup (default: 0.1)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: orchestrates the full fine-tuning pipeline."""
    args = parse_args()
    set_all_seeds(args.seed)
    device = resolve_device(args.device)

    logger.info("=" * 60)
    logger.info("TRANSFORMER FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    logger.info("Config: %s", json.dumps(vars(args), indent=2))

    # ------------------------------------------------------------------
    # 1. Data loading & splitting
    # ------------------------------------------------------------------
    train_ds, val_ds, text_col, label_col = load_and_split_data(
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Keep raw texts for error analysis before tokenization overwrites them
    raw_val_texts: list[str] = val_ds[text_col]

    # ------------------------------------------------------------------
    # 2. Tokenization
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = tokenize_dataset(train_ds, tokenizer, text_col, args.max_length)
    val_ds = tokenize_dataset(val_ds, tokenizer, text_col, args.max_length)
    logger.info("Tokenization complete. Train cols: %s", train_ds.column_names)

    # ------------------------------------------------------------------
    # 3. Model setup
    # ------------------------------------------------------------------
    logger.info("Loading model: %s (num_labels=2)", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded – Total params: %s | Trainable: %s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    use_fp16 = device.type == "cuda"
    if device.type == "mps":
        logger.info("MPS detected – disabling fp16 (not supported on Apple Silicon)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=use_fp16,
        seed=args.seed,
        report_to="none",
        disable_tqdm=False,
        use_mps_device=(device.type == "mps"),
    )

    # ------------------------------------------------------------------
    # 5. Trainer with early stopping
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    start_time = time.time()
    try:
        logger.info("Starting training...")
        train_result = trainer.train()
        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds", elapsed)
        logger.info(
            "Training loss: %.4f | Steps: %d",
            train_result.training_loss,
            train_result.global_step,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "CUDA/MPS out of memory! Try reducing --batch_size (current: %d) "
                "or --max_length (current: %d).",
                args.batch_size,
                args.max_length,
            )
            # Attempt to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise SystemExit(1) from exc
        raise
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("Training failed after %.1f seconds: %s", elapsed, exc)
        raise

    # ------------------------------------------------------------------
    # 7. Save best model + tokenizer
    # ------------------------------------------------------------------
    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info("Best model saved to %s", best_model_dir)

    # ------------------------------------------------------------------
    # 8. Evaluate on validation set
    # ------------------------------------------------------------------
    metrics = evaluate_model(trainer, val_ds, raw_val_texts)

    # ------------------------------------------------------------------
    # 9. Test predictions & submission
    # ------------------------------------------------------------------
    generate_test_predictions(
        trainer=trainer,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        text_col=text_col,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # 10. Log experiment results
    # ------------------------------------------------------------------
    log_experiment_results(metrics, args, elapsed, args.output_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
