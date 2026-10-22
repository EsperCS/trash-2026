#!/usr/bin/env python3
"""5-Fold Fine-Tuning Pipeline: patronus-studio/wolf-defender-prompt-injection (ModernBERT)

Target: RMIT Hackathon 2026 - AI Security Track (3-Class Classification).
Classes: BENIGN (0), JAILBREAK (1), PROMPT_INJECTION (2).

Pipeline Architecture:
1. Feature Engineering & Metadata Injection:
   "Lang: {language} | Length: {len_bucket} ({word_len} words, {char_len} chars) | Prompt: {text}"
   Len buckets: SHORT (< 300 chars), MEDIUM (< 600 chars), LONG (>= 600 chars).
   Stratified 5-Fold split based on combined key: `language + "_" + label`.

2. Model & Loss Configuration:
   Backbone: patronus-studio/wolf-defender-prompt-injection (ModernBERT).
   Head: 3 classes with ignore_mismatched_sizes=True.
   Loss: Weighted Cross-Entropy Loss with class_weights = [1.0, 1.8, 5.0] and label_smoothing = 0.05.
   Training: 3 epochs per fold, max_length=512, batch_size=32, lr=3e-5, warmup=0.1, weight_decay=0.01.

3. Out-Of-Fold (OOF) & Threshold Optimization:
   Collect OOF probabilities across 14,000 samples and Test probabilities across 6,000 samples.
   EDA Rule: if char_len > 600, scale PROMPT_INJECTION probability by 0.2 (reduce by 80%).
   Nelder-Mead optimization on multiplier vector [w0, w1, w2] to directly maximize Macro-F1.

4. Submission & Artifacts:
   Save predictions to src/submission_wolf_5fold.csv.
   Save comprehensive validation report to src/wolf_5fold_report.txt.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_wolf_5fold")

LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
CLASS_NAMES = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LANGUAGES = ["pt", "fr", "it", "en"]


# ---------------------------------------------------------------------------
# 1. Feature Engineering & Metadata Injection
# ---------------------------------------------------------------------------


def build_metadata_text(text: str, language: str) -> Tuple[str, int]:
    """Inject language and length metadata prefix into the input prompt.

    Formula: "Lang: {language} | Length: {len_bucket} ({word_len} words, {char_len} chars) | Prompt: {text}"
    Buckets: SHORT (< 300 chars), MEDIUM (< 600 chars), LONG (>= 600 chars).
    """
    clean_text = str(text) if pd.notna(text) else ""
    char_len = len(clean_text)
    word_len = len(clean_text.split())

    if char_len < 300:
        len_bucket = "SHORT"
    elif char_len < 600:
        len_bucket = "MEDIUM"
    else:
        len_bucket = "LONG"

    engineered = (
        f"Lang: {language} | Length: {len_bucket} "
        f"({word_len} words, {char_len} chars) | Prompt: {clean_text}"
    )
    return engineered, char_len


def preprocess_dataframe(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Apply feature engineering and extract char lengths."""
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["language"] = df["language"].fillna("en").astype(str).str.strip().str.lower()

    engineered_texts = []
    char_lengths = []

    for text, lang in zip(df["text"].values, df["language"].values):
        eng, c_len = build_metadata_text(text, lang)
        engineered_texts.append(eng)
        char_lengths.append(c_len)

    df["input_text"] = engineered_texts
    df["char_len"] = char_lengths

    if not is_test:
        df["label"] = df["label"].astype(str).str.strip()
        df["label_id"] = df["label"].map(LABEL2ID)
        df["strat_key"] = df["language"] + "_" + df["label"]

    return df


# ---------------------------------------------------------------------------
# 2. Dataset Discovery
# ---------------------------------------------------------------------------


def find_file(candidates: List[str]) -> Path:
    """Find dataset file across candidate paths."""
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if p.is_file():
            return p

    search_roots = [Path.cwd(), Path("/content")]
    for root in search_roots:
        if root.exists():
            for fname in candidates:
                base = Path(fname).name
                for match in root.rglob(base):
                    if match.is_file():
                        return match.resolve()

    raise FileNotFoundError(f"Could not locate file in: {candidates}")


# ---------------------------------------------------------------------------
# 3. Custom Trainer with Weighted Cross-Entropy Loss
# ---------------------------------------------------------------------------


def create_weighted_trainer_class():
    """Dynamically define WeightedLossTrainer inheriting from transformers.Trainer."""
    import torch
    import torch.nn as nn
    from transformers import Trainer

    class WeightedLossTrainer(Trainer):
        def __init__(
            self,
            *args,
            class_weights: Optional[torch.Tensor] = None,
            label_smoothing: float = 0.05,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.label_smoothing = label_smoothing

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            if labels is not None:
                weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
                loss_fct = nn.CrossEntropyLoss(
                    weight=weights,
                    label_smoothing=self.label_smoothing,
                )
                loss = loss_fct(
                    logits.view(-1, model.config.num_labels),
                    labels.view(-1),
                )
            else:
                loss = outputs.get("loss")

            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


# ---------------------------------------------------------------------------
# 4. Metrics & Evaluation Utilities
# ---------------------------------------------------------------------------


def compute_metrics_fn(eval_pred):
    """Compute Macro-F1 and per-class F1 for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    per_class = f1_score(labels, preds, average=None, zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "f1_benign": float(per_class[0]) if len(per_class) > 0 else 0.0,
        "f1_jailbreak": float(per_class[1]) if len(per_class) > 1 else 0.0,
        "f1_prompt_injection": float(per_class[2]) if len(per_class) > 2 else 0.0,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    preds: np.ndarray,
    languages: np.ndarray,
) -> Dict[str, Any]:
    """Compute overall Macro-F1, Accuracy, Per-Class F1, and Per-Language Macro-F1."""
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_true, preds)
    macro_prec = precision_score(y_true, preds, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, preds, average="macro", zero_division=0)
    per_class = f1_score(y_true, preds, average=None, zero_division=0)

    # Multi-lingual breakdown
    df_eval = pd.DataFrame({"true": y_true, "pred": preds, "lang": languages})
    lang_f1: Dict[str, float] = {}
    for lang in LANGUAGES:
        sub = df_eval[df_eval["lang"] == lang]
        if len(sub) == 0:
            lang_f1[f"f1_{lang}"] = 0.0
            continue
        score = f1_score(sub["true"].values, sub["pred"].values, average="macro", zero_division=0)
        lang_f1[f"f1_{lang}"] = float(score)

    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "precision_macro": float(macro_prec),
        "recall_macro": float(macro_rec),
        "f1_benign": float(per_class[0]) if len(per_class) > 0 else 0.0,
        "f1_jailbreak": float(per_class[1]) if len(per_class) > 1 else 0.0,
        "f1_prompt_injection": float(per_class[2]) if len(per_class) > 2 else 0.0,
        **lang_f1,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# 5. HuggingFace PyTorch Dataset
# ---------------------------------------------------------------------------


class HackathonDataset:
    """Lightweight PyTorch Dataset for Sequence Classification."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer: Any, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            import torch
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# 6. Memory Cleanup
# ---------------------------------------------------------------------------


def cleanup_memory():
    """Aggressively clear memory between folds."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 7. Main 5-Fold Training & Optimization Pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="5-Fold Fine-Tuning of Wolf-Defender (ModernBERT) for RMIT Hackathon 2026")
    parser.add_argument("--data_dir", type=str, default="data/rmit-hackathon-2026", help="Directory with train.csv and test.csv")
    parser.add_argument("--output_dir", type=str, default="models/wolf_defender_5fold", help="Output directory for checkpoints")
    parser.add_argument("--submission_path", type=str, default="src/submission_wolf_5fold.csv", help="Path to save submission CSV")
    parser.add_argument("--report_path", type=str, default="src/wolf_5fold_report.txt", help="Path to save validation report")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of Stratified folds")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional sample limit for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("🐺 RMIT HACKATHON 2026: 5-FOLD WOLF-DEFENDER (ModernBERT) SOTA PIPELINE")
    logger.info("=" * 80)

    # 1. Load Data
    train_candidates = [
        f"{args.data_dir}/train.csv",
        "data/rmit-hackathon-2026/train.csv",
        "../data/rmit-hackathon-2026/train.csv",
        "/content/trash-2026/data/rmit-hackathon-2026/train.csv",
    ]
    test_candidates = [
        f"{args.data_dir}/test.csv",
        "data/rmit-hackathon-2026/test.csv",
        "../data/rmit-hackathon-2026/test.csv",
        "/content/trash-2026/data/rmit-hackathon-2026/test.csv",
    ]

    train_file = find_file(train_candidates)
    test_file = find_file(test_candidates)

    logger.info(f"Loaded Train: {train_file}")
    logger.info(f"Loaded Test : {test_file}")

    df_train_raw = pd.read_csv(train_file)
    df_test_raw = pd.read_csv(test_file)

    if args.sample_size and args.sample_size < len(df_train_raw):
        logger.info(f"Sampling {args.sample_size} records stratified by label for smoke test...")
        df_train_raw = (
            df_train_raw.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(int(np.ceil(args.sample_size * len(x) / len(df_train_raw))), random_state=args.seed))
            .head(args.sample_size)
            .reset_index(drop=True)
        )
        df_test_raw = df_test_raw.head(min(len(df_test_raw), args.sample_size)).reset_index(drop=True)

    # 2. Preprocess & Metadata Injection
    df_train = preprocess_dataframe(df_train_raw, is_test=False)
    df_test = preprocess_dataframe(df_test_raw, is_test=True)

    logger.info(f"Train Shape: {df_train.shape} | Test Shape: {df_test.shape}")
    logger.info(f"Train Label Distribution:\n{df_train['label'].value_counts(normalize=True).apply(lambda x: f'{x*100:.2f}%').to_string()}")
    logger.info(f"Sample Engineered Input:\n{df_train['input_text'].iloc[0][:150]}...")

    # 3. Setup Model Backbone & Tokenizer
    model_name = "patronus-studio/wolf-defender-prompt-injection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Hardware Dtype Check
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Compute Device: {device.upper()} | BF16: {use_bf16} | FP16: {use_fp16}")

    # Loss Configuration
    # class_weights = [1.0, 1.8, 5.0] to handle (62% : 31% : 7%) imbalance
    class_weights_tensor = torch.tensor([1.0, 1.8, 5.0], dtype=torch.float32)
    WeightedLossTrainer = create_weighted_trainer_class()

    # 4. Stratified 5-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    strat_keys = df_train["strat_key"].values

    n_train = len(df_train)
    n_test = len(df_test)
    oof_probs = np.zeros((n_train, 3), dtype=np.float32)
    test_probs = np.zeros((n_test, 3), dtype=np.float32)
    fold_macro_f1s = []

    test_dataset = HackathonDataset(
        texts=df_test["input_text"].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    t_start = time.time()

    # 5. Training Loop across Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, strat_keys)):
        logger.info(f"\n{'='*75}\n🔥 STARTING FOLD {fold + 1}/{args.num_folds}\n{'='*75}")
        cleanup_memory()

        fold_train_df = df_train.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df_train.iloc[val_idx].reset_index(drop=True)

        train_ds = HackathonDataset(
            texts=fold_train_df["input_text"].tolist(),
            labels=fold_train_df["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        val_ds = HackathonDataset(
            texts=fold_val_df["input_text"].tolist(),
            labels=fold_val_df["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        fold_output_dir = Path(args.output_dir) / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Model instantiation with 3 classes
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
        )

        import inspect
        sig_params = inspect.signature(TrainingArguments.__init__).parameters

        args_dict = {
            "output_dir": str(fold_output_dir),
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "num_train_epochs": args.epochs,
            "weight_decay": 0.01,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "logging_steps": 50,
            "report_to": "none",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "dataloader_num_workers": 2 if os.name != "nt" else 0,
            "seed": args.seed + fold,
        }

        # Dynamic compatibility across transformers v4 and v5
        if "warmup_ratio" in sig_params:
            args_dict["warmup_ratio"] = 0.1
        elif "warmup_steps" in sig_params:
            args_dict["warmup_steps"] = 0.1

        if "eval_strategy" in sig_params:
            args_dict["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in sig_params:
            args_dict["evaluation_strategy"] = "epoch"

        if "save_strategy" in sig_params:
            args_dict["save_strategy"] = "epoch"

        filtered_args = {k: v for k, v in args_dict.items() if k in sig_params}
        training_args = TrainingArguments(**filtered_args)

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
            class_weights=class_weights_tensor,
            label_smoothing=0.05,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train Fold Model
        trainer.train()

        # OOF Prediction for Validation set
        logger.info(f"Predicting OOF validation set for Fold {fold + 1}...")
        val_predictions = trainer.predict(val_ds)
        val_logits = val_predictions.predictions
        val_fold_probs = softmax(val_logits, axis=-1)
        oof_probs[val_idx] = val_fold_probs

        val_preds = np.argmax(val_fold_probs, axis=-1)
        fold_f1 = f1_score(fold_val_df["label_id"].values, val_preds, average="macro", zero_division=0)
        fold_macro_f1s.append(fold_f1)
        logger.info(f"✅ Fold {fold + 1} Best Validation Macro-F1: {fold_f1:.4f}")

        # Test set inference for Fold
        logger.info(f"Predicting Test set for Fold {fold + 1}...")
        test_predictions = trainer.predict(test_dataset)
        test_logits = test_predictions.predictions
        test_fold_probs = softmax(test_logits, axis=-1)
        test_probs += test_fold_probs / args.num_folds

        # Clean fold artifacts from VRAM
        del model, trainer, train_ds, val_ds
        cleanup_memory()

    total_train_time = time.time() - t_start
    logger.info(f"\n🎉 ALL {args.num_folds} FOLDS COMPLETED in {total_train_time:.1f}s ({total_train_time/60:.2f} min)")
    logger.info(f"Mean Fold Validation Macro-F1: {np.mean(fold_macro_f1s):.4f} +/- {np.std(fold_macro_f1s):.4f}")

    # -----------------------------------------------------------------------
    # 6. Out-Of-Fold Evaluation & Threshold Optimization
    # -----------------------------------------------------------------------
    y_true_all = df_train["label_id"].values
    languages_all = df_train["language"].values
    char_lens_train = df_train["char_len"].values
    char_lens_test = df_test["char_len"].values

    # Step A: Raw OOF Evaluation
    raw_oof_preds = np.argmax(oof_probs, axis=-1)
    raw_eval = evaluate_predictions(y_true_all, raw_oof_preds, languages_all)

    logger.info("\n" + "=" * 75)
    logger.info(f"📊 RAW OOF BENCHMARK (Before Post-Processing):")
    logger.info(f"   Macro-F1            : {raw_eval['macro_f1']:.4f}")
    logger.info(f"   Accuracy            : {raw_eval['accuracy']:.4f}")
    logger.info(f"   F1 BENIGN           : {raw_eval['f1_benign']:.4f}")
    logger.info(f"   F1 JAILBREAK        : {raw_eval['f1_jailbreak']:.4f}")
    logger.info(f"   F1 PROMPT_INJECTION : {raw_eval['f1_prompt_injection']:.4f}")
    logger.info(f"   Languages F1        : PT={raw_eval['f1_pt']:.4f} | FR={raw_eval['f1_fr']:.4f} | IT={raw_eval['f1_it']:.4f} | EN={raw_eval['f1_en']:.4f}")

    # Step B: Apply EDA Rule
    # If char_len > 600, scale PROMPT_INJECTION probability by 0.2 (reduce by 80%)
    logger.info("\nApplying EDA Rule (if char_len > 600, scale PROMPT_INJECTION probability by 0.2)...")
    oof_probs_eda = oof_probs.copy()
    long_mask_train = char_lens_train > 600
    oof_probs_eda[long_mask_train, 2] *= 0.2
    # Renormalize
    oof_probs_eda = oof_probs_eda / oof_probs_eda.sum(axis=-1, keepdims=True)

    test_probs_eda = test_probs.copy()
    long_mask_test = char_lens_test > 600
    test_probs_eda[long_mask_test, 2] *= 0.2
    test_probs_eda = test_probs_eda / test_probs_eda.sum(axis=-1, keepdims=True)

    eda_oof_preds = np.argmax(oof_probs_eda, axis=-1)
    eda_eval = evaluate_predictions(y_true_all, eda_oof_preds, languages_all)
    logger.info(f"   Macro-F1 after EDA Rule: {eda_eval['macro_f1']:.4f} (Delta: {eda_eval['macro_f1'] - raw_eval['macro_f1']:+.4f})")

    # Step C: Nelder-Mead Threshold / Multiplier Optimization
    logger.info("\nOptimizing multiplier vector [w0, w1, w2] with Nelder-Mead to directly maximize Macro-F1...")

    def objective(w):
        scaled = oof_probs_eda * w
        p = np.argmax(scaled, axis=-1)
        score = f1_score(y_true_all, p, average="macro", zero_division=0)
        return -score

    opt_res = minimize(
        objective,
        x0=[1.0, 1.0, 1.0],
        method="Nelder-Mead",
        options={"maxiter": 500, "disp": False},
    )

    best_weights = opt_res.x
    best_weights = best_weights / best_weights[0]  # Normalize relative to class 0
    logger.info(f"Optimal Multiplier Weights [w_benign, w_jailbreak, w_injection]: {np.round(best_weights, 4).tolist()}")

    tuned_oof_probs = oof_probs_eda * best_weights
    tuned_oof_preds = np.argmax(tuned_oof_probs, axis=-1)
    tuned_eval = evaluate_predictions(y_true_all, tuned_oof_preds, languages_all)

    logger.info("=" * 75)
    logger.info(f"🎯 TUNED OOF BENCHMARK (Post-Processing & Nelder-Mead):")
    logger.info(f"   Macro-F1            : {tuned_eval['macro_f1']:.4f} (Gain: {tuned_eval['macro_f1'] - raw_eval['macro_f1']:+.4f})")
    logger.info(f"   Accuracy            : {tuned_eval['accuracy']:.4f}")
    logger.info(f"   F1 BENIGN           : {tuned_eval['f1_benign']:.4f}")
    logger.info(f"   F1 JAILBREAK        : {tuned_eval['f1_jailbreak']:.4f}")
    logger.info(f"   F1 PROMPT_INJECTION : {tuned_eval['f1_prompt_injection']:.4f}")
    logger.info(f"   Languages F1        : PT={tuned_eval['f1_pt']:.4f} | FR={tuned_eval['f1_fr']:.4f} | IT={tuned_eval['f1_it']:.4f} | EN={tuned_eval['f1_en']:.4f}")
    logger.info("=" * 75)

    # -----------------------------------------------------------------------
    # 7. Generate Submission & Check Distribution
    # -----------------------------------------------------------------------
    test_final_probs = test_probs_eda * best_weights
    test_final_preds = np.argmax(test_final_probs, axis=-1)
    test_final_labels = [ID2LABEL[idx] for idx in test_final_preds]

    sub = pd.DataFrame({
        "id": df_test_raw["id"].values,
        "label": test_final_labels,
    })

    sub_path = Path(args.submission_path).expanduser().resolve()
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    logger.info(f"\nSaved Submission to: {sub_path} ({len(sub)} rows)")

    # Distribution Sanity Check
    pred_counts = sub["label"].value_counts()
    pred_pct = sub["label"].value_counts(normalize=True) * 100

    logger.info("\n" + "=" * 60)
    logger.info("📊 SUBMISSION PREDICTION DISTRIBUTION vs TRAIN BENCHMARK:")
    logger.info("=" * 60)
    train_pct = df_train_raw["label"].value_counts(normalize=True) * 100
    for lbl in CLASS_NAMES:
        t_val = train_pct.get(lbl, 0.0)
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        logger.info(f"  {lbl:<18}: Predicted {p_cnt:>5} ({p_val:5.2f}%) | Expected Train ~{t_val:5.2f}%")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 8. Save Comprehensive Report
    # -----------------------------------------------------------------------
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 80,
        "🐺 RMIT HACKATHON 2026: 5-FOLD WOLF-DEFENDER SOTA REPORT",
        "=" * 80,
        f"Model Backbone       : {model_name} (ModernBERT)",
        f"Num Folds            : {args.num_folds}",
        f"Epochs per Fold      : {args.epochs}",
        f"Batch Size           : {args.batch_size}",
        f"Learning Rate        : {args.learning_rate}",
        f"Class Weights        : [1.0, 1.8, 5.0]",
        f"Label Smoothing      : 0.05",
        f"Total Training Time  : {total_train_time:.1f}s ({total_train_time/60:.2f} min)",
        f"Optimal Weights      : [w0={best_weights[0]:.4f}, w1={best_weights[1]:.4f}, w2={best_weights[2]:.4f}]",
        "",
        "--- FOLD VALIDATION SCORES ---",
    ]
    for i, score in enumerate(fold_macro_f1s):
        report_lines.append(f"Fold {i+1} Macro-F1: {score:.4f}")
    report_lines.extend([
        f"Mean Fold Macro-F1: {np.mean(fold_macro_f1s):.4f} +/- {np.std(fold_macro_f1s):.4f}",
        "",
        "--- OVERALL OUT-OF-FOLD (OOF) COMPARISON ---",
        f"Metric                  | Raw OOF    | EDA Post   | Tuned OOF",
        f"------------------------+------------+------------+-----------",
        f"Macro-F1                | {raw_eval['macro_f1']:.4f}     | {eda_eval['macro_f1']:.4f}     | {tuned_eval['macro_f1']:.4f}",
        f"Accuracy                | {raw_eval['accuracy']:.4f}     | {eda_eval['accuracy']:.4f}     | {tuned_eval['accuracy']:.4f}",
        f"F1 BENIGN               | {raw_eval['f1_benign']:.4f}     | {eda_eval['f1_benign']:.4f}     | {tuned_eval['f1_benign']:.4f}",
        f"F1 JAILBREAK            | {raw_eval['f1_jailbreak']:.4f}     | {eda_eval['f1_jailbreak']:.4f}     | {tuned_eval['f1_jailbreak']:.4f}",
        f"F1 PROMPT_INJECTION     | {raw_eval['f1_prompt_injection']:.4f}     | {eda_eval['f1_prompt_injection']:.4f}     | {tuned_eval['f1_prompt_injection']:.4f}",
        "",
        "--- MULTI-LINGUAL BREAKDOWN (Tuned OOF) ---",
        f"Portuguese (pt) Macro-F1: {tuned_eval['f1_pt']:.4f}",
        f"French (fr) Macro-F1    : {tuned_eval['f1_fr']:.4f}",
        f"Italian (it) Macro-F1   : {tuned_eval['f1_it']:.4f}",
        f"English (en) Macro-F1   : {tuned_eval['f1_en']:.4f}",
        "",
        "--- CONFUSION MATRIX (Tuned OOF) ---",
        f"Labels: [BENIGN, JAILBREAK, PROMPT_INJECTION]",
        str(tuned_eval["confusion_matrix"]),
        "",
        "--- SUBMISSION DISTRIBUTION (6,000 test rows) ---",
    ])
    for lbl in CLASS_NAMES:
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        report_lines.append(f"  {lbl:<18}: {p_cnt:>5} ({p_val:5.2f}%)")
    report_lines.append("=" * 80)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved Report to: {report_path}")
    logger.info("Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
