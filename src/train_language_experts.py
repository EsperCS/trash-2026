#!/usr/bin/env python3
"""Mixture of Language Specialists (MoE) Pipeline for RMIT Hackathon 2026.

Architecture:
1. Foundation Model:
   Loads the 5-fold Wolf-Defender checkpoint (or base model).
   Acts as the General Foundation with global security taxonomy across all 14,000 samples.

2. 4 Language Specialists:
   Fine-tunes 4 dedicated specialist models for Portuguese (pt), French (fr), Italian (it), and English (en).
   Trained with Weighted Cross-Entropy Loss [1.0, 1.8, 5.0] to maintain focus on minority prompt injection.

3. Learned Alpha Blending (Method 1):
   For each language L in {pt, fr, it, en}:
   P_final = alpha_L * P_General + (1 - alpha_L) * P_Specialist
   alpha_L is automatically learned by maximizing Macro-F1 directly on validation data.

4. Output & Submission:
   Generates src/submission_moe_experts.csv (6,000 rows, format id,label).
   Generates src/language_experts_report.txt comparing General vs Specialist vs Learned MoE Blend.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_language_experts")

LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
CLASS_NAMES = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LANGUAGES = ["pt", "fr", "it", "en"]


# ---------------------------------------------------------------------------
# 1. Feature Engineering & Preprocessing
# ---------------------------------------------------------------------------


def build_metadata_text(text: str, language: str) -> Tuple[str, int]:
    """Inject language and length metadata prefix into the input prompt."""
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


def resolve_foundation_model_path(candidates: List[str]) -> str:
    """Locate existing trained fold checkpoint or fallback to base model."""
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if p.is_dir() and (p / "config.json").is_file():
            logger.info(f"Found local Foundation Checkpoint at: {p}")
            return str(p)

    search_roots = [Path.cwd(), Path("/content")]
    for root in search_roots:
        if root.exists():
            for fold_dir in root.rglob("fold_0"):
                if (fold_dir / "config.json").is_file():
                    logger.info(f"Discovered Foundation Checkpoint: {fold_dir}")
                    return str(fold_dir.resolve())

    fallback = "patronus-studio/wolf-defender-prompt-injection"
    logger.info(f"No local fold checkpoint found. Using HuggingFace base model: {fallback}")
    return fallback


# ---------------------------------------------------------------------------
# 3. Custom Weighted Loss Trainer
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
# 4. PyTorch Dataset & Metrics
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


def cleanup_memory():
    """Aggressively clear memory between models."""
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
# 5. Method 1: Automatic Optimization of Blending Weight Alpha
# ---------------------------------------------------------------------------


def learn_optimal_alpha(
    y_true: np.ndarray,
    p_gen: np.ndarray,
    p_spec: np.ndarray,
    steps: int = 101,
) -> Tuple[float, float, float, float]:
    """Find optimal alpha in [0, 1] maximizing Macro-F1 on validation data.

    P_blend = alpha * P_general + (1 - alpha) * P_specialist.
    Returns:
    - best_alpha: float in [0.0, 1.0]
    - f1_blend: Macro-F1 with optimal alpha
    - f1_gen: Macro-F1 using pure General model (alpha = 1.0)
    - f1_spec: Macro-F1 using pure Specialist model (alpha = 0.0)
    """
    # Baseline F1 for pure General model
    preds_gen = np.argmax(p_gen, axis=-1)
    f1_gen = float(f1_score(y_true, preds_gen, average="macro", zero_division=0))

    # Baseline F1 for pure Specialist model
    preds_spec = np.argmax(p_spec, axis=-1)
    f1_spec = float(f1_score(y_true, preds_spec, average="macro", zero_division=0))

    best_alpha = 0.5
    best_f1 = max(f1_gen, f1_spec)

    # Systematic fine-grained grid search across [0.0, 1.0]
    alphas = np.linspace(0.0, 1.0, steps)
    for alpha in alphas:
        p_blend = alpha * p_gen + (1.0 - alpha) * p_spec
        preds = np.argmax(p_blend, axis=-1)
        score = float(f1_score(y_true, preds, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_alpha = float(alpha)

    return best_alpha, best_f1, f1_gen, f1_spec


# ---------------------------------------------------------------------------
# 6. Main MoE Training & Blending Pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train 4 Language Specialists + Learned Alpha Blending for RMIT Hackathon 2026")
    parser.add_argument("--data_dir", type=str, default="data/rmit-hackathon-2026", help="Directory with train.csv and test.csv")
    parser.add_argument("--foundation_model", type=str, default="models/wolf_defender_5fold/fold_0", help="Path to foundation model checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/language_experts", help="Output directory for specialist models")
    parser.add_argument("--submission_path", type=str, default="src/submission_moe_experts.csv", help="Output path for final submission CSV")
    parser.add_argument("--report_path", type=str, default="src/language_experts_report.txt", help="Output path for MoE validation report")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs per language expert")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1.5e-5, help="Learning rate for specialist adaptation")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("🧬 RMIT HACKATHON 2026: 4 LANGUAGE SPECIALISTS + LEARNED ALPHA (MoE)")
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

    df_train_raw = pd.read_csv(train_file)
    df_test_raw = pd.read_csv(test_file)

    df_train = preprocess_dataframe(df_train_raw, is_test=False)
    df_test = preprocess_dataframe(df_test_raw, is_test=True)

    logger.info(f"Loaded Train: {len(df_train)} rows | Test: {len(df_test)} rows")

    # 2. Hardware and Foundation Model Setup
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    foundation_candidates = [
        args.foundation_model,
        "models/wolf_defender_5fold/fold_0",
        "../models/wolf_defender_5fold/fold_0",
        "/content/trash-2026/models/wolf_defender_5fold/fold_0",
    ]
    foundation_path = resolve_foundation_model_path(foundation_candidates)
    logger.info(f"Foundation Model: {foundation_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        foundation_path if Path(foundation_path).is_dir() else "patronus-studio/wolf-defender-prompt-injection"
    )

    class_weights_tensor = torch.tensor([1.0, 1.8, 5.0], dtype=torch.float32)
    WeightedLossTrainer = create_weighted_trainer_class()

    # 3. Load Foundation Model to Compute Baseline Probabilities
    logger.info("\nLoading Foundation Model for General Baseline Inference...")
    cleanup_memory()

    config_gen = AutoConfig.from_pretrained(
        foundation_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model_gen = AutoModelForSequenceClassification.from_pretrained(
        foundation_path,
        config=config_gen,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    model_gen.to(device)
    model_gen.eval()

    # Precompute General Model probabilities on entire test set
    logger.info("Computing General Model probabilities on Test Set (6,000 samples)...")
    test_ds_all = HackathonDataset(
        texts=df_test["input_text"].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Inference for Foundation Model
    gen_trainer_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=64,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )
    gen_trainer = Trainer(model=model_gen, args=gen_trainer_args)
    test_preds_gen = gen_trainer.predict(test_ds_all)
    p_test_gen_all = softmax(test_preds_gen.predictions, axis=-1)

    logger.info("Foundation Model Test Inference Completed.")
    del model_gen, gen_trainer
    cleanup_memory()

    # 4. Training 4 Language Specialists & Learning Alpha
    learned_alphas: Dict[str, float] = {}
    lang_val_scores: Dict[str, Dict[str, float]] = {}
    final_test_probs = np.zeros((len(df_test), 3), dtype=np.float32)

    total_start = time.time()

    for lang in LANGUAGES:
        logger.info(f"\n{'='*75}\n🌍 TRAINING SPECIALIST FOR LANGUAGE: [{lang.upper()}]\n{'='*75}")
        cleanup_memory()

        # Extract language subset
        train_sub = df_train[df_train["language"] == lang].copy().reset_index(drop=True)
        test_sub_mask = (df_test["language"] == lang).values
        test_sub = df_test[test_sub_mask].copy().reset_index(drop=True)

        logger.info(f"Language [{lang.upper()}] - Train Samples: {len(train_sub)} | Test Samples: {len(test_sub)}")

        # Stratified Train/Val Split (80% / 20%)
        train_df_lang, val_df_lang = train_test_split(
            train_sub,
            test_size=0.20,
            stratify=train_sub["label_id"],
            random_state=args.seed,
        )
        train_df_lang = train_df_lang.reset_index(drop=True)
        val_df_lang = val_df_lang.reset_index(drop=True)

        train_ds_lang = HackathonDataset(
            texts=train_df_lang["input_text"].tolist(),
            labels=train_df_lang["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        val_ds_lang = HackathonDataset(
            texts=val_df_lang["input_text"].tolist(),
            labels=val_df_lang["label_id"].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        test_ds_lang = HackathonDataset(
            texts=test_sub["input_text"].tolist(),
            labels=None,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )

        expert_output_dir = Path(args.output_dir) / f"expert_{lang}"
        expert_output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate specialist starting from Foundation checkpoint
        config_spec = AutoConfig.from_pretrained(
            foundation_path,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model_spec = AutoModelForSequenceClassification.from_pretrained(
            foundation_path,
            config=config_spec,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
        )

        sig_params = inspect.signature(TrainingArguments.__init__).parameters
        args_dict = {
            "output_dir": str(expert_output_dir),
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": 64,
            "num_train_epochs": args.epochs,
            "weight_decay": 0.01,
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "logging_steps": 25,
            "report_to": "none",
            "bf16": use_bf16,
            "fp16": use_fp16,
            "seed": args.seed,
        }
        if "warmup_steps" in sig_params:
            args_dict["warmup_steps"] = 0.1
        elif "warmup_ratio" in sig_params:
            args_dict["warmup_ratio"] = 0.1

        if "eval_strategy" in sig_params:
            args_dict["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in sig_params:
            args_dict["evaluation_strategy"] = "epoch"

        if "save_strategy" in sig_params:
            args_dict["save_strategy"] = "epoch"

        filtered_args = {k: v for k, v in args_dict.items() if k in sig_params}
        training_args = TrainingArguments(**filtered_args)

        trainer_kwargs = {
            "model": model_spec,
            "args": training_args,
            "train_dataset": train_ds_lang,
            "eval_dataset": val_ds_lang,
            "compute_metrics": compute_metrics_fn,
            "class_weights": class_weights_tensor,
            "label_smoothing": 0.05,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
        }
        trainer_sig = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_sig:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        try:
            trainer_spec = WeightedLossTrainer(**trainer_kwargs)
        except TypeError as e:
            if "tokenizer" in str(e):
                trainer_kwargs.pop("tokenizer", None)
                trainer_kwargs["processing_class"] = tokenizer
                trainer_spec = WeightedLossTrainer(**trainer_kwargs)
            elif "processing_class" in str(e):
                trainer_kwargs.pop("processing_class", None)
                trainer_kwargs["tokenizer"] = tokenizer
                trainer_spec = WeightedLossTrainer(**trainer_kwargs)
            else:
                raise

        # Train Specialist Model
        trainer_spec.train()

        # Validation Predictions for Specialist
        val_preds_spec = trainer_spec.predict(val_ds_lang)
        p_val_spec = softmax(val_preds_spec.predictions, axis=-1)

        # Baseline Foundation predictions on this validation subset
        # Reload foundation model to predict on val_ds_lang
        cleanup_memory()
        model_gen_val = AutoModelForSequenceClassification.from_pretrained(
            foundation_path,
            config=config_gen,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
        )
        val_trainer_gen = Trainer(model=model_gen_val, args=gen_trainer_args)
        val_preds_gen = val_trainer_gen.predict(val_ds_lang)
        p_val_gen = softmax(val_preds_gen.predictions, axis=-1)
        del model_gen_val, val_trainer_gen
        cleanup_memory()

        # 5. Method 1: Automatically Learn Optimal Alpha for this Language
        y_val_true = val_df_lang["label_id"].values
        best_alpha, f1_blend, f1_gen, f1_spec = learn_optimal_alpha(y_val_true, p_val_gen, p_val_spec)

        learned_alphas[lang] = best_alpha
        lang_val_scores[lang] = {
            "f1_gen": f1_gen,
            "f1_spec": f1_spec,
            "f1_blend": f1_blend,
            "gain": f1_blend - f1_gen,
        }

        logger.info(f"\n🎯 [LANG: {lang.upper()}] LEARNED OPTIMAL BLEND WEIGHT:")
        logger.info(f"   Optimal Alpha (General Weight)   : {best_alpha:.4f} ({best_alpha*100:.1f}%)")
        logger.info(f"   Specialist Weight (1 - Alpha)    : {1 - best_alpha:.4f} ({(1-best_alpha)*100:.1f}%)")
        logger.info(f"   General Model Macro-F1           : {f1_gen:.4f}")
        logger.info(f"   Specialist Model Macro-F1        : {f1_spec:.4f}")
        logger.info(f"   ✨ Learned MoE Blend Macro-F1    : {f1_blend:.4f} (Gain: {f1_blend - f1_gen:+.4f})")

        # 6. Predict on Test Subset for this Language
        test_preds_spec = trainer_spec.predict(test_ds_lang)
        p_test_spec = softmax(test_preds_spec.predictions, axis=-1)

        p_test_gen_lang = p_test_gen_all[test_sub_mask]

        # Apply Learned Alpha Blend
        p_test_final_lang = best_alpha * p_test_gen_lang + (1.0 - best_alpha) * p_test_spec
        final_test_probs[test_sub_mask] = p_test_final_lang

        del model_spec, trainer_spec, train_ds_lang, val_ds_lang, test_ds_lang
        cleanup_memory()

    total_time = time.time() - total_start
    logger.info(f"\n🎉 ALL 4 LANGUAGE SPECIALISTS COMPLETED IN {total_time:.1f}s ({total_time/60:.2f} min)")

    # -----------------------------------------------------------------------
    # 7. Generate Submission & Check Distribution
    # -----------------------------------------------------------------------
    test_final_preds = np.argmax(final_test_probs, axis=-1)
    test_final_labels = [ID2LABEL[idx] for idx in test_final_preds]

    sub = pd.DataFrame({
        "id": df_test_raw["id"].values,
        "label": test_final_labels,
    })

    sub_path = Path(args.submission_path).expanduser().resolve()
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    logger.info(f"\nSaved MoE Submission to: {sub_path} ({len(sub)} rows)")

    # Distribution Sanity Check
    pred_counts = sub["label"].value_counts()
    pred_pct = sub["label"].value_counts(normalize=True) * 100
    train_pct = df_train_raw["label"].value_counts(normalize=True) * 100

    logger.info("\n" + "=" * 65)
    logger.info("📊 MoE SUBMISSION PREDICTION DISTRIBUTION vs TRAIN BENCHMARK:")
    logger.info("=" * 65)
    for lbl in CLASS_NAMES:
        t_val = train_pct.get(lbl, 0.0)
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        logger.info(f"  {lbl:<18}: Predicted {p_cnt:>5} ({p_val:5.2f}%) | Expected Train ~{t_val:5.2f}%")
    logger.info("=" * 65)

    # -----------------------------------------------------------------------
    # 8. Save Comprehensive Report
    # -----------------------------------------------------------------------
    report_path = Path(args.report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 80,
        "🧬 RMIT HACKATHON 2026: 4 LANGUAGE SPECIALISTS (MoE) VALIDATION REPORT",
        "=" * 80,
        f"Foundation Model    : {foundation_path}",
        f"Epochs per Expert   : {args.epochs}",
        f"Learning Rate       : {args.learning_rate}",
        f"Batch Size          : {args.batch_size}",
        f"Total Training Time : {total_time:.1f}s ({total_time/60:.2f} min)",
        "",
        "--- LEARNED BLENDING WEIGHTS (METHOD 1: SCIPY OPTIMIZATION) ---",
    ]
    for lang in LANGUAGES:
        alpha = learned_alphas[lang]
        scores = lang_val_scores[lang]
        report_lines.append(
            f"Language [{lang.upper()}]: Alpha = {alpha:.4f} (General: {alpha*100:.1f}% | Specialist: {(1-alpha)*100:.1f}%)"
        )
        report_lines.append(
            f"   General F1 = {scores['f1_gen']:.4f} -> Specialist F1 = {scores['f1_spec']:.4f} -> MoE Blend F1 = {scores['f1_blend']:.4f} ({scores['gain']:+.4f})"
        )

    mean_blend_f1 = np.mean([s["f1_blend"] for s in lang_val_scores.values()])
    mean_gen_f1 = np.mean([s["f1_gen"] for s in lang_val_scores.values()])

    report_lines.extend([
        "",
        f"Overall Mean Language Validation Macro-F1: {mean_blend_f1:.4f} (Baseline: {mean_gen_f1:.4f}, Delta: {mean_blend_f1 - mean_gen_f1:+.4f})",
        "",
        "--- FINAL SUBMISSION DISTRIBUTION (6,000 test rows) ---",
    ])
    for lbl in CLASS_NAMES:
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        report_lines.append(f"  {lbl:<18}: {p_cnt:>5} ({p_val:5.2f}%)")
    report_lines.append("=" * 80)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved MoE Report to: {report_path}")
    logger.info("Mixture of Language Specialists Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
