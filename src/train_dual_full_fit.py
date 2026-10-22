#!/usr/bin/env python3
"""Dual Champions Full-Fit (100% Train Retraining) Pipeline for RMIT Hackathon 2026.

Concept:
- Combine Dual Language Champions with 100% Full-Dataset training (zero held-out data):
  1. Romance Champion (ModernBERT): Trains on 100% of Romance train data (12,600 samples: PT, FR, IT).
     Predicts 5,400 Romance test samples (90% of test set).
  2. English Champion (DeBERTa-v3): Trains on 100% of English train data (1,400 samples: EN, with minority oversampling).
     Predicts 600 English test samples (10% of test set).
  3. Smart Router: Merges test predictions back into the original test order.
  4. Generates src/submission_dual_full_fit.csv and src/dual_full_fit_report.txt.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_dual_full_fit")

LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
CLASS_NAMES = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]


# ---------------------------------------------------------------------------
# 1. Feature Engineering
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
# 2. File Discovery & Memory Management
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


def cleanup_memory():
    """Aggressively clear CUDA and CPU memory."""
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

            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(
                    weight=weights,
                    label_smoothing=self.label_smoothing,
                )
            else:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset & Trainer Instantiation
# ---------------------------------------------------------------------------


def create_tokenized_dataset_class():
    """Create PyTorch Dataset class."""
    import torch
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
        def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer, max_length: int = 512):
            self.texts = list(texts)
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            text = self.texts[idx]
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    return TextDataset


def make_training_args(
    output_dir: Path,
    learning_rate: float,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    use_bf16: bool,
    use_fp16: bool,
    seed: int,
):
    """Instantiate TrainingArguments dynamically compatible with transformers v4 and v5."""
    from transformers import TrainingArguments

    sig_params = inspect.signature(TrainingArguments.__init__).parameters
    args_dict = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "save_total_limit": 1,
        "logging_steps": 25,
        "report_to": "none",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "dataloader_num_workers": 2 if os.name != "nt" else 0,
        "seed": seed,
    }

    if "warmup_steps" in sig_params:
        args_dict["warmup_steps"] = 0.1
    elif "warmup_ratio" in sig_params:
        args_dict["warmup_ratio"] = 0.1

    if "save_strategy" in sig_params:
        args_dict["save_strategy"] = "epoch"

    filtered = {k: v for k, v in args_dict.items() if k in sig_params}
    return TrainingArguments(**filtered)


def instantiate_trainer(
    WeightedLossTrainer,
    model,
    training_args,
    train_ds,
    tokenizer,
    class_weights_tensor,
    label_smoothing: float = 0.05,
):
    """Instantiate Trainer dynamically compatible with processing_class or tokenizer."""
    from transformers import Trainer

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "class_weights": class_weights_tensor,
        "label_smoothing": label_smoothing,
    }
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    try:
        return WeightedLossTrainer(**trainer_kwargs)
    except TypeError as e:
        if "tokenizer" in str(e):
            trainer_kwargs.pop("tokenizer", None)
            trainer_kwargs["processing_class"] = tokenizer
            return WeightedLossTrainer(**trainer_kwargs)
        elif "processing_class" in str(e):
            trainer_kwargs.pop("processing_class", None)
            trainer_kwargs["tokenizer"] = tokenizer
            return WeightedLossTrainer(**trainer_kwargs)
        else:
            raise


# ---------------------------------------------------------------------------
# 5. Inference Helper with Batched PyTorch
# ---------------------------------------------------------------------------


def predict_probabilities(model, tokenizer, texts: List[str], batch_size: int = 64, max_length: int = 512, device: str = "cuda") -> np.ndarray:
    """Predict probabilities on raw texts using direct PyTorch batching."""
    import torch

    model.eval()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 3))


# ---------------------------------------------------------------------------
# 6. Main Orchestrator
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Dual Champions Full-Fit (100% Retraining) Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/rmit-hackathon-2026")
    parser.add_argument("--romance_model", type=str, default="patronus-studio/wolf-defender-prompt-injection")
    parser.add_argument("--english_model", type=str, default="protectai/deberta-v3-base-prompt-injection-v2")
    parser.add_argument("--epochs_romance", type=int, default=4, help="Epochs for Romance Champion on 12,600 samples (default 4)")
    parser.add_argument("--epochs_english", type=int, default=5, help="Epochs for English Champion on 1,400 samples (default 5)")
    parser.add_argument("--batch_size_romance", type=int, default=32)
    parser.add_argument("--batch_size_english", type=int, default=16)
    parser.add_argument("--lr_romance", type=float, default=3e-5)
    parser.add_argument("--lr_english", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="models/dual_champions_full_fit")
    parser.add_argument("--output_submission", type=str, default="src/submission_dual_full_fit.csv")
    parser.add_argument("--output_report", type=str, default="src/dual_full_fit_report.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    logger.info("=" * 85)
    logger.info("👑 RMIT HACKATHON 2026: DUAL CHAMPIONS FULL-FIT (100% DATASET RETRAINING)")
    logger.info(f"   Romance Champion : {args.romance_model} (Epochs: {args.epochs_romance}, Batch: {args.batch_size_romance})")
    logger.info(f"   English Champion : {args.english_model} (Epochs: {args.epochs_english}, Batch: {args.batch_size_english})")
    logger.info(f"   Compute Device   : {device.upper()} | bf16={use_bf16} | fp16={use_fp16}")
    logger.info("=" * 85)

    # 1. Load Data
    data_dir = Path(args.data_dir)
    train_path = find_file([str(data_dir / "train.csv"), "train.csv"])
    test_path = find_file([str(data_dir / "test.csv"), "test.csv"])

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = preprocess_dataframe(df_train, is_test=False)
    df_test = preprocess_dataframe(df_test, is_test=True)

    # Split into Romance (pt, fr, it) and English (en)
    romance_langs = {"pt", "fr", "it"}
    train_romance = df_train[df_train["language"].isin(romance_langs)].copy().reset_index(drop=True)
    train_english = df_train[df_train["language"] == "en"].copy().reset_index(drop=True)

    test_romance_mask = df_test["language"].isin(romance_langs)
    test_english_mask = df_test["language"] == "en"

    logger.info(f"📊 Dataset Splits:")
    logger.info(f"   Romance Train Samples (100%): {len(train_romance):,} | Test Samples: {test_romance_mask.sum():,}")
    logger.info(f"   English Train Samples (100%): {len(train_english):,} | Test Samples: {test_english_mask.sum():,}")

    WeightedLossTrainer = create_weighted_trainer_class()
    TextDataset = create_tokenized_dataset_class()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class weights [BENIGN=1.0, JAILBREAK=1.8, PROMPT_INJECTION=5.0]
    class_weights_tensor = torch.tensor([1.0, 1.8, 5.0], dtype=torch.float32)

    # =======================================================================
    # PHASE 1: TRAIN ROMANCE CHAMPION ON 100% ROMANCE DATA (12,600 samples)
    # =======================================================================
    logger.info("\n" + "=" * 85)
    logger.info("🌍 [PHASE 1] TRAINING ROMANCE CHAMPION ON 100% DATA (12,600 SAMPLES)")
    logger.info("=" * 85)

    romance_tokenizer = AutoTokenizer.from_pretrained(args.romance_model)
    romance_config = AutoConfig.from_pretrained(
        args.romance_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    romance_model = AutoModelForSequenceClassification.from_pretrained(
        args.romance_model,
        config=romance_config,
        ignore_mismatched_sizes=True,
    )

    romance_train_ds = TextDataset(
        texts=train_romance["input_text"].tolist(),
        labels=train_romance["label_id"].values,
        tokenizer=romance_tokenizer,
        max_length=args.max_length,
    )

    romance_train_args = make_training_args(
        output_dir=out_dir / "romance_champion",
        learning_rate=args.lr_romance,
        batch_size=args.batch_size_romance,
        eval_batch_size=args.batch_size_romance * 2,
        epochs=args.epochs_romance,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        seed=args.seed,
    )

    romance_trainer = instantiate_trainer(
        WeightedLossTrainer=WeightedLossTrainer,
        model=romance_model,
        training_args=romance_train_args,
        train_ds=romance_train_ds,
        tokenizer=romance_tokenizer,
        class_weights_tensor=class_weights_tensor,
    )

    t0_romance = time.time()
    romance_trainer.train()
    t_romance = time.time() - t0_romance
    logger.info(f"✅ Romance Champion trained in {t_romance:.1f}s!")

    # Predict on Romance Test Set
    test_romance_texts = df_test.loc[test_romance_mask, "input_text"].tolist()
    test_romance_lens = df_test.loc[test_romance_mask, "char_len"].values
    logger.info(f"🔮 Predicting on {len(test_romance_texts):,} Romance test samples...")

    probs_romance = predict_probabilities(
        model=romance_model,
        tokenizer=romance_tokenizer,
        texts=test_romance_texts,
        batch_size=args.batch_size_romance * 2,
        max_length=args.max_length,
        device=device,
    )

    # EDA Rule Post-Processing: Long prompts (char_len > 600) are rarely injection
    for idx, c_len in enumerate(test_romance_lens):
        if c_len > 600:
            probs_romance[idx, 2] *= 0.2
            row_sum = np.sum(probs_romance[idx])
            if row_sum > 0:
                probs_romance[idx] /= row_sum

    # Save Romance Model Checkpoint
    romance_model.save_pretrained(out_dir / "romance_champion_final")
    romance_tokenizer.save_pretrained(out_dir / "romance_champion_final")

    del romance_model, romance_trainer, romance_train_ds
    cleanup_memory()

    # =======================================================================
    # PHASE 2: TRAIN ENGLISH CHAMPION ON 100% ENGLISH DATA (1,400 samples)
    # =======================================================================
    logger.info("\n" + "=" * 85)
    logger.info("🇬🇧 [PHASE 2] TRAINING ENGLISH CHAMPION ON 100% DATA (1,400 SAMPLES)")
    logger.info("=" * 85)

    # Oversample minority injection class in English data (from ~100 samples up to 3x)
    inj_mask = train_english["label"] == "PROMPT_INJECTION"
    df_inj = train_english[inj_mask]
    if len(df_inj) > 0:
        logger.info(f"   Applying minority oversampling on {len(df_inj)} English Injection samples (x3)...")
        train_english_augmented = pd.concat([train_english, df_inj, df_inj], ignore_index=True)
    else:
        train_english_augmented = train_english

    english_tokenizer = AutoTokenizer.from_pretrained(args.english_model)
    english_config = AutoConfig.from_pretrained(
        args.english_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    english_model = AutoModelForSequenceClassification.from_pretrained(
        args.english_model,
        config=english_config,
        ignore_mismatched_sizes=True,
    )

    english_train_ds = TextDataset(
        texts=train_english_augmented["input_text"].tolist(),
        labels=train_english_augmented["label_id"].values,
        tokenizer=english_tokenizer,
        max_length=args.max_length,
    )

    english_train_args = make_training_args(
        output_dir=out_dir / "english_champion",
        learning_rate=args.lr_english,
        batch_size=args.batch_size_english,
        eval_batch_size=args.batch_size_english * 2,
        epochs=args.epochs_english,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        seed=args.seed,
    )

    english_trainer = instantiate_trainer(
        WeightedLossTrainer=WeightedLossTrainer,
        model=english_model,
        training_args=english_train_args,
        train_ds=english_train_ds,
        tokenizer=english_tokenizer,
        class_weights_tensor=class_weights_tensor,
    )

    t0_english = time.time()
    english_trainer.train()
    t_english = time.time() - t0_english
    logger.info(f"✅ English Champion trained in {t_english:.1f}s!")

    # Predict on English Test Set
    test_english_texts = df_test.loc[test_english_mask, "input_text"].tolist()
    test_english_lens = df_test.loc[test_english_mask, "char_len"].values
    logger.info(f"🔮 Predicting on {len(test_english_texts):,} English test samples...")

    probs_english = predict_probabilities(
        model=english_model,
        tokenizer=english_tokenizer,
        texts=test_english_texts,
        batch_size=args.batch_size_english * 2,
        max_length=args.max_length,
        device=device,
    )

    for idx, c_len in enumerate(test_english_lens):
        if c_len > 600:
            probs_english[idx, 2] *= 0.2
            row_sum = np.sum(probs_english[idx])
            if row_sum > 0:
                probs_english[idx] /= row_sum

    # Save English Model Checkpoint
    english_model.save_pretrained(out_dir / "english_champion_final")
    english_tokenizer.save_pretrained(out_dir / "english_champion_final")

    del english_model, english_trainer, english_train_ds
    cleanup_memory()

    # =======================================================================
    # PHASE 3: SMART ROUTER MERGING & SUBMISSION
    # =======================================================================
    logger.info("\n" + "=" * 85)
    logger.info("🧠 [PHASE 3] SMART ROUTER MERGING & VALIDATION")
    logger.info("=" * 85)

    pred_labels = [None] * len(df_test)

    # Route Romance
    romance_indices = df_test.index[test_romance_mask].tolist()
    romance_preds = np.argmax(probs_romance, axis=1)
    for idx_pos, df_idx in enumerate(romance_indices):
        pred_labels[df_idx] = ID2LABEL[romance_preds[idx_pos]]

    # Route English
    english_indices = df_test.index[test_english_mask].tolist()
    english_preds = np.argmax(probs_english, axis=1)
    for idx_pos, df_idx in enumerate(english_indices):
        pred_labels[df_idx] = ID2LABEL[english_preds[idx_pos]]

    df_test["predicted_label"] = pred_labels

    # Submission file
    sub_df = pd.DataFrame({
        "id": df_test["id"].values,
        "label": df_test["predicted_label"].values,
    })

    sub_path = Path(args.output_submission)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(sub_path, index=False)
    logger.info(f"🎯 Saved Dual Full-Fit Submission to: {sub_path} ({len(sub_df):,} rows)")

    # Class distribution breakdown
    counts = sub_df["label"].value_counts()
    logger.info("\n📊 Final Test Predictions Distribution:")
    for cls in CLASS_NAMES:
        c = counts.get(cls, 0)
        pct = (c / len(sub_df)) * 100
        logger.info(f"   - {cls:18s}: {c:5d} ({pct:6.2f}%)")

    # Generate Report
    report_content = f"""# RMIT HACKATHON 2026: DUAL CHAMPIONS FULL-FIT REPORT
======================================================================
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Pipeline: Dual Champions Full-Fit (100% Training on 14,000 samples)

1. Model Architectures & Full-Dataset Training:
   - Romance Champion (PT, FR, IT): {args.romance_model} (ModernBERT)
     - Trained on 100% Romance samples: {len(train_romance):,} rows
     - Epochs: {args.epochs_romance} | Learning Rate: {args.lr_romance} | Batch Size: {args.batch_size_romance}
     - Time: {t_romance:.1f}s
   - English Champion (EN): {args.english_model} (DeBERTa-v3)
     - Trained on 100% English samples: {len(train_english):,} rows (with 3x injection oversampling)
     - Epochs: {args.epochs_english} | Learning Rate: {args.lr_english} | Batch Size: {args.batch_size_english}
     - Time: {t_english:.1f}s

2. Test Set Predictions Breakdown (6,000 samples):
   - Romance Test Routed : {len(romance_indices):,} samples ({len(romance_indices)/len(df_test)*100:.1f}%)
   - English Test Routed : {len(english_indices):,} samples ({len(english_indices)/len(df_test)*100:.1f}%)

3. Predicted Class Distribution:
   - BENIGN            : {counts.get('BENIGN', 0)} ({counts.get('BENIGN', 0)/len(sub_df)*100:.2f}%)
   - JAILBREAK         : {counts.get('JAILBREAK', 0)} ({counts.get('JAILBREAK', 0)/len(sub_df)*100:.2f}%)
   - PROMPT_INJECTION  : {counts.get('PROMPT_INJECTION', 0)} ({counts.get('PROMPT_INJECTION', 0)/len(sub_df)*100:.2f}%)

4. Output Files:
   - Submission: {args.output_submission}
   - Models Dir: {args.output_dir}
======================================================================
"""
    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(report_content, encoding="utf-8")
    logger.info(f"📝 Full-Fit Report written to: {rep_path}")
    logger.info("✨ ALL TASKS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
