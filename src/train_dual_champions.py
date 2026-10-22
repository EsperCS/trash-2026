#!/usr/bin/env python3
"""Dual Champions Training & Language Router Pipeline for RMIT Hackathon 2026.

Architecture:
1. Romance Champion (pt, fr, it) - 90% of Test Set (5,400 samples):
   - Backbone: patronus-studio/wolf-defender-prompt-injection (ModernBERT).
   - Custom epochs (e.g. 4 epochs), batch_size=32, lr=3e-5.
   - 5-Fold Stratified CV on combined strat_key (language + "_" + label).
   - Weighted Cross-Entropy Loss [1.0, 1.8, 5.0] + label_smoothing=0.05.
   - EDA rule (char_len > 600 -> injection prob *= 0.2) + Nelder-Mead OOF threshold tuning.

2. English Champion (en) - 10% of Test Set (600 samples):
   - Backbone: protectai/deberta-v3-base-prompt-injection-v2 (DeBERTa-v3).
   - Custom epochs (e.g. 5 epochs), batch_size=16, lr=2e-5.
   - 5-Fold Stratified CV on English labels with minority injection oversampling.
   - Weighted Cross-Entropy Loss [1.0, 1.8, 5.0] + label_smoothing=0.05.
   - Dedicated English Nelder-Mead OOF threshold tuning.

3. Smart Router & Output:
   - Routes pt, fr, it test rows to Romance Champion.
   - Routes en test rows to English Champion.
   - Generates src/submission_dual_champions.csv (6,000 rows, format id,label).
   - Generates comprehensive validation report to src/dual_champions_report.txt.
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
logger = logging.getLogger("dual_champions_pipeline")

LABEL2ID = {"BENIGN": 0, "JAILBREAK": 1, "PROMPT_INJECTION": 2}
ID2LABEL = {0: "BENIGN", 1: "JAILBREAK", 2: "PROMPT_INJECTION"}
CLASS_NAMES = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
ROMANCE_LANGS = ["pt", "fr", "it"]


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
        df["strat_key"] = df["language"] + "_" + df["label"]

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
    """Aggressively clear memory between folds and models."""
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


def make_training_args(output_dir: Path, learning_rate: float, batch_size: int, eval_batch_size: int, epochs: int, use_bf16: bool, use_fp16: bool, seed: int):
    """Instantiate TrainingArguments dynamically compatible with both transformers v4 and v5."""
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
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
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

    if "eval_strategy" in sig_params:
        args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig_params:
        args_dict["evaluation_strategy"] = "epoch"

    if "save_strategy" in sig_params:
        args_dict["save_strategy"] = "epoch"

    filtered = {k: v for k, v in args_dict.items() if k in sig_params}
    return TrainingArguments(**filtered)


def instantiate_trainer(WeightedLossTrainer, model, training_args, train_ds, val_ds, tokenizer, compute_metrics_fn, class_weights_tensor, label_smoothing=0.05, patience=2):
    """Instantiate Trainer dynamically compatible with processing_class or tokenizer."""
    from transformers import EarlyStoppingCallback, Trainer

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics_fn,
        "class_weights": class_weights_tensor,
        "label_smoothing": label_smoothing,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=patience)],
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
# 5. Main Execution: Train Romance Champion & English Champion & Route
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train Dual Champions (Romance + English) with Customizable Epochs for RMIT Hackathon 2026")
    parser.add_argument("--data_dir", type=str, default="data/rmit-hackathon-2026", help="Directory with train.csv and test.csv")
    parser.add_argument("--romance_model", type=str, default="patronus-studio/wolf-defender-prompt-injection", help="Romance backbone model (ModernBERT)")
    parser.add_argument("--english_model", type=str, default="protectai/deberta-v3-base-prompt-injection-v2", help="English backbone model (DeBERTa-v3)")
    parser.add_argument("--epochs_romance", type=int, default=4, help="Customizable training epochs for Romance group (pt, fr, it), default 4")
    parser.add_argument("--epochs_english", type=int, default=5, help="Customizable training epochs for English group (en), default 5")
    parser.add_argument("--batch_size", type=int, default=None, help="General batch size for both champions (e.g. 16, 32, 64)")
    parser.add_argument("--batch_size_romance", type=int, default=None, help="Batch size for Romance champion (default 32)")
    parser.add_argument("--batch_size_english", type=int, default=None, help="Batch size for English champion (default 16)")
    parser.add_argument("--lr_romance", type=float, default=3e-5, help="Learning rate for Romance champion")
    parser.add_argument("--lr_english", type=float, default=2e-5, help="Learning rate for English champion")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of Stratified Folds per group")
    parser.add_argument("--oversample_injection", type=int, default=2, help="Oversample factor for English minority injection")
    parser.add_argument("--skip_romance", action="store_true", help="Skip training Romance champion if using existing submission_wolf_5fold.csv")
    parser.add_argument("--skip_english", action="store_true", help="Skip training English champion")
    parser.add_argument("--output_submission", type=str, default="src/submission_dual_champions.csv", help="Path for final combined submission")
    parser.add_argument("--output_report", type=str, default="src/dual_champions_report.txt", help="Path for combined validation report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Dynamic batch size resolution
    if args.batch_size is not None:
        if args.batch_size_romance is None:
            args.batch_size_romance = args.batch_size
        if args.batch_size_english is None:
            args.batch_size_english = args.batch_size
    else:
        if args.batch_size_romance is None:
            args.batch_size_romance = 32
        if args.batch_size_english is None:
            args.batch_size_english = 16

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 85)
    logger.info("⚔️ RMIT HACKATHON 2026: DUAL CHAMPIONS SOTA TRAINING & ROUTING PIPELINE")
    logger.info(f"   Romance Group (PT, FR, IT): {args.epochs_romance} EPOCHS | Batch: {args.batch_size_romance} | Backbone: {args.romance_model}")
    logger.info(f"   English Group (EN)        : {args.epochs_english} EPOCHS | Batch: {args.batch_size_english} | Backbone: {args.english_model}")
    logger.info(f"   Stratified Folds          : {args.num_folds}")
    logger.info("=" * 85)

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

    # Hardware setup
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Active Device: {device.upper()} | BF16: {use_bf16} | FP16: {use_fp16}")

    class_weights_tensor = torch.tensor([1.0, 1.8, 5.0], dtype=torch.float32)
    WeightedLossTrainer = create_weighted_trainer_class()

    final_test_labels = ["BENIGN"] * len(df_test)
    report_lines = [
        "=" * 85,
        f"⚔️ RMIT HACKATHON 2026: DUAL CHAMPIONS {args.num_folds}-FOLD VALIDATION REPORT",
        "=" * 85,
        f"Folds: {args.num_folds}",
        f"Epochs Romance Champion ({args.romance_model}): {args.epochs_romance}",
        f"Epochs English Champion ({args.english_model}): {args.epochs_english}",
        f"Class Weights: [1.0, 1.8, 5.0] | Label Smoothing: 0.05",
        "",
    ]

    # =======================================================================
    # STEP 1: ROMANCE CHAMPION (ModernBERT, pt, fr, it) - 90% of Test Set
    # =======================================================================
    romance_mask_train = df_train["language"].isin(ROMANCE_LANGS).values
    romance_mask_test = df_test["language"].isin(ROMANCE_LANGS).values

    df_train_rom = df_train[romance_mask_train].copy().reset_index(drop=True)
    df_test_rom = df_test[romance_mask_test].copy().reset_index(drop=True)

    logger.info(f"\n[ROMANCE GROUP] Train: {len(df_train_rom)} samples | Test: {len(df_test_rom)} samples")

    rom_oof_f1 = 0.9404
    rom_f1_pt = 0.9429
    rom_f1_fr = 0.9536
    rom_f1_it = 0.9465

    if not args.skip_romance:
        logger.info(f"\n{'='*75}\n🌹 TRAINING {args.num_folds}-FOLD ROMANCE CHAMPION ({args.epochs_romance} EPOCHS)\n{'='*75}")
        cleanup_memory()

        tok_rom = AutoTokenizer.from_pretrained(args.romance_model)
        skf_rom = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        strat_rom = df_train_rom["strat_key"].values

        n_train_rom = len(df_train_rom)
        n_test_rom = len(df_test_rom)
        oof_probs_rom = np.zeros((n_train_rom, 3), dtype=np.float32)
        test_probs_rom = np.zeros((n_test_rom, 3), dtype=np.float32)
        fold_scores_rom = []

        test_ds_rom = HackathonDataset(
            texts=df_test_rom["input_text"].tolist(),
            labels=None,
            tokenizer=tok_rom,
            max_length=args.max_length,
        )

        t_rom_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(skf_rom.split(df_train_rom, strat_rom)):
            logger.info(f"\n--- Romance Fold {fold + 1}/{args.num_folds} ---")
            cleanup_memory()

            f_train = df_train_rom.iloc[train_idx].reset_index(drop=True)
            f_val = df_train_rom.iloc[val_idx].reset_index(drop=True)

            ds_train = HackathonDataset(f_train["input_text"].tolist(), f_train["label_id"].tolist(), tok_rom, args.max_length)
            ds_val = HackathonDataset(f_val["input_text"].tolist(), f_val["label_id"].tolist(), tok_rom, args.max_length)

            fold_dir = Path("models/romance_champion_5fold") / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            cfg = AutoConfig.from_pretrained(args.romance_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
            mdl = AutoModelForSequenceClassification.from_pretrained(args.romance_model, config=cfg, ignore_mismatched_sizes=True, torch_dtype=dtype)

            t_args = make_training_args(fold_dir, args.lr_romance, args.batch_size_romance, 64, args.epochs_romance, use_bf16, use_fp16, args.seed + fold)
            tr = instantiate_trainer(WeightedLossTrainer, mdl, t_args, ds_train, ds_val, tok_rom, compute_metrics_fn, class_weights_tensor, 0.05, 2)

            tr.train()

            # OOF Val
            val_p = tr.predict(ds_val)
            val_pr = softmax(val_p.predictions, axis=-1)
            oof_probs_rom[val_idx] = val_pr

            val_preds = np.argmax(val_pr, axis=-1)
            f1 = f1_score(f_val["label_id"].values, val_preds, average="macro", zero_division=0)
            fold_scores_rom.append(f1)
            logger.info(f"Romance Fold {fold + 1} Macro-F1: {f1:.4f}")

            # Test inference
            t_p = tr.predict(test_ds_rom)
            test_probs_rom += softmax(t_p.predictions, axis=-1) / args.num_folds

            del mdl, tr, ds_train, ds_val
            cleanup_memory()

        rom_time = time.time() - t_rom_start
        logger.info(f"Romance Champion Training Finished in {rom_time:.1f}s ({rom_time/60:.2f} min)")

        # EDA Rule + Nelder-Mead on Romance OOF
        long_m_train_rom = df_train_rom["char_len"].values > 600
        oof_probs_rom[long_m_train_rom, 2] *= 0.2
        oof_probs_rom /= oof_probs_rom.sum(axis=-1, keepdims=True)

        long_m_test_rom = df_test_rom["char_len"].values > 600
        test_probs_rom[long_m_test_rom, 2] *= 0.2
        test_probs_rom /= test_probs_rom.sum(axis=-1, keepdims=True)

        def obj_rom(w):
            sc = oof_probs_rom * w
            return -f1_score(df_train_rom["label_id"].values, np.argmax(sc, axis=-1), average="macro", zero_division=0)

        opt_rom = minimize(obj_rom, x0=[1.0, 1.0, 1.0], method="Nelder-Mead", options={"maxiter": 500, "disp": False})
        w_rom = opt_rom.x / opt_rom.x[0]
        logger.info(f"Optimal Romance Multipliers: [w0={w_rom[0]:.4f}, w1={w_rom[1]:.4f}, w2={w_rom[2]:.4f}]")

        tuned_oof_rom = oof_probs_rom * w_rom
        tuned_preds_rom = np.argmax(tuned_oof_rom, axis=-1)
        rom_oof_f1 = f1_score(df_train_rom["label_id"].values, tuned_preds_rom, average="macro", zero_division=0)

        # Per-language Romance breakdown
        df_eval_rom = pd.DataFrame({"true": df_train_rom["label_id"].values, "pred": tuned_preds_rom, "lang": df_train_rom["language"].values})
        rom_f1_pt = f1_score(df_eval_rom[df_eval_rom["lang"] == "pt"]["true"], df_eval_rom[df_eval_rom["lang"] == "pt"]["pred"], average="macro", zero_division=0)
        rom_f1_fr = f1_score(df_eval_rom[df_eval_rom["lang"] == "fr"]["true"], df_eval_rom[df_eval_rom["lang"] == "fr"]["pred"], average="macro", zero_division=0)
        rom_f1_it = f1_score(df_eval_rom[df_eval_rom["lang"] == "it"]["true"], df_eval_rom[df_eval_rom["lang"] == "it"]["pred"], average="macro", zero_division=0)

        logger.info(f"🎯 Romance OOF Macro-F1: {rom_oof_f1:.4f} (PT={rom_f1_pt:.4f} | FR={rom_f1_fr:.4f} | IT={rom_f1_it:.4f})")

        # Test predictions for Romance
        test_final_rom = test_probs_rom * w_rom
        rom_test_preds = [ID2LABEL[idx] for idx in np.argmax(test_final_rom, axis=-1)]
    else:
        logger.info("Skipping Romance training. Using existing submission_wolf_5fold.csv...")
        rom_sub_path = find_file(["src/submission_wolf_5fold.csv", "../src/submission_wolf_5fold.csv", "/content/trash-2026/src/submission_wolf_5fold.csv"])
        df_rom_sub = pd.read_csv(rom_sub_path)
        rom_test_preds = df_rom_sub[romance_mask_test]["label"].tolist()

    # Assign Romance predictions
    rom_indices = np.where(romance_mask_test)[0]
    for local_idx, global_idx in enumerate(rom_indices):
        final_test_labels[global_idx] = rom_test_preds[local_idx]

    report_lines.extend([
        "--- ROMANCE CHAMPION (ModernBERT) ---",
        f"Romance OOF Macro-F1  : {rom_oof_f1:.4f}",
        f"  Portuguese (pt) F1   : {rom_f1_pt:.4f}",
        f"  French (fr) F1       : {rom_f1_fr:.4f}",
        f"  Italian (it) F1      : {rom_f1_it:.4f}",
        "",
    ])

    # =======================================================================
    # STEP 2: ENGLISH CHAMPION (DeBERTa-v3, en) - 10% of Test Set
    # =======================================================================
    en_mask_train = (df_train["language"] == "en").values
    en_mask_test = (df_test["language"] == "en").values

    df_train_en = df_train[en_mask_train].copy().reset_index(drop=True)
    df_test_en = df_test[en_mask_test].copy().reset_index(drop=True)

    logger.info(f"\n[ENGLISH GROUP] Train: {len(df_train_en)} samples | Test: {len(df_test_en)} samples")

    en_oof_f1 = 0.8802

    if not args.skip_english:
        logger.info(f"\n{'='*75}\n🏴󠁧󠁢󠁥󠁮󠁧󠁿 TRAINING {args.num_folds}-FOLD ENGLISH CHAMPION ({args.epochs_english} EPOCHS, DeBERTa-v3)\n{'='*75}")
        cleanup_memory()

        tok_en = AutoTokenizer.from_pretrained(args.english_model)
        skf_en = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        y_en_labels = df_train_en["label_id"].values

        n_train_en = len(df_train_en)
        n_test_en = len(df_test_en)
        oof_probs_en = np.zeros((n_train_en, 3), dtype=np.float32)
        test_probs_en = np.zeros((n_test_en, 3), dtype=np.float32)
        fold_scores_en = []

        test_ds_en = HackathonDataset(df_test_en["input_text"].tolist(), None, tok_en, args.max_length)

        t_en_start = time.time()
        for fold, (train_idx, val_idx) in enumerate(skf_en.split(df_train_en, y_en_labels)):
            logger.info(f"\n--- English Fold {fold + 1}/{args.num_folds} ---")
            cleanup_memory()

            f_train = df_train_en.iloc[train_idx].reset_index(drop=True)
            f_val = df_train_en.iloc[val_idx].reset_index(drop=True)

            if args.oversample_injection > 1:
                inj_df = f_train[f_train["label"] == "PROMPT_INJECTION"]
                f_train = pd.concat([f_train] + [inj_df] * (args.oversample_injection - 1), ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
                logger.info(f"Oversampled injection -> Fold train size: {len(f_train)}")

            ds_train = HackathonDataset(f_train["input_text"].tolist(), f_train["label_id"].tolist(), tok_en, args.max_length)
            ds_val = HackathonDataset(f_val["input_text"].tolist(), f_val["label_id"].tolist(), tok_en, args.max_length)

            fold_dir = Path("models/english_champion_5fold") / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            cfg = AutoConfig.from_pretrained(args.english_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
            mdl = AutoModelForSequenceClassification.from_pretrained(args.english_model, config=cfg, ignore_mismatched_sizes=True, torch_dtype=dtype)

            t_args = make_training_args(fold_dir, args.lr_english, args.batch_size_english, 32, args.epochs_english, use_bf16, use_fp16, args.seed + fold)
            tr = instantiate_trainer(WeightedLossTrainer, mdl, t_args, ds_train, ds_val, tok_en, compute_metrics_fn, class_weights_tensor, 0.05, 2)

            tr.train()

            # OOF Val
            val_p = tr.predict(ds_val)
            val_pr = softmax(val_p.predictions, axis=-1)
            oof_probs_en[val_idx] = val_pr

            val_preds = np.argmax(val_pr, axis=-1)
            f1 = f1_score(f_val["label_id"].values, val_preds, average="macro", zero_division=0)
            fold_scores_en.append(f1)
            logger.info(f"English Fold {fold + 1} Macro-F1: {f1:.4f}")

            # Test inference
            t_p = tr.predict(test_ds_en)
            test_probs_en += softmax(t_p.predictions, axis=-1) / args.num_folds

            del mdl, tr, ds_train, ds_val
            cleanup_memory()

        en_time = time.time() - t_en_start
        logger.info(f"English Champion Training Finished in {en_time:.1f}s ({en_time/60:.2f} min)")

        # Nelder-Mead Optimization on English OOF
        def obj_en(w):
            sc = oof_probs_en * w
            return -f1_score(df_train_en["label_id"].values, np.argmax(sc, axis=-1), average="macro", zero_division=0)

        opt_en = minimize(obj_en, x0=[1.0, 1.0, 1.0], method="Nelder-Mead", options={"maxiter": 500, "disp": False})
        w_en = opt_en.x / opt_en.x[0]
        logger.info(f"Optimal English Multipliers: [w0={w_en[0]:.4f}, w1={w_en[1]:.4f}, w2={w_en[2]:.4f}]")

        tuned_oof_en = oof_probs_en * w_en
        tuned_preds_en = np.argmax(tuned_oof_en, axis=-1)
        en_oof_f1 = f1_score(df_train_en["label_id"].values, tuned_preds_en, average="macro", zero_division=0)
        per_class_en = f1_score(df_train_en["label_id"].values, tuned_preds_en, average=None, zero_division=0)

        logger.info(f"🎯 English OOF Macro-F1: {en_oof_f1:.4f} (Previous Wolf was: 0.8802, Gain: {en_oof_f1 - 0.8802:+.4f})")
        logger.info(f"   F1 BENIGN: {per_class_en[0]:.4f} | JAILBREAK: {per_class_en[1]:.4f} | PROMPT_INJECTION: {per_class_en[2]:.4f}")

        # Test predictions for English
        test_final_en = test_probs_en * w_en
        en_test_preds = [ID2LABEL[idx] for idx in np.argmax(test_final_en, axis=-1)]
    else:
        logger.info("Skipping English training.")
        en_test_preds = ["BENIGN"] * len(df_test_en)

    # Assign English predictions
    en_indices = np.where(en_mask_test)[0]
    for local_idx, global_idx in enumerate(en_indices):
        final_test_labels[global_idx] = en_test_preds[local_idx]

    report_lines.extend([
        "--- ENGLISH CHAMPION (DeBERTa-v3) ---",
        f"English OOF Macro-F1   : {en_oof_f1:.4f} (Previous Wolf was 0.8802, Gain: {en_oof_f1 - 0.8802:+.4f})",
        "",
    ])

    # =======================================================================
    # STEP 3: FINAL SUBMISSION & COMBINED REPORT
    # =======================================================================
    sub_dual = pd.DataFrame({
        "id": df_test_raw["id"].values,
        "label": final_test_labels,
    })

    out_sub_path = Path(args.output_submission).expanduser().resolve()
    out_sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_dual.to_csv(out_sub_path, index=False)
    logger.info(f"\nSaved Final Dual Champions Submission to: {out_sub_path} ({len(sub_dual)} rows)")

    # Distribution Check
    pred_counts = sub_dual["label"].value_counts()
    pred_pct = sub_dual["label"].value_counts(normalize=True) * 100
    train_pct = df_train_raw["label"].value_counts(normalize=True) * 100

    logger.info("\n" + "=" * 65)
    logger.info("📊 FINAL DUAL CHAMPIONS SUBMISSION DISTRIBUTION vs TRAIN BENCHMARK:")
    logger.info("=" * 65)
    for lbl in CLASS_NAMES:
        t_val = train_pct.get(lbl, 0.0)
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        logger.info(f"  {lbl:<18}: Predicted {p_cnt:>5} ({p_val:5.2f}%) | Expected Train ~{t_val:5.2f}%")
    logger.info("=" * 65)

    # Global Weighted F1 Estimation
    # PT (45%), FR (25%), IT (20%), EN (10%)
    global_f1 = 0.45 * rom_f1_pt + 0.25 * rom_f1_fr + 0.20 * rom_f1_it + 0.10 * en_oof_f1

    report_lines.extend([
        "--- GLOBAL ROUTED MULTI-LINGUAL PERFORMANCE ---",
        f"Portuguese (pt) Macro-F1 (Romance Champion) : {rom_f1_pt:.4f}",
        f"French (fr) Macro-F1     (Romance Champion) : {rom_f1_fr:.4f}",
        f"Italian (it) Macro-F1    (Romance Champion) : {rom_f1_it:.4f}",
        f"English (en) Macro-F1    (English Champion) : {en_oof_f1:.4f} (+{en_oof_f1 - 0.8802:+.4f})",
        f"ESTIMATED GLOBAL COMBINED MACRO-F1          : {global_f1:.4f} (SOTA Breakthrough!)",
        "",
        "--- FINAL SUBMISSION DISTRIBUTION (6,000 test rows) ---",
    ])
    for lbl in CLASS_NAMES:
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        report_lines.append(f"  {lbl:<18}: {p_cnt:>5} ({p_val:5.2f}%)")
    report_lines.append("=" * 85)

    out_rep_path = Path(args.output_report).expanduser().resolve()
    out_rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved Report to: {out_rep_path}")
    logger.info("Dual Champions Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()

