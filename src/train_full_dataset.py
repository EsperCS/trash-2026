#!/usr/bin/env python3
"""Full Dataset Retraining (Full-Fit 100% Train) for RMIT Hackathon 2026.

Concept (Kaggle Grandmaster Playbook):
- 5-Fold Cross-Validation holds out 20% of data in each fold to validate hyperparameters and prove F1 ~ 0.9404.
- In Full-Fit Training, we train on 100% of train.csv (all 14,000 samples, zero held-out data).
- The model sees ALL 14,000 examples: all 980 prompt injections, 4,340 jailbreaks, and all multilingual nuances.
- Generates src/submission_full_fit.csv and src/full_fit_report.txt.
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
logger = logging.getLogger("train_full_dataset")

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
    """Aggressively clear memory."""
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
# 4. PyTorch Dataset
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
        "logging_steps": 50,
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


def instantiate_trainer(WeightedLossTrainer, model, training_args, train_ds, tokenizer, class_weights_tensor, label_smoothing=0.05):
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
# 5. Main Execution: Full-Fit Training on 100% of Data
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Full-Fit Retraining on 100% of Data for RMIT Hackathon 2026")
    parser.add_argument("--data_dir", type=str, default="data/rmit-hackathon-2026", help="Directory with train.csv and test.csv")
    parser.add_argument("--model_name", type=str, default="patronus-studio/wolf-defender-prompt-injection", help="Backbone model (ModernBERT)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs on full data (Customizable, default 4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default="models/wolf_defender_full_fit", help="Directory for full-fit checkpoint")
    parser.add_argument("--output_submission", type=str, default="src/submission_full_fit.csv", help="Path for full-fit submission CSV")
    parser.add_argument("--output_report", type=str, default="src/full_fit_report.txt", help="Path for full-fit report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 85)
    logger.info(f"🚀 RMIT HACKATHON 2026: FULL-FIT RETRAINING ON 100% OF DATA (14,000 SAMPLES)")
    logger.info(f"   Model Backbone : {args.model_name}")
    logger.info(f"   Total Epochs   : {args.epochs}")
    logger.info(f"   Batch Size     : {args.batch_size}")
    logger.info(f"   Learning Rate  : {args.learning_rate}")
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

    logger.info(f"Loaded 100% Train: {len(df_train)} samples (Zero samples discarded!)")
    logger.info(f"Loaded Test      : {len(df_test)} samples")
    logger.info(f"Train Label Distribution:\n{df_train['label'].value_counts(normalize=True).apply(lambda x: f'{x*100:.2f}%').to_string()}")

    # 2. Hardware and Model Setup
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Active Device: {device.upper()} | BF16: {use_bf16} | FP16: {use_fp16}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    class_weights_tensor = torch.tensor([1.0, 1.8, 5.0], dtype=torch.float32)
    WeightedLossTrainer = create_weighted_trainer_class()

    # 3. Create 100% Training Dataset
    train_ds_all = HackathonDataset(
        texts=df_train["input_text"].tolist(),
        labels=df_train["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_ds_all = HackathonDataset(
        texts=df_test["input_text"].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    out_model_dir = Path(args.output_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )

    t_args = make_training_args(
        output_dir=out_model_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=64,
        epochs=args.epochs,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        seed=args.seed,
    )

    trainer = instantiate_trainer(
        WeightedLossTrainer=WeightedLossTrainer,
        model=model,
        training_args=t_args,
        train_ds=train_ds_all,
        tokenizer=tokenizer,
        class_weights_tensor=class_weights_tensor,
        label_smoothing=0.05,
    )

    # 4. Train Model on 100% of Data
    logger.info(f"\n{'='*75}\n🔥 TRAINING MODEL ON ALL 14,000 SAMPLES ({args.epochs} EPOCHS)...\n{'='*75}")
    cleanup_memory()

    t_start = time.time()
    trainer.train()
    total_time = time.time() - t_start

    logger.info(f"✅ Training on 100% Data Completed in {total_time:.1f}s ({total_time/60:.2f} min)")

    # Save final model checkpoint
    trainer.save_model(str(out_model_dir))
    tokenizer.save_pretrained(str(out_model_dir))
    logger.info(f"Saved Full-Fit Model Checkpoint to: {out_model_dir}")

    # 5. Test Set Inference
    logger.info("\nPredicting on Test Set (6,000 samples)...")
    test_preds = trainer.predict(test_ds_all)
    test_probs = softmax(test_preds.predictions, axis=-1)

    # 6. Apply EDA Rule & Optimal Calibrated Multipliers
    # EDA Rule: if char_len > 600, scale PROMPT_INJECTION probability by 0.2
    char_lens_test = df_test["char_len"].values
    long_mask_test = char_lens_test > 600
    test_probs[long_mask_test, 2] *= 0.2
    test_probs = test_probs / test_probs.sum(axis=-1, keepdims=True)

    # Optimal multipliers from CV: [w0=1.0000, w1=0.9492, w2=0.9315]
    best_weights = np.array([1.0000, 0.9492, 0.9315], dtype=np.float32)
    test_final_probs = test_probs * best_weights
    test_final_preds = np.argmax(test_final_probs, axis=-1)
    test_final_labels = [ID2LABEL[idx] for idx in test_final_preds]

    # 7. Generate Submission
    sub = pd.DataFrame({
        "id": df_test_raw["id"].values,
        "label": test_final_labels,
    })

    out_sub_path = Path(args.output_submission).expanduser().resolve()
    out_sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_sub_path, index=False)
    logger.info(f"\nSaved Full-Fit Submission to: {out_sub_path} ({len(sub)} rows)")

    # 8. Distribution Check
    pred_counts = sub["label"].value_counts()
    pred_pct = sub["label"].value_counts(normalize=True) * 100
    train_pct = df_train_raw["label"].value_counts(normalize=True) * 100

    logger.info("\n" + "=" * 65)
    logger.info("📊 FULL-FIT SUBMISSION PREDICTION DISTRIBUTION vs TRAIN BENCHMARK:")
    logger.info("=" * 65)
    for lbl in CLASS_NAMES:
        t_val = train_pct.get(lbl, 0.0)
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        logger.info(f"  {lbl:<18}: Predicted {p_cnt:>5} ({p_val:5.2f}%) | Expected Train ~{t_val:5.2f}%")
    logger.info("=" * 65)

    # 9. Save Full-Fit Report
    out_rep_path = Path(args.output_report).expanduser().resolve()
    out_rep_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 80,
        "🚀 RMIT HACKATHON 2026: FULL-FIT (100% DATA) RETRAINING REPORT",
        "=" * 80,
        f"Model Backbone       : {args.model_name} (ModernBERT)",
        f"Training Samples     : 14,000 (100% of train.csv, zero samples held out)",
        f"Test Samples         : 6,000 (test.csv)",
        f"Total Epochs         : {args.epochs}",
        f"Batch Size           : {args.batch_size}",
        f"Learning Rate        : {args.learning_rate}",
        f"Class Weights        : [1.0, 1.8, 5.0]",
        f"Label Smoothing      : 0.05",
        f"Total Training Time  : {total_time:.1f}s ({total_time/60:.2f} min)",
        f"Saved Checkpoint     : {out_model_dir}",
        "",
        "--- SUBMISSION DISTRIBUTION (6,000 test rows) ---",
    ]
    for lbl in CLASS_NAMES:
        p_val = pred_pct.get(lbl, 0.0)
        p_cnt = pred_counts.get(lbl, 0)
        report_lines.append(f"  {lbl:<18}: {p_cnt:>5} ({p_val:5.2f}%) | Expected Train ~{train_pct.get(lbl, 0.0):.2f}%")
    report_lines.append("=" * 80)

    with open(out_rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved Report to: {out_rep_path}")
    logger.info("Full-Fit Retraining Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
