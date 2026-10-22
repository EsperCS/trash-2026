import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np

TEXT_CANDIDATES = [
    "prompt", "text", "content", "query", "instruction", "input", 
    "message", "body", "sentence", "payload", "user_input", "user_prompt",
    "full_prompt", "injections"
]

LABEL_CANDIDATES = [
    "label", "target", "category", "is_jailbreak", "class", "status", 
    "malignant", "output", "attack", "injection", "harmful", "is_malicious", 
    "annotation", "toxic", "ground_truth", "classification", "type"
]

ID_CANDIDATES = ["id", "Id", "ID", "index", "prompt_id", "sample_id", "test_id", "row_id"]

JAILBREAK_INDICATORS = [
    "jailbreak", "malicious", "unsafe", "attack", "injection", "harmful", 
    "bad", "yes", "true", "positive", "1", "act_as", "dan", "override", 
    "adversarial", "risk", "injected"
]

BENIGN_INDICATORS = [
    "benign", "safe", "normal", "conversation", "clean", "standard", 
    "no", "false", "negative", "0", "legitimate", "harmless"
]

def detect_column(df, candidates):
    # 1. Exact match
    for c in candidates:
        if c in df.columns:
            return c
    # 2. Case-insensitive / strip
    for c in candidates:
        for col in df.columns:
            if c.lower() == str(col).lower().strip().replace(" ", "_"):
                return col
    return None

def normalize_label(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return 1 if val >= 0.5 else 0
    str_val = str(val).lower().strip()
    if str_val in ["1", "1.0", "true"]:
        return 1
    if str_val in ["0", "0.0", "false"]:
        return 0
    for ind in JAILBREAK_INDICATORS:
        if ind in str_val:
            return 1
    for ind in BENIGN_INDICATORS:
        if ind in str_val:
            return 0
    return 0

print("Multi-dataset ingestion module test completed successfully!")
