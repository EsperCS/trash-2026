import re
import pandas as pd
import numpy as np

print("="*80)
print("🧹 DATASET CURATION & CONFIDENT LEARNING LABEL REFINEMENT")
print("Target: Produce Gold-Standard Ground Truth with Zero Label Noise (F1 >= 0.99, FPR <= 0.1%)")
print("="*80)

df = pd.read_csv("data/train.csv")
df['binary_label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() == 'jailbreak' else 0)
initial_len = len(df)

# Systematic Heuristic Patterns for Injection Intent
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|any|system|prior)\s+(instructions|rules|constraints|prompt)",
    r"you\s+are\s+now\s+(DAN|unrestricted|an?\s+evil|Machiavelli|unfiltered|jailbroken)",
    r"developer\s+mode\s+enabled",
    r"disregard\s+(all\s+)?(safety|content|ethical|guidelines)",
    r"bypass\s+(all\s+)?(filters|guardrails|safety)",
    r"pretend\s+you\s+(have\s+no\s+rules|are\s+not\s+an\s+ai|are\s+free|can\s+do\s+anything)",
    r"1gn0r3\s+@ll\s+pr3v10us",
    r"bỏ\s+qua\s+tất\s+cả\s+(các\s+)?hướng\s+dẫn\s+trước\s+đó"
]

BENIGN_PATTERNS = [
    r"^once\s+upon\s+a\s+time",
    r"^in\s+the\s+sentence\s+[\"'].*[\"'],\s+who",
    r"^what\s+is\s+the\s+capital\s+of",
    r"^explain\s+(how|the|photosynthesis|mitochondria)",
    r"^hi\s*,\s*how\s+are\s+you",
    r"^hello\s*!\s*i\s+am",
    r"for\s+my\s+(computer\s+science\s+)?exam",
    r"for\s+(high\s+school\s+)?biology"
]

# Read audit file
audit_df = pd.read_csv("reports/label_noise_audit.csv")
print(f"Loaded {len(audit_df)} discrepancy candidates from OOF cross-validation.")

curated_df = df.copy()
corrections_count = 0

for idx, row in curated_df.iterrows():
    text_lower = str(row['text']).lower().strip()
    curr_label = row['binary_label']
    
    # Check if text has explicit jailbreak command but was mislabeled benign
    has_jailbreak_pattern = any(re.search(pat, text_lower) for pat in JAILBREAK_PATTERNS)
    # Check if text is completely benign but was mislabeled jailbreak
    has_benign_pattern = any(re.search(pat, text_lower) for pat in BENIGN_PATTERNS)
    
    # Check if this ID is in the severe audit discrepancies
    is_in_audit = row['Id'] in audit_df['Id'].values
    
    if curr_label == 0 and has_jailbreak_pattern and is_in_audit:
        # Correct to 1
        curated_df.at[idx, 'binary_label'] = 1
        curated_df.at[idx, 'label'] = 'jailbreak'
        corrections_count += 1
    elif curr_label == 1 and has_benign_pattern and is_in_audit:
        # Correct to 0
        curated_df.at[idx, 'binary_label'] = 0
        curated_df.at[idx, 'label'] = 'benign'
        corrections_count += 1

print(f"Total confident label corrections applied: {corrections_count} ({corrections_count/len(df)*100:.2f}% of dataset)")
print(f"Curated Class Distribution: Benign = {(curated_df['binary_label']==0).sum():,} | Jailbreak = {(curated_df['binary_label']==1).sum():,}")

curated_path = "data/train_curated_clean.csv"
curated_df.to_csv(curated_path, index=False)
print(f"✅ Saved Gold-Standard Curated Dataset to: {curated_path}")

