import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

SEED = 42
df = pd.read_csv("data/train.csv")
df['binary_label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() == 'jailbreak' else 0)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_probs = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['binary_label'])):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    
    pipe = Pipeline([
        ('tfidf', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=80000, sublinear_tf=True)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=80000, sublinear_tf=True)),
        ])),
        ('clf', LogisticRegression(C=10.0, max_iter=1500, random_state=SEED))
    ])
    pipe.fit(train_fold['text'].astype(str).values, train_fold['binary_label'].values)
    oof_probs[val_idx] = pipe.predict_proba(val_fold['text'].astype(str).values)[:, 1]

df['oof_prob'] = oof_probs
df['pred_binary'] = (oof_probs >= 0.5).astype(int)

# Identify severe label discrepancies:
# 1. Labeled Jailbreak (1) but OOF P < 0.10 (Severe False Positives in ground truth)
gt_jailbreak_pred_benign = df[(df['binary_label'] == 1) & (df['oof_prob'] < 0.10)].copy()
gt_jailbreak_pred_benign['discrepancy'] = 'GT_Jailbreak_Model_Confident_Benign'

# 2. Labeled Benign (0) but OOF P > 0.90 (Severe False Negatives in ground truth)
gt_benign_pred_jailbreak = df[(df['binary_label'] == 0) & (df['oof_prob'] > 0.90)].copy()
gt_benign_pred_jailbreak['discrepancy'] = 'GT_Benign_Model_Confident_Jailbreak'

print(f"Total samples with severe label discrepancy (P < 0.10 for Jailbreak or P > 0.90 for Benign): {len(gt_jailbreak_pred_benign) + len(gt_benign_pred_jailbreak)}")
print(f"  - GT Jailbreak but Model P < 0.10: {len(gt_jailbreak_pred_benign)}")
print(f"  - GT Benign but Model P > 0.90: {len(gt_benign_pred_jailbreak)}")

discrepancies = pd.concat([gt_jailbreak_pred_benign, gt_benign_pred_jailbreak])
discrepancies = discrepancies.sort_values(by='oof_prob')
discrepancies.to_csv("reports/label_noise_audit.csv", index=False)
print("Saved reports/label_noise_audit.csv")

print("\nSample Confident Label Errors in Raw Dataset:")
for idx, row in discrepancies.head(8).iterrows():
    print(f"ID {row['Id']} | GT: {row['label']:<9} | OOF P: {row['oof_prob']:.4f} | Text: {row['text'][:90]}...")

for idx, row in discrepancies.tail(8).iterrows():
    print(f"ID {row['Id']} | GT: {row['label']:<9} | OOF P: {row['oof_prob']:.4f} | Text: {row['text'][:90]}...")
