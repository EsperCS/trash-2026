import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

SEED = 42
df = pd.read_csv("data/train.csv")
df['binary_label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() == 'jailbreak' else 0)

# Load Malignant
df_mal = pd.read_csv("data/malignant/malignant_paraphrase.csv")
df_mal['binary_label'] = df_mal['category'].apply(lambda x: 0 if str(x).lower().strip() == 'conversation' else 1)

# 5-Fold Cross-Validation on RMIT Dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds_sft = np.zeros(len(df))
oof_preds_adv = np.zeros(len(df))
oof_preds_ens = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['binary_label'])):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    
    # 1. Base Model on Train Fold
    pipe1 = Pipeline([
        ('tfidf', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=80000, sublinear_tf=True, strip_accents='unicode')),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=80000, sublinear_tf=True, strip_accents='unicode')),
        ])),
        ('clf', LogisticRegression(C=12.0, max_iter=1500, random_state=SEED))
    ])
    pipe1.fit(train_fold['text'].astype(str).values, train_fold['binary_label'].values)
    p_val1 = pipe1.predict_proba(val_fold['text'].astype(str).values)[:, 1]
    oof_preds_sft[val_idx] = p_val1
    
    # 2. Adversarial Model (Train Fold + Malignant Dataset)
    comb_train = pd.concat([train_fold[['text', 'binary_label']], df_mal[['text', 'binary_label']]], ignore_index=True)
    pipe2 = Pipeline([
        ('tfidf', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=100000, sublinear_tf=True, strip_accents='unicode')),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=100000, sublinear_tf=True, strip_accents='unicode')),
        ])),
        ('clf', LogisticRegression(C=15.0, max_iter=1500, random_state=SEED))
    ])
    pipe2.fit(comb_train['text'].astype(str).values, comb_train['binary_label'].values)
    p_val2 = pipe2.predict_proba(val_fold['text'].astype(str).values)[:, 1]
    oof_preds_adv[val_idx] = p_val2
    
    # Blend: 50% pipe1 + 50% pipe2
    oof_preds_ens[val_idx] = 0.50 * p_val1 + 0.50 * p_val2

y_true = df['binary_label'].values
auc_overall = roc_auc_score(y_true, oof_preds_ens)
print("="*75)
print(f"5-FOLD STRATIFIED OUT-OF-FOLD (OOF) ROC-AUC: {auc_overall:.5f} (Entire N=4,000 dataset)")
print("="*75)

# Detailed Threshold Sweep on 4,000 out-of-fold samples
sweep_data = []
for th in np.linspace(0.10, 0.90, 81):
    preds = (oof_preds_ens >= th).astype(int)
    cm = confusion_matrix(y_true, preds)
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f = f1_score(y_true, preds, zero_division=0)
    acc = accuracy_score(y_true, preds)
    sweep_data.append({
        'th': th, 'FPR': fpr, 'Precision': p, 'Recall': r, 'F1': f, 'Accuracy': acc,
        'TP': cm[1, 1], 'FP': cm[0, 1], 'TN': cm[0, 0], 'FN': cm[1, 0]
    })

df_sweep = pd.DataFrame(sweep_data)
# Find max F1
best_f1_row = df_sweep.loc[df_sweep['F1'].idxmax()]
print(f"\n[PEAK F1 OPERATING POINT]")
print(f"Threshold: {best_f1_row['th']:.2f} | F1: {best_f1_row['F1']:.4f} | Acc: {best_f1_row['Accuracy']*100:.2f}% | Prec: {best_f1_row['Precision']*100:.2f}% | Rec: {best_f1_row['Recall']*100:.2f}% | FPR: {best_f1_row['FPR']*100:.2f}%")

# Find low FPR (<1.5% and <2%) operating points
low_fpr_rows = df_sweep[df_sweep['FPR'] <= 0.020]
if not low_fpr_rows.empty:
    best_low_fpr = low_fpr_rows.loc[low_fpr_rows['F1'].idxmax()]
    print(f"\n[LOW FALSE POSITIVE RATE POINT (FPR <= 2.0%)]")
    print(f"Threshold: {best_low_fpr['th']:.2f} | FPR: {best_low_fpr['FPR']*100:.2f}% ({int(best_low_fpr['FP'])}/2401 FP) | Prec: {best_low_fpr['Precision']*100:.2f}% | Rec: {best_low_fpr['Recall']*100:.2f}% | F1: {best_low_fpr['F1']:.4f}")

lowest_fpr_rows = df_sweep[df_sweep['FPR'] <= 0.010]
if not lowest_fpr_rows.empty:
    best_lowest = lowest_fpr_rows.loc[lowest_fpr_rows['F1'].idxmax()]
    print(f"\n[ULTRA-LOW FALSE POSITIVE RATE POINT (FPR <= 1.0%)]")
    print(f"Threshold: {best_lowest['th']:.2f} | FPR: {best_lowest['FPR']*100:.2f}% ({int(best_lowest['FP'])}/2401 FP) | Prec: {best_lowest['Precision']*100:.2f}% | Rec: {best_lowest['Recall']*100:.2f}% | F1: {best_lowest['F1']:.4f}")

