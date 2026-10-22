import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

SEED = 42
df = pd.read_csv("data/train.csv")
df['binary_label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() == 'jailbreak' else 0)

train_df, val_df = train_test_split(df, test_size=0.20, stratify=df['binary_label'], random_state=SEED)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

pipe = Pipeline([
    ('tfidf', FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1, 3), max_features=75000, sublinear_tf=True)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=75000, sublinear_tf=True)),
    ])),
    ('clf', LogisticRegression(C=10.0, max_iter=1000, random_state=SEED))
])

pipe.fit(train_df['text'].astype(str).values, train_df['binary_label'].values)
probs = pipe.predict_proba(val_df['text'].astype(str).values)[:, 1]
val_df['pred_prob'] = probs
val_df['pred_label'] = (probs >= 0.5).astype(int)

# False Positives: Actual Benign (0), Predicted Jailbreak (1)
fps = val_df[(val_df['binary_label'] == 0) & (val_df['pred_label'] == 1)]
print(f"Total False Positives (FPR = {len(fps)/480*100:.2f}%): {len(fps)}")
print("\nTop 5 False Positives (highest predicted probability):")
for idx, row in fps.sort_values(by='pred_prob', ascending=False).head(5).iterrows():
    print(f"Prob: {row['pred_prob']:.4f} | ID: {row['Id']} | Text: {row['text'][:120]}...")

# False Negatives: Actual Jailbreak (1), Predicted Benign (0)
fns = val_df[(val_df['binary_label'] == 1) & (val_df['pred_label'] == 0)]
print(f"\nTotal False Negatives: {len(fns)}")
print("\nTop 5 False Negatives (lowest predicted probability):")
for idx, row in fns.sort_values(by='pred_prob', ascending=True).head(5).iterrows():
    print(f"Prob: {row['pred_prob']:.4f} | ID: {row['Id']} | Text: {row['text'][:120]}...")

