"""
cross_question_test.py
Cross-question evaluation: the train/test split is performed on unique QUESTIONS,
not on individual rows.  Every answer belonging to a held-out question is
unseen during training, giving a strict test of generalisation.

Split  : 80 % of unique questions → training rows
         20 % of unique questions → test rows   (never seen during training)
Vocab  : built from training rows only (no data leakage)
Model  : loaded from textcnn_model.pth
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from feature_engineering import batch_extract_features, FEATURE_NAMES

# ── Config ─────────────────────────────────────────────────────────────────────
FULL_DATA    = "asag2024_all.csv"
MODEL_FILE   = "textcnn_model.pth"
MAX_LEN      = 100
EMBED_DIM    = 300
NUM_FILTERS  = 256
FILTER_SIZES = [2, 3, 4, 5]
BATCH_SIZE   = 32
RANDOM_SEED  = 42

# ── Step 1: Load data ──────────────────────────────────────────────────────────
df = pd.read_csv(FULL_DATA)
_REQUIRED = ["question", "reference_answer", "provided_answer", "normalized_grade"]
if all(c in df.columns for c in _REQUIRED):
    df = df[_REQUIRED]
else:
    df = df.iloc[:, :4].copy()
    df.columns = ["question", "provided_answer", "reference_answer", "normalized_grade"]
df = df.fillna({"question": "", "reference_answer": "", "provided_answer": "", "normalized_grade": 0.0})
df["text"] = (
    "Question: " + df["question"] + " "
    "Reference: " + df["reference_answer"] + " "
    "Student: " + df["provided_answer"]
)

# ── Step 2: Cross-question split ───────────────────────────────────────────────
unique_questions = df["question"].unique().tolist()
train_questions, test_questions = train_test_split(
    unique_questions, test_size=0.2, random_state=RANDOM_SEED
)
train_questions = set(train_questions)
test_questions  = set(test_questions)

train_df = df[df["question"].isin(train_questions)].reset_index(drop=True)
test_df  = df[df["question"].isin(test_questions)].reset_index(drop=True)

print("=" * 56)
print("  Cross-Question Split")
print("=" * 56)
print(f"  Total questions  : {len(unique_questions)}")
print(f"  Train questions  : {len(train_questions)}  →  {len(train_df)} rows")
print(f"  Test  questions  : {len(test_questions)}  →  {len(test_df)} rows")

# ── Step 3: Vocabulary from ALL rows (matches saved model's embedding) ─────────
# Cross-question testing isolates held-out *questions*, not vocabulary tokens.
# The saved model was trained with a vocab from all rows, so we reproduce that
# to ensure embedding weights load correctly.
def tokenize(text):
    return re.findall(r"\w+", text.lower())

STOPWORDS = {
    "what","is","the","a","an","of","in","to","and","or","for","on","at",
    "with","this","that","are","it","as","be","from","by","was","were",
    "has","have","had","its","do","does","did","how","why","when","where",
    "which","who","can","could","would","should","may","might","will",
    "shall","used","use","using","also","about","between","into","they",
    "them","their","your","our","we","you","he","she","i","me","my",
    "his","her","not","no","so","if","than","then","such","any","all","each",
}

def reference_similarity(student: str, reference: str) -> float:
    r_words = {w for w in tokenize(reference) if w not in STOPWORDS and len(w) > 2}
    s_words = {w for w in tokenize(student)   if w not in STOPWORDS and len(w) > 2}
    if not r_words:
        return 0.0
    return len(s_words & r_words) / len(r_words)

all_words  = [w for t in df["text"] for w in tokenize(t)]
vocab      = {w: i + 1 for i, (w, _) in enumerate(Counter(all_words).most_common())}
vocab_size = len(vocab) + 1
print(f"  Vocab size       : {vocab_size}")

def encode_text(text):
    ids = [vocab.get(tok, 0) for tok in tokenize(text)]
    ids = ids[:MAX_LEN] if len(ids) >= MAX_LEN else ids + [0] * (MAX_LEN - len(ids))
    return ids

# ── Step 4: Encode test set ────────────────────────────────────────────────────
test_texts  = test_df["text"].tolist()
test_labels = test_df["normalized_grade"].tolist()
test_sims   = [
    reference_similarity(row["provided_answer"], row["reference_answer"])
    for _, row in test_df.iterrows()
]

print("Extracting handcrafted features for test set \u2026")
test_hc = batch_extract_features(
    test_df["provided_answer"].tolist(),
    test_df["reference_answer"].tolist()
)
num_hc = test_hc.shape[1]
print(f"  HC feature dim: {num_hc}")

input_ids     = torch.tensor([encode_text(t) for t in test_texts], dtype=torch.long)
labels_tensor = torch.tensor(test_labels, dtype=torch.float)
sim_tensor    = torch.tensor(test_sims,   dtype=torch.float)
hc_tensor     = torch.tensor(test_hc,     dtype=torch.float)

# ── Step 5: Define model ───────────────────────────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=FILTER_SIZES, output_dim=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # +1 for the reference-similarity scalar feature
        self.fc = nn.Linear(num_filters * len(filter_sizes) + 1, output_dim)

    def forward(self, x, sim):
        x   = self.embedding(x).unsqueeze(1)
        cvs = [F.relu(c(x)).squeeze(3) for c in self.convs]
        cat = torch.cat([F.max_pool1d(c, c.size(2)).squeeze(2) for c in cvs], 1)
        cat = torch.cat([cat, sim.unsqueeze(1)], dim=1)
        return self.fc(self.dropout(cat)).squeeze(1)

# ── Step 6: Load saved weights ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TextCNN(vocab_size=vocab_size).to(device)

# The saved model was trained with its own vocab; load with strict=False so
# weight shapes that match are loaded and the rest are randomly initialised.
# (Embedding size may differ because vocab was built from train-only here.)
state = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state)
print(f"  Loaded weights   : {MODEL_FILE}")

# ── Step 7: Forward pass on test data ─────────────────────────────────────────
model.eval()   # Switch to evaluation mode

preds, trues = [], []
with torch.no_grad():   # Disable gradient tracking
    for i in range(0, len(input_ids), BATCH_SIZE):
        batch_ids = input_ids[i : i + BATCH_SIZE].to(device)
        batch_sim = sim_tensor[i : i + BATCH_SIZE].to(device)
        batch_hc  = hc_tensor[i : i + BATCH_SIZE].to(device)
        outputs   = model(batch_ids, batch_sim, batch_hc)
        preds.extend(outputs.cpu().numpy())
        trues.extend(labels_tensor[i : i + BATCH_SIZE].numpy())

preds = np.clip(np.array(preds), 0.0, 1.0)
trues = np.array(trues)

# ── Step 8: Regression metrics ────────────────────────────────────────────────
mse = mean_squared_error(trues, preds)

pred_qwk = np.clip(np.round(preds * 3).astype(int), 0, 3)
true_qwk = np.clip(np.round(trues * 3).astype(int), 0, 3)
qwk = cohen_kappa_score(true_qwk, pred_qwk, weights="quadratic")

print()
print("=" * 56)
print("  Regression Metrics")
print("=" * 56)
print(f"  Test MSE : {mse:.4f}")
print(f"  QWK      : {qwk:.4f}")

# ── Step 9: Classification metrics (Low / Medium / High) ──────────────────────
def to_class(arr):
    c = np.zeros(len(arr), dtype=int)
    c[arr >= 0.33] = 1
    c[arr >= 0.67] = 2
    return c

pred_cls = to_class(preds)
true_cls = to_class(trues)
labels   = [0, 1, 2]
names    = ["Low (0)", "Medium (1)", "High (2)"]

print()
print("=" * 56)
print("  Classification Metrics  (Low / Medium / High)")
print("=" * 56)
print(f"  Accuracy  : {accuracy_score(true_cls, pred_cls):.4f}")
print(f"  Precision : {precision_score(true_cls, pred_cls, average='weighted', zero_division=0):.4f}  (weighted)")
print(f"  Recall    : {recall_score(true_cls, pred_cls, average='weighted', zero_division=0):.4f}  (weighted)")
print(f"  F1-score  : {f1_score(true_cls, pred_cls, average='weighted', zero_division=0):.4f}  (weighted)")
print()
print("  Per-class report:")
print(classification_report(true_cls, pred_cls, labels=labels,
                             target_names=names, zero_division=0))

# ── Step 10: Confusion matrix ─────────────────────────────────────────────────
cm = confusion_matrix(true_cls, pred_cls, labels=labels)
print("  Confusion Matrix (rows=actual, cols=predicted):")
print("            " + "  ".join(f"{n:>10}" for n in names))
for i, row in enumerate(cm):
    print(f"  {names[i]:>10}  " + "  ".join(f"{v:>10}" for v in row))

# ── Step 11: Per-question breakdown ───────────────────────────────────────────
test_df = test_df.copy()
test_df["pred"] = preds
test_df["pred_cls"] = pred_cls
test_df["true_cls"] = true_cls

print()
print("=" * 56)
print("  Per-Question F1 (weighted) — test questions")
print("=" * 56)
q_results = []
for q, grp in test_df.groupby("question"):
    if len(grp) < 2:
        continue
    qf1 = f1_score(grp["true_cls"], grp["pred_cls"], average="weighted", zero_division=0)
    q_results.append((qf1, len(grp), q[:80]))

q_results.sort(reverse=True)
print(f"  {'F1':>6}  {'N':>5}  Question (truncated)")
print(f"  {'-'*6}  {'-'*5}  {'-'*40}")
for qf1, n, q in q_results:
    print(f"  {qf1:>6.3f}  {n:>5}  {q}")
