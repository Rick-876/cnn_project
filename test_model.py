"""
test_model.py
Loads the saved TextCNN model and evaluates it on the test set (test_set.csv).
Rebuilds the same vocabulary from the full dataset to match training embeddings.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, cohen_kappa_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
FULL_DATA  = "asag2024_all.csv"
MODEL_FILE = "textcnn_model.pth"

# ── Hyperparameters (must match training) ──────────────────────────────────────
MAX_LEN      = 100
EMBED_DIM    = 300
NUM_FILTERS  = 128
FILTER_SIZES = [2, 3, 4, 5]
BATCH_SIZE   = 32

# ── Step 1: Rebuild vocabulary from the full dataset ───────────────────────────
# The pipeline built vocab from all_texts (train + test combined), so we
# reproduce that by loading the full CSV and re-running the same split/vocab logic.
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

all_texts  = df["text"].tolist()
all_labels = df["normalized_grade"].tolist()

_, test_df_split = train_test_split(df, test_size=0.2, random_state=42)
test_texts  = test_df_split["text"].tolist()
test_labels = test_df_split["normalized_grade"].tolist()

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

# Compute sim scores for test split
test_sims = [
    reference_similarity(row["provided_answer"], row["reference_answer"])
    for _, row in test_df_split.iterrows()
]

words = [word for text in all_texts for word in tokenize(text)]
vocab       = {word: i + 1 for i, (word, _) in enumerate(Counter(words).most_common())}
vocab_size  = len(vocab) + 1

def encode_text(text):
    tokens = tokenize(text)
    ids = [vocab.get(token, 0) for token in tokens]
    ids = ids[:MAX_LEN] if len(ids) >= MAX_LEN else ids + [0] * (MAX_LEN - len(ids))
    return ids

# ── Step 2: Prepare tensors ────────────────────────────────────────────────────
input_ids     = torch.tensor([encode_text(t) for t in test_texts], dtype=torch.long)
labels_tensor = torch.tensor(test_labels, dtype=torch.float)
sim_tensor    = torch.tensor(test_sims,   dtype=torch.float)

# ── Step 3: Define model (identical architecture to training) ──────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=FILTER_SIZES, output_dim=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # +1 for reference-similarity scalar feature
        self.fc = nn.Linear(num_filters * len(filter_sizes) + 1, output_dim)

    def forward(self, x, sim):
        x   = self.embedding(x).unsqueeze(1)
        cvs = [F.relu(c(x)).squeeze(3) for c in self.convs]
        cat = torch.cat([F.max_pool1d(c, c.size(2)).squeeze(2) for c in cvs], 1)
        cat = torch.cat([cat, sim.unsqueeze(1)], dim=1)
        return self.fc(self.dropout(cat)).squeeze(1)

# ── Step 4: Load saved weights ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TextCNN(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))

# ── Step 5: Evaluate ───────────────────────────────────────────────────────────
model.eval()  # Switch to evaluation mode

preds, trues = [], []

with torch.no_grad():  # Disable gradient tracking
    for i in range(0, len(input_ids), BATCH_SIZE):
        batch_ids = input_ids[i : i + BATCH_SIZE].to(device)
        batch_sim = sim_tensor[i : i + BATCH_SIZE].to(device)
        outputs   = model(batch_ids, batch_sim)
        preds.extend(outputs.cpu().numpy())
        trues.extend(labels_tensor[i : i + BATCH_SIZE].numpy())

# ── Step 6: Report metrics ─────────────────────────────────────────────────────
preds = np.array(preds)
trues = np.array(trues)

# — Regression ——————————————————————————————————————————————————————————————
mse = mean_squared_error(trues, preds)
print("=" * 52)
print(f"  Regression")
print("=" * 52)
print(f"  Test MSE : {mse:.4f}")

# QWK (Quadratic Weighted Kappa)
pred_classes_qwk = np.clip(np.round(preds * 3).astype(int), 0, 3)
true_classes_qwk = np.clip(np.round(trues * 3).astype(int), 0, 3)
qwk = cohen_kappa_score(true_classes_qwk, pred_classes_qwk, weights="quadratic")
print(f"  QWK      : {qwk:.4f}")

# — Classification (3 classes: 0=low, 1=medium, 2=high) ———————————————————
# Bin continuous 0-1 grades: [0, 0.33) → 0, [0.33, 0.67) → 1, [0.67, 1.0] → 2
def to_class(arr):
    classes = np.zeros(len(arr), dtype=int)
    classes[arr >= 0.33] = 1
    classes[arr >= 0.67] = 2
    return classes

pred_classes = to_class(np.clip(preds, 0.0, 1.0))
true_classes = to_class(np.clip(trues, 0.0, 1.0))
class_labels = [0, 1, 2]
class_names  = ["Low (0)", "Medium (1)", "High (2)"]

print()
print("=" * 52)
print("  Classification  (Low=0 / Medium=1 / High=2)")
print("=" * 52)
print(f"  Accuracy  : {accuracy_score(true_classes, pred_classes):.4f}")
print(f"  Precision : {precision_score(true_classes, pred_classes, average='weighted', zero_division=0):.4f}  (weighted)")
print(f"  Recall    : {recall_score(true_classes, pred_classes, average='weighted', zero_division=0):.4f}  (weighted)")
print(f"  F1-score  : {f1_score(true_classes, pred_classes, average='weighted', zero_division=0):.4f}  (weighted)")

print()
print("  Per-class report:")
print(classification_report(true_classes, pred_classes,
                             labels=class_labels, target_names=class_names,
                             zero_division=0))

cm = confusion_matrix(true_classes, pred_classes, labels=class_labels)
print("  Confusion Matrix (rows=actual, cols=predicted):")
header = "          " + "  ".join(f"{n:>10}" for n in class_names)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>10}" for v in row)
    print(f"  {class_names[i]:>8}  {row_str}")

# — Sample predictions ————————————————————————————————————————————————————
print()
print("=" * 52)
print("  Sample predictions (first 10)")
print("=" * 52)
print(f"  {'Predicted':>10}  {'Actual':>8}  {'Pred Class':>10}  {'True Class':>10}")
for p, t, pc, tc in zip(preds[:10], trues[:10], pred_classes[:10], true_classes[:10]):
    print(f"  {p:>10.3f}  {t:>8.3f}  {pc:>10}  {tc:>10}")
