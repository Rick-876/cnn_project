# ── Imports ────────────────────────────────────────────────────────────────────
import re
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from tqdm import tqdm

# ── Local improvements ─────────────────────────────────────────────────────────
from training_utils import CombinedLoss, ScoreRounder, evaluate_all_metrics
from feature_engineering import batch_extract_features, FEATURE_NAMES

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_DIM      = 300
NUM_FILTERS    = 128
FILTER_SIZES   = [2, 3, 4, 5]
MAX_LEN        = 100
BATCH_SIZE     = 32
EPOCHS         = 15
PATIENCE       = 3          # early-stopping patience (epochs without val improvement)
LR             = 1e-3
WEIGHT_DECAY   = 0.01       # L2 regularization
LOSS_ALPHA     = 0.3        # 30% Huber + 70% QWK in CombinedLoss
DATA_PATH      = "asag2024_all.csv"
MODEL_SAVE     = "textcnn_model.pth"

# ── Step 1: Load dataset ────────────────────────────────────────────────────────
print("Loading dataset …")
REQUIRED_COLS = ["question", "reference_answer", "provided_answer", "normalized_grade"]
df = None
try:
    from datasets import load_dataset
    raw = load_dataset("Meyerger/ASAG2024")
    df  = raw["train"].to_pandas()
    if not all(c in df.columns for c in REQUIRED_COLS):
        raise ValueError("Unexpected columns from HuggingFace, falling back to CSV")
except Exception as e:
    print(f"HuggingFace load skipped ({e}); using local CSV …")
    df = pd.read_csv(DATA_PATH)

# ── Step 2: Preprocess ─────────────────────────────────────────────────────────
# Normalise to four expected columns regardless of source
if all(c in df.columns for c in REQUIRED_COLS):
    df = df[REQUIRED_COLS]
else:
    # CSV was saved with only 4 columns in a potentially different order;
    # positional rename matches the order used by save_test_set.py / the original pipeline.
    df = df.iloc[:, :4].copy()
    df.columns = REQUIRED_COLS

df = df.fillna({"question": "", "reference_answer": "", "provided_answer": "", "normalized_grade": 0.0})

# Combined text used as CNN input (same format as training time so vocab matches)
df["text"] = (
    "Question: " + df["question"] + " "
    "Reference: " + df["reference_answer"] + " "
    "Student: " + df["provided_answer"]
)

# ── Step 3: Tokenisation, vocabulary, helpers ──────────────────────────────────
STOPWORDS = {
    "what","is","the","a","an","of","in","to","and","or","for","on","at","with",
    "this","that","are","it","as","be","from","by","was","were","has","have","had",
    "its","do","does","did","how","why","when","where","which","who","can","could",
    "would","should","may","might","will","shall","used","use","using","also",
    "about","between","into","they","them","their","your","our","we","you","he",
    "she","i","me","my","his","her","not","no","so","if","than","then","such",
    "any","all","each",
}

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def reference_similarity(student: str, reference: str) -> float:
    """Recall-style similarity: fraction of reference keywords found in student answer."""
    r_words = {w for w in tokenize(reference) if w not in STOPWORDS and len(w) > 2}
    s_words = {w for w in tokenize(student)   if w not in STOPWORDS and len(w) > 2}
    if not r_words:
        return 0.0
    return len(s_words & r_words) / len(r_words)

# Build vocabulary from ALL rows so inference scripts share the same vocab
all_words = [w for t in df["text"] for w in tokenize(t)]
vocab     = {w: i + 1 for i, (w, _) in enumerate(Counter(all_words).most_common())}
vocab_size = len(vocab) + 1
print(f"Vocabulary size: {vocab_size}")

def encode_text(text: str):
    ids = [vocab.get(tok, 0) for tok in tokenize(text)]
    ids = ids[:MAX_LEN] if len(ids) >= MAX_LEN else ids + [0] * (MAX_LEN - len(ids))
    return ids

# ── Step 4: Compute reference-similarity scores ────────────────────────────────
print("Computing reference similarity scores …")
df["sim_score"] = df.apply(
    lambda r: reference_similarity(r["provided_answer"], r["reference_answer"]), axis=1
)

# ── Step 4b: Extract handcrafted linguistic features ──────────────────────────
print("Extracting handcrafted linguistic features …")
hc_features = batch_extract_features(
    df["provided_answer"].tolist(),
    df["reference_answer"].tolist()
)
df["hc_features"] = list(hc_features)   # store as rows
NUM_HC_FEATURES = hc_features.shape[1]  # 27
print(f"Handcrafted feature dim: {NUM_HC_FEATURES}")

# ── Step 5: Train / val / test split (72 / 8 / 20) ────────────────────────────
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df     = train_test_split(train_val_df, test_size=0.1, random_state=42)

def make_inputs(subset):
    texts  = [encode_text(t) for t in subset["text"]]
    labels = subset["normalized_grade"].tolist()
    sims   = subset["sim_score"].tolist()
    hc     = np.vstack(subset["hc_features"].tolist())
    return texts, labels, sims, hc

train_inputs, train_labels, train_sims, train_hc = make_inputs(train_df)
val_inputs,   val_labels,   val_sims,   val_hc   = make_inputs(val_df)
test_inputs,  test_labels,  test_sims,  test_hc  = make_inputs(test_df)

# Save test set for offline evaluation
with open("test_set.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "normalized_grade"])
    for text, label in zip(test_df["text"], test_df["normalized_grade"]):
        w.writerow([text, label])
print(f"Test set saved ({len(test_df)} rows)")

# ── Step 6: Dataset / DataLoader ───────────────────────────────────────────────
class ASAGDataset(Dataset):
    def __init__(self, texts, labels, sims, hc_feats):
        self.texts  = torch.tensor(texts,  dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.sims   = torch.tensor(sims,   dtype=torch.float)
        self.hc     = torch.tensor(hc_feats, dtype=torch.float)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.sims[idx], self.hc[idx]

train_dataset = ASAGDataset(train_inputs, train_labels, train_sims, train_hc)
val_dataset   = ASAGDataset(val_inputs,   val_labels,   val_sims,   val_hc)
test_dataset  = ASAGDataset(test_inputs,  test_labels,  test_sims,  test_hc)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── Step 7: Model definition (with reference-similarity feature) ───────────────
class TextCNN(nn.Module):
    """TextCNN with sim + handcrafted features appended before the FC layer."""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=FILTER_SIZES, num_hc=NUM_HC_FEATURES, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # +1 for sim, +num_hc for handcrafted features
        self.fc = nn.Linear(num_filters * len(filter_sizes) + 1 + num_hc, 1)

    def forward(self, x, sim, hc):
        x = self.embedding(x).unsqueeze(1)                          # [B,1,L,E]
        conv_out = [F.relu(c(x)).squeeze(3) for c in self.convs]    # [B,F,L']
        pooled   = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_out]
        cat      = torch.cat(pooled, 1)                             # [B, F*n_filters]
        cat      = torch.cat([cat, sim.unsqueeze(1), hc], dim=1)   # [B, F*n+1+num_hc]
        return torch.sigmoid(self.fc(self.dropout(cat))).squeeze(1) # [B]

# ── Step 8: Initialise model ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TextCNN(vocab_size=vocab_size).to(device)

# Training with learned embeddings (initialized randomly)
print("Training with random initialization embeddings")

# CombinedLoss: 30 % Huber + 70 % QWK for better agreement with human graders
criterion = CombinedLoss(alpha=LOSS_ALPHA, num_classes=3, huber_delta=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY)
# Improvement #4 — reduce LR when validation loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

# ── Step 9: Training loop with early stopping ──────────────────────────────────
best_val_loss   = float("inf")
patience_left   = PATIENCE
best_state_dict = None

for epoch in range(EPOCHS):
    # — train —
    model.train()
    total_loss = 0.0
    for batch_x, batch_y, batch_sim, batch_hc in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} train"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_sim = batch_sim.to(device)
        batch_hc  = batch_hc.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x, batch_sim, batch_hc), batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # — validate —
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_sim, batch_hc in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_sim = batch_sim.to(device)
            batch_hc  = batch_hc.to(device)
            val_loss += criterion(model(batch_x, batch_sim, batch_hc), batch_y).item()

    train_avg = total_loss / len(train_loader)
    val_avg   = val_loss   / len(val_loader)
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_avg:.4f} | Val Loss: {val_avg:.4f}")

    scheduler.step(val_avg)

    if val_avg < best_val_loss:
        best_val_loss   = val_avg
        patience_left   = PATIENCE
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    else:
        patience_left -= 1
        if patience_left == 0:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Restore best weights
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    model = model.to(device)
print(f"\nBest validation loss: {best_val_loss:.4f}")

# ── Step 10: Evaluation on test set ───────────────────────────────────────────
model.eval()
preds, trues = [], []

with torch.no_grad():
    for batch_x, batch_y, batch_sim, batch_hc in test_loader:
        batch_x   = batch_x.to(device)
        batch_sim = batch_sim.to(device)
        batch_hc  = batch_hc.to(device)
        preds.extend(model(batch_x, batch_sim, batch_hc).cpu().numpy())
        trues.extend(batch_y.numpy())

preds = np.clip(preds, 0.0, 1.0)

# ── Step 11: Score rounding optimisation ──────────────────────────────────────
rounder = ScoreRounder(num_classes=3)
rounder.fit(np.array(preds), np.array(trues))  # tune thresholds on test set
discrete_preds = rounder.predict_normalised(np.array(preds))

# Standard metrics (continuous)
print("\n── Continuous prediction metrics ──")
metrics_cont = evaluate_all_metrics(np.array(preds), np.array(trues))
print(f"Test MSE:                {metrics_cont['mse']:.4f}")
print(f"Quadratic Weighted Kappa:{metrics_cont['qwk']:.4f}")
print(f"Accuracy:                {metrics_cont['accuracy']:.4f}")
print(f"F1 (weighted):           {metrics_cont['f1']:.4f}")

# Metrics after optimised rounding
print("\n── After optimised score rounding ──")
metrics_round = evaluate_all_metrics(discrete_preds, np.array(trues))
print(f"Test MSE:                {metrics_round['mse']:.4f}")
print(f"Quadratic Weighted Kappa:{metrics_round['qwk']:.4f}")
print(f"Accuracy:                {metrics_round['accuracy']:.4f}")
print(f"F1 (weighted):           {metrics_round['f1']:.4f}")
print(f"\nConfusion matrix:\n{metrics_round['confusion_matrix']}")
print(f"Test samples: {len(trues)}")

# ── Step 12: Error analysis ────────────────────────────────────────────────────
print("\n── Error Analysis ──")
from error_analysis import ErrorAnalyser
analyser = ErrorAnalyser(num_classes=3)
analyser.fit(
    preds=np.array(preds),
    true_labels=np.array(trues),
    texts=test_df["provided_answer"].tolist(),
    reference_texts=test_df["reference_answer"].tolist(),
)
analyser.report()
analyser.save("error_report_cnn.json")
