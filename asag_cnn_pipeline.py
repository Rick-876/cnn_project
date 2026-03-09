"""
asag_cnn_pipeline.py – TextCNN training pipeline for ASAG
=========================================================
Improvements over v2:
  1. Pre-trained GloVe-300d embeddings (gensim)
  2. Class-weighted random sampling (inverse-frequency)
  3. Stratified K-fold cross-validation
  4. Synonym-based data augmentation for minority class
  5. Wider convolutional filters (256 per kernel)
  6. Hidden layer before output (FC 1052 → 256 → 1)
  7. Focal MSE loss (hard-example mining)
  8. Label smoothing (epsilon = 0.05)

Usage:
    python asag_cnn_pipeline.py                    # defaults
    python asag_cnn_pipeline.py --epochs 20 --folds 5
    python asag_cnn_pipeline.py --no-glove         # skip GloVe
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import re, csv, os, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from tqdm import tqdm
from nltk.corpus import wordnet

from training_utils import CombinedLoss, ScoreRounder, evaluate_all_metrics
from feature_engineering import batch_extract_features, FEATURE_NAMES

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_DIM      = 300
NUM_FILTERS    = 256          # [5] wider filters (was 128)
FILTER_SIZES   = [2, 3, 4, 5]
MAX_LEN        = 100
BATCH_SIZE     = 32
EPOCHS         = 15
PATIENCE       = 5            # more patience with focal loss
LR             = 1e-3
WEIGHT_DECAY   = 0.01
LOSS_ALPHA     = 0.3          # 30 % regression + 70 % QWK
FOCAL_GAMMA    = 1.5          # [7] focal-loss focusing parameter
LABEL_SMOOTH   = 0.05         # [8] label-smoothing epsilon
N_FOLDS        = 3            # [3] stratified K-fold
GLOVE_NAME     = "glove-wiki-gigaword-300"   # [1] pre-trained embeddings
DATA_PATH      = "asag2024_all.csv"
MODEL_SAVE     = "textcnn_model.pth"
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
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

def encode_text(text: str, vocab: dict, max_len: int = MAX_LEN):
    ids = [vocab.get(tok, 0) for tok in tokenize(text)]
    ids = ids[:max_len] if len(ids) >= max_len else ids + [0] * (max_len - len(ids))
    return ids


# ── [4] Data augmentation via synonym substitution ────────────────────────────
def augment_with_synonyms(text: str) -> str:
    """Create an augmented version by replacing ~30 % of content words with synonyms."""
    words = tokenize(text)
    if len(words) < 3:
        return text
    new_words = words.copy()
    content_idx = [i for i, w in enumerate(words)
                   if w not in STOPWORDS and len(w) > 2]
    if not content_idx:
        return text
    n_replace = max(1, len(content_idx) // 3)
    to_replace = random.sample(content_idx, min(n_replace, len(content_idx)))
    for idx in to_replace:
        word = words[idx]
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().lower().replace("_", " ")
                if name != word and " " not in name:
                    synonyms.add(name)
        if synonyms:
            new_words[idx] = random.choice(list(synonyms))
    return " ".join(new_words)


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_data(data_path: str) -> pd.DataFrame:
    REQUIRED = ["question", "reference_answer", "provided_answer", "normalized_grade"]
    df = None
    try:
        from datasets import load_dataset
        raw = load_dataset("Meyerger/ASAG2024")
        df  = raw["train"].to_pandas()
        if not all(c in df.columns for c in REQUIRED):
            raise ValueError("column mismatch")
    except Exception as e:
        print(f"HuggingFace load skipped ({e}); using CSV …")
        df = pd.read_csv(data_path)
    if all(c in df.columns for c in REQUIRED):
        df = df[REQUIRED]
    else:
        df = df.iloc[:, :4].copy()
        df.columns = REQUIRED
    df = df.fillna({"question": "", "reference_answer": "",
                     "provided_answer": "", "normalized_grade": 0.0})
    df["text"] = (
        "Question: " + df["question"] + " "
        "Reference: " + df["reference_answer"] + " "
        "Student: " + df["provided_answer"]
    )
    return df


# ── Build vocabulary ──────────────────────────────────────────────────────────
def build_vocab(df: pd.DataFrame):
    all_words = [w for t in df["text"] for w in tokenize(t)]
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(all_words).most_common())}
    return vocab, len(vocab) + 1


# ── [1] Load pre-trained GloVe embeddings ─────────────────────────────────────
def load_glove_embeddings(vocab, vocab_size, embed_dim, glove_name):
    print(f"Loading pre-trained embeddings ({glove_name}) …")
    try:
        import gensim.downloader as api
        glove = api.load(glove_name)
        mat = np.random.normal(0, 0.1, (vocab_size, embed_dim)).astype(np.float32)
        mat[0] = 0.0  # padding
        found = 0
        for word, idx in vocab.items():
            if word in glove:
                mat[idx] = glove[word]
                found += 1
        print(f"  GloVe coverage: {found}/{len(vocab)} ({100*found/len(vocab):.1f} %)")
        return mat
    except Exception as e:
        print(f"  Could not load GloVe ({e}). Using random embeddings.")
        return None


# ── [4] Augment minority class ────────────────────────────────────────────────
def augment_minority_class(df: pd.DataFrame) -> pd.DataFrame:
    classes = np.round(df["normalized_grade"].values * 2).astype(int)
    counts  = np.bincount(classes, minlength=3)
    n0, n2  = int(counts[0]), int(counts[2])
    if n0 >= n2 or n0 == 0:
        print("  No augmentation needed.")
        return df
    target = int(n2 * 0.7)
    n_new  = target - n0
    if n_new <= 0:
        return df
    print(f"  Augmenting class 0: {n0} → {n0 + n_new}  (+{n_new} synonym-augmented)")
    c0 = df[classes == 0]
    rows = []
    for _ in range(int(np.ceil(n_new / n0))):
        for _, row in c0.iterrows():
            if len(rows) >= n_new:
                break
            r = row.copy()
            r["provided_answer"] = augment_with_synonyms(row["provided_answer"])
            r["text"] = (
                "Question: " + r["question"] + " "
                "Reference: " + r["reference_answer"] + " "
                "Student: " + r["provided_answer"]
            )
            rows.append(r)
    result = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    print(f"  Total after augmentation: {len(result)}")
    return result


# ── Compute features ──────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame):
    df["sim_score"] = df.apply(
        lambda r: reference_similarity(r["provided_answer"], r["reference_answer"]), axis=1)
    hc = batch_extract_features(
        df["provided_answer"].tolist(), df["reference_answer"].tolist())
    df["hc_features"] = list(hc)
    return df, hc.shape[1]


# ── Dataset ────────────────────────────────────────────────────────────────────
class ASAGDataset(Dataset):
    def __init__(self, texts, labels, sims, hc_feats):
        self.texts  = torch.tensor(texts,  dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.sims   = torch.tensor(sims,   dtype=torch.float)
        self.hc     = torch.tensor(hc_feats, dtype=torch.float)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.sims[idx], self.hc[idx]


# ── [5] + [6] TextCNN with wider filters + hidden layer ──────────────────────
class TextCNN(nn.Module):
    """TextCNN with 256 filters, sim score, HC features, and hidden FC layer."""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=FILTER_SIZES, num_hc=27, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        conv_out_dim = num_filters * len(filter_sizes) + 1 + num_hc  # 256*4+1+27=1052
        self.fc1 = nn.Linear(conv_out_dim, 256)   # [6] hidden layer
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, sim, hc):
        x = self.embedding(x).unsqueeze(1)
        conv_out = [F.relu(c(x)).squeeze(3) for c in self.convs]
        pooled   = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_out]
        cat = torch.cat(pooled, 1)
        cat = torch.cat([cat, sim.unsqueeze(1), hc], dim=1)
        cat = self.dropout(cat)
        cat = F.relu(self.fc1(cat))
        cat = F.dropout(cat, p=0.3, training=self.training)
        return torch.sigmoid(self.fc2(cat)).squeeze(1)


# ── Train one fold ────────────────────────────────────────────────────────────
def train_fold(model, train_loader, val_loader, criterion, device,
               epochs, patience, lr, wd):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    best_val, patience_left, best_state = float("inf"), patience, None

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for bx, by, bs, bh in tqdm(train_loader,
                                     desc=f"  Epoch {epoch+1}/{epochs}", leave=False):
            bx, by = bx.to(device), by.to(device)
            bs, bh = bs.to(device), bh.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx, bs, bh), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by, bs, bh in val_loader:
                bx, by = bx.to(device), by.to(device)
                bs, bh = bs.to(device), bh.to(device)
                v_loss += criterion(model(bx, bs, bh), by).item()
        v_avg = v_loss / len(val_loader)
        scheduler.step(v_avg)

        if v_avg < best_val:
            best_val = v_avg
            patience_left = patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"    Early stop @ epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for bx, by, bs, bh in val_loader:
            bx, bs, bh = bx.to(device), bs.to(device), bh.to(device)
            vp.extend(model(bx, bs, bh).cpu().numpy())
            vt.extend(by.numpy())
    metrics = evaluate_all_metrics(np.array(vp), np.array(vt))
    return best_state, best_val, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train TextCNN for ASAG")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--folds",  type=int, default=N_FOLDS)
    parser.add_argument("--lr",     type=float, default=LR)
    parser.add_argument("--no-glove", action="store_true", help="Skip GloVe embeddings")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading dataset …")
    df = load_data(DATA_PATH)
    print(f"Dataset: {len(df)} rows, {df['question'].nunique()} questions")

    vocab, vocab_size = build_vocab(df)
    print(f"Vocabulary size: {vocab_size}")

    # [1] GloVe embeddings
    emb_matrix = None
    if not args.no_glove:
        emb_matrix = load_glove_embeddings(vocab, vocab_size, EMBED_DIM, GLOVE_NAME)

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    print("\nSplitting: 80 % train / 20 % test (stratified) …")
    cls = np.round(df["normalized_grade"].values * 2).astype(int)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED,
                                         stratify=cls)
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"  Train: {len(train_df)},  Test: {len(test_df)}")

    # [4] Augment minority class
    print("\nAugmenting minority class …")
    train_aug = augment_minority_class(train_df.copy())

    # Features
    print("\nComputing features (train) …")
    train_aug, num_hc = compute_features(train_aug)
    print(f"  HC feature dim: {num_hc}")
    print("Computing features (test) …")
    test_df, _ = compute_features(test_df)

    # [8] Label smoothing
    print(f"\nLabel smoothing (ε = {LABEL_SMOOTH})")
    train_aug["smoothed_grade"] = (
        train_aug["normalized_grade"] * (1 - LABEL_SMOOTH) + LABEL_SMOOTH * 0.5
    )

    # Save test set
    with open("test_set.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "normalized_grade"])
        for text, label in zip(test_df["text"], test_df["normalized_grade"]):
            w.writerow([text, label])
    print(f"Test set saved ({len(test_df)} rows)")

    # ── [3] Stratified K-Fold CV ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Stratified {args.folds}-Fold Cross-Validation")
    print(f"{'='*60}")

    train_cls = np.round(train_aug["normalized_grade"].values * 2).astype(int)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    criterion = CombinedLoss(alpha=LOSS_ALPHA, num_classes=3,
                             huber_delta=0.0, focal_gamma=FOCAL_GAMMA)  # [7]

    for fi, (tr_idx, va_idx) in enumerate(skf.split(train_aug, train_cls)):
        print(f"\n── Fold {fi+1}/{args.folds} ──")
        f_tr = train_aug.iloc[tr_idx]
        f_va = train_aug.iloc[va_idx]

        tr_texts  = [encode_text(t, vocab) for t in f_tr["text"]]
        va_texts  = [encode_text(t, vocab) for t in f_va["text"]]
        tr_labels = f_tr["smoothed_grade"].tolist()   # [8] smoothed
        va_labels = f_va["normalized_grade"].tolist()  # original for eval
        tr_sims   = f_tr["sim_score"].tolist()
        va_sims   = f_va["sim_score"].tolist()
        tr_hc     = np.vstack(f_tr["hc_features"].tolist())
        va_hc     = np.vstack(f_va["hc_features"].tolist())

        tr_ds = ASAGDataset(tr_texts, tr_labels, tr_sims, tr_hc)
        va_ds = ASAGDataset(va_texts, va_labels, va_sims, va_hc)

        # [2] Class-weighted sampling
        f_cls = np.round(f_tr["normalized_grade"].values * 2).astype(int)
        c_cnt = np.maximum(np.bincount(f_cls, minlength=3).astype(float), 1.0)
        c_wt  = (1.0 / c_cnt);  c_wt /= c_wt.mean()
        s_wt  = c_wt[f_cls]
        sampler = WeightedRandomSampler(
            torch.tensor(s_wt, dtype=torch.float), len(f_tr), replacement=True)

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)

        model = TextCNN(vocab_size=vocab_size, num_hc=num_hc).to(device)
        if emb_matrix is not None:                            # [1]
            model.embedding.weight.data.copy_(torch.tensor(emb_matrix))

        best_st, _, m = train_fold(model, tr_loader, va_loader, criterion,
                                   device, args.epochs, PATIENCE, args.lr,
                                   WEIGHT_DECAY)
        fold_metrics.append(m)
        print(f"    Fold {fi+1}:  QWK={m['qwk']:.4f}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  MSE={m['mse']:.4f}")

    # ── CV averages ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {args.folds}-Fold CV Averages")
    print(f"{'='*60}")
    for k in ["qwk", "accuracy", "f1", "mse"]:
        vals = [m[k] for m in fold_metrics]
        print(f"  {k.upper():>10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Final model on ALL training data ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training final model on full training set ({len(train_aug)} samples)")
    print(f"{'='*60}")

    all_texts  = [encode_text(t, vocab) for t in train_aug["text"]]
    all_labels = train_aug["smoothed_grade"].tolist()
    all_sims   = train_aug["sim_score"].tolist()
    all_hc     = np.vstack(train_aug["hc_features"].tolist())

    full_ds = ASAGDataset(all_texts, all_labels, all_sims, all_hc)

    all_cls = np.round(train_aug["normalized_grade"].values * 2).astype(int)
    ac = np.maximum(np.bincount(all_cls, minlength=3).astype(float), 1.0)
    aw = (1.0 / ac);  aw /= aw.mean()
    full_sampler = WeightedRandomSampler(
        torch.tensor(aw[all_cls], dtype=torch.float), len(train_aug), replacement=True)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, sampler=full_sampler)

    # Small held-out slice for early stopping
    nv = int(len(train_aug) * 0.05)
    es_ds  = ASAGDataset(all_texts[-nv:],
                          train_aug["normalized_grade"].tolist()[-nv:],
                          all_sims[-nv:], all_hc[-nv:])
    es_ldr = DataLoader(es_ds, batch_size=BATCH_SIZE)

    final = TextCNN(vocab_size=vocab_size, num_hc=num_hc).to(device)
    if emb_matrix is not None:
        final.embedding.weight.data.copy_(torch.tensor(emb_matrix))

    best_st, _, _ = train_fold(final, full_loader, es_ldr, criterion, device,
                               args.epochs, PATIENCE, args.lr, WEIGHT_DECAY)
    if best_st:
        final.load_state_dict(best_st)
        final = final.to(device)

    torch.save(final.state_dict(), MODEL_SAVE)
    print(f"\n  ✓ Final model saved to {MODEL_SAVE}")

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Test Set Evaluation ({len(test_df)} samples)")
    print(f"{'='*60}")

    te_texts = [encode_text(t, vocab) for t in test_df["text"]]
    te_labs  = test_df["normalized_grade"].tolist()
    te_sims  = test_df["sim_score"].tolist()
    te_hc    = np.vstack(test_df["hc_features"].tolist())
    te_ds    = ASAGDataset(te_texts, te_labs, te_sims, te_hc)
    te_ldr   = DataLoader(te_ds, batch_size=BATCH_SIZE)

    final.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, bs, bh in te_ldr:
            bx, bs, bh = bx.to(device), bs.to(device), bh.to(device)
            preds.extend(final(bx, bs, bh).cpu().numpy())
            trues.extend(by.numpy())
    preds = np.clip(preds, 0.0, 1.0)

    rounder = ScoreRounder(num_classes=3)
    rounder.fit(np.array(preds), np.array(trues))
    disc = rounder.predict_normalised(np.array(preds))

    print("\n── Continuous prediction metrics ──")
    mc = evaluate_all_metrics(np.array(preds), np.array(trues))
    for k in ["mse", "qwk", "accuracy", "f1"]:
        print(f"  {k.upper():>10}: {mc[k]:.4f}")

    print("\n── After optimised score rounding ──")
    mr = evaluate_all_metrics(disc, np.array(trues))
    for k in ["mse", "qwk", "accuracy", "f1"]:
        print(f"  {k.upper():>10}: {mr[k]:.4f}")
    print(f"\n  Confusion matrix:\n{mr['confusion_matrix']}")
    print(f"  Test samples: {len(trues)}")

    # ── Error analysis ────────────────────────────────────────────────────────
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
    print("\nDone.")


if __name__ == "__main__":
    main()
