"""
train_cpu.py
CPU-optimised training pipeline for HybridGrader.

Strategy: pre-compute transformer (DistilBERT) CLS embeddings in one pass
(forward-only, no grad), then train only the lightweight FC scoring head
on the frozen embeddings + handcrafted features. This avoids back-propagating
through the transformer, cutting training time from hours to minutes.

Steps:
    1. Load data and extract handcrafted features (27-dim)
    2. Extract DistilBERT CLS embeddings for all samples (~15 min)
    3. Train FC scoring head with K-fold CV on [CLS ‖ HC] features
    4. Report metrics (QWK, Accuracy, F1, MSE)
    5. Save best model + error analysis

Usage:
    python train_cpu.py                    # defaults
    python train_cpu.py --epochs 20        # more epochs (still fast)
    python train_cpu.py --folds 5          # 5-fold CV
"""

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from feature_engineering import batch_extract_features, FEATURE_NAMES
from training_utils import (CombinedLoss, compute_class_weights,
                             ScoreRounder, evaluate_all_metrics)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    'encoder_name': 'distilbert-base-uncased',
    'max_length': 128,
    'extract_batch': 64,       # batch for embedding extraction (forward only)
    'train_batch': 128,        # batch for FC training (very fast)
    'epochs': 20,              # FC-only = cheap, so we can run more
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'folds': 3,
    'loss_alpha': 0.3,
    'num_classes': 3,
    'data_path': 'asag2024_all.csv',
    'save_dir': 'hybrid_models',
    'seed': 42,
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    texts: list,
    encoder_name: str,
    max_length: int = 128,
    batch_size: int = 64,
    cache_path: str = None,
) -> np.ndarray:
    """Extract CLS embeddings from a pretrained transformer.

    Args:
        texts:        List of input texts.
        encoder_name: HuggingFace model name.
        max_length:   Max token sequence length.
        batch_size:   Batch size for extraction.
        cache_path:   If set, save/load embeddings to this .npy file.

    Returns:
        np.ndarray of shape (N, hidden_size).
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"  Loading encoder: {encoder_name} …")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name)
    model.eval()

    hidden_size = model.config.hidden_size
    all_embeds = np.zeros((len(texts), hidden_size), dtype=np.float32)

    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"  Extracting {len(texts)} embeddings ({n_batches} batches) …")

    for i in tqdm(range(0, len(texts), batch_size), desc="  embed",
                  total=n_batches):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
        )
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].numpy()  # [B, H]
        all_embeds[i:i + len(batch_texts)] = cls

    # Cache for reuse
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, all_embeds)
        print(f"  Embeddings cached → {cache_path}")

    return all_embeds


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: FC scoring head
# ──────────────────────────────────────────────────────────────────────────────

class ScoringHead(nn.Module):
    """Lightweight FC scoring head on pre-computed features.

    Input: [CLS_embedding ‖ handcrafted_features]  → score ∈ [0,1]
    """

    def __init__(self, embed_dim: int, num_hc: int = 27,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        input_dim = embed_dim + num_hc

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class EmbeddingDataset(Dataset):
    """Dataset of pre-computed embeddings + handcrafted features."""

    def __init__(self, embeddings: np.ndarray, hc_features: np.ndarray,
                 labels: np.ndarray):
        self.x = torch.tensor(
            np.concatenate([embeddings, hc_features], axis=1),
            dtype=torch.float32,
        )
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        preds.extend(pred.numpy())
        labels.extend(y.numpy())
    preds = np.clip(np.array(preds), 0, 1)
    labels = np.array(labels)
    metrics = evaluate_all_metrics(preds, labels)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return preds, labels, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_training(cfg: dict):
    """Run full CPU-optimised training pipeline."""
    os.makedirs(cfg['save_dir'], exist_ok=True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    start_time = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading dataset …")
    df = pd.read_csv(cfg['data_path'])
    if 'Question' in df.columns:
        df = df.rename(columns={
            'Question': 'question',
            'Student Answer': 'provided_answer',
            'Reference Answer': 'reference_answer',
            'Human Score/Grade': 'normalized_grade',
        })
    if df['normalized_grade'].max() > 1.0:
        df['normalized_grade'] /= df['normalized_grade'].max()

    student_texts = df['provided_answer'].fillna('').tolist()
    reference_texts = df['reference_answer'].fillna('').tolist()
    combined_texts = [
        f"Question: {q} Student answer: {a}"
        for q, a in zip(df['question'].fillna(''), df['provided_answer'].fillna(''))
    ]
    labels = df['normalized_grade'].values.astype(np.float32)
    strat_bins = np.round(labels * (cfg['num_classes'] - 1)).astype(int)

    print(f"  {len(labels)} samples loaded")

    # ── Extract handcrafted features ───────────────────────────────────────
    print("\nExtracting handcrafted features …")
    hc_features = batch_extract_features(student_texts, reference_texts)
    print(f"  Shape: {hc_features.shape}")

    # ── Extract transformer embeddings ─────────────────────────────────────
    print("\nExtracting transformer embeddings …")
    cache = os.path.join(cfg['save_dir'], 'embeddings_cache.npy')
    embeddings = extract_embeddings(
        combined_texts,
        cfg['encoder_name'],
        max_length=cfg['max_length'],
        batch_size=cfg['extract_batch'],
        cache_path=cache,
    )
    embed_dim = embeddings.shape[1]
    print(f"  Shape: {embeddings.shape} (dim={embed_dim})")

    # ── K-Fold training ────────────────────────────────────────────────────
    print(f"\nStarting {cfg['folds']}-fold CV training …")
    skf = StratifiedKFold(n_splits=cfg['folds'], shuffle=True,
                          random_state=cfg['seed'])
    fold_results = []
    all_oof_preds = np.zeros(len(labels))

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(embeddings, strat_bins)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {cfg['folds']}")
        print(f"{'='*60}")

        # Split data
        train_ds = EmbeddingDataset(
            embeddings[train_idx], hc_features[train_idx], labels[train_idx])
        val_ds = EmbeddingDataset(
            embeddings[val_idx], hc_features[val_idx], labels[val_idx])

        # Weighted sampler
        sample_wts = compute_class_weights(labels[train_idx], cfg['num_classes'])
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_wts, dtype=torch.float32),
            num_samples=len(train_ds),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg['train_batch'], sampler=sampler)
        val_loader = DataLoader(
            val_ds, batch_size=cfg['train_batch'], shuffle=False)

        # Model
        model = ScoringHead(
            embed_dim=embed_dim,
            num_hc=len(FEATURE_NAMES),
            dropout=cfg['dropout'],
        )

        optimizer = AdamW(model.parameters(), lr=cfg['lr'],
                          weight_decay=cfg['weight_decay'])
        criterion = CombinedLoss(alpha=cfg['loss_alpha'],
                                 num_classes=cfg['num_classes'])

        # Training
        best_qwk = -1.0
        patience = 5
        patience_left = patience
        best_state = None
        best_preds = None

        for epoch in range(cfg['epochs']):
            t_loss = train_epoch(model, train_loader, optimizer, criterion)
            v_preds, v_labels, v_metrics = eval_epoch(
                model, val_loader, criterion)

            qwk = v_metrics['qwk']
            if (epoch + 1) % 5 == 0 or epoch == 0 or qwk > best_qwk:
                print(f"  Epoch {epoch+1:02d} | "
                      f"train_loss={t_loss:.4f} | "
                      f"val_loss={v_metrics['loss']:.4f} | "
                      f"QWK={qwk:.4f} | "
                      f"Acc={v_metrics['accuracy']:.4f} | "
                      f"F1={v_metrics['f1']:.4f}")

            if qwk > best_qwk:
                best_qwk = qwk
                patience_left = patience
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_preds = v_preds.copy()
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Post-processing with ScoreRounder
        if best_state is not None:
            model.load_state_dict(best_state)

        rounder = ScoreRounder(num_classes=cfg['num_classes'])
        rounder.fit(best_preds, v_labels)
        rounded_preds = rounder.predict_normalised(best_preds)
        final_metrics = evaluate_all_metrics(
            rounded_preds, v_labels, cfg['num_classes'])

        print(f"\n  Fold {fold_idx+1} final | "
              f"QWK={final_metrics['qwk']:.4f} | "
              f"Acc={final_metrics['accuracy']:.4f} | "
              f"F1={final_metrics['f1']:.4f} | "
              f"MSE={final_metrics['mse']:.4f}")

        # Save fold model
        model_path = os.path.join(
            cfg['save_dir'], f'scoring_head_fold{fold_idx}.pth')
        torch.save(best_state, model_path)
        np.save(os.path.join(
            cfg['save_dir'], f'rounder_fold{fold_idx}.npy'),
            rounder.thresholds)

        all_oof_preds[val_idx] = best_preds
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_qwk': float(best_qwk),
            'final_qwk': float(final_metrics['qwk']),
            'accuracy': float(final_metrics['accuracy']),
            'f1': float(final_metrics['f1']),
            'mse': float(final_metrics['mse']),
        })

    # ── Overall OOF metrics ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("OVERALL OUT-OF-FOLD METRICS")
    print(f"{'='*60}")
    oof_metrics = evaluate_all_metrics(
        np.clip(all_oof_preds, 0, 1), labels, cfg['num_classes'])

    print(f"  OOF QWK:      {oof_metrics['qwk']:.4f}")
    print(f"  OOF Accuracy: {oof_metrics['accuracy']:.4f}")
    print(f"  OOF F1:       {oof_metrics['f1']:.4f}")
    print(f"  OOF MSE:      {oof_metrics['mse']:.4f}")
    print(f"\n  Time: {elapsed/60:.1f} min")

    print(f"\nPer-fold summary:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: QWK={r['final_qwk']:.4f} | "
              f"Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | "
              f"MSE={r['mse']:.4f}")

    avg_qwk = np.mean([r['final_qwk'] for r in fold_results])
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_mse = np.mean([r['mse'] for r in fold_results])

    print(f"\nMean across folds:")
    print(f"  QWK:      {avg_qwk:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  F1:       {avg_f1:.4f}")
    print(f"  MSE:      {avg_mse:.4f}")

    # ── Error analysis ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")
    try:
        from error_analysis import ErrorAnalyser
        analyser = ErrorAnalyser(num_classes=cfg['num_classes'])
        analyser.fit(
            np.clip(all_oof_preds, 0, 1), labels,
            texts=student_texts,
            reference_texts=reference_texts,
        )
        analyser.report(verbose=True)
        analyser.save(os.path.join(cfg['save_dir'], 'error_report.json'))
    except Exception as e:
        print(f"  Error analysis failed: {e}")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        'config': {k: v for k, v in cfg.items()
                   if isinstance(v, (str, int, float))},
        'fold_results': fold_results,
        'oof_metrics': {
            'qwk': float(oof_metrics['qwk']),
            'accuracy': float(oof_metrics['accuracy']),
            'f1': float(oof_metrics['f1']),
            'mse': float(oof_metrics['mse']),
        },
        'mean_metrics': {
            'qwk': float(avg_qwk),
            'accuracy': float(avg_acc),
            'f1': float(avg_f1),
            'mse': float(avg_mse),
        },
        'training_time_min': round(elapsed / 60, 1),
    }
    summary_path = os.path.join(cfg['save_dir'], 'cpu_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CPU-optimised hybrid ASAG training")
    p.add_argument('--encoder', default=DEFAULTS['encoder_name'],
                   help='Transformer encoder name')
    p.add_argument('--folds', type=int, default=DEFAULTS['folds'])
    p.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    p.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    p.add_argument('--batch', type=int, default=DEFAULTS['train_batch'])
    p.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    p.add_argument('--data', default=DEFAULTS['data_path'])
    p.add_argument('--save_dir', default=DEFAULTS['save_dir'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update({
        'encoder_name': args.encoder,
        'folds': args.folds,
        'epochs': args.epochs,
        'lr': args.lr,
        'train_batch': args.batch,
        'dropout': args.dropout,
        'data_path': args.data,
        'save_dir': args.save_dir,
    })

    print("=" * 60)
    print("CPU-OPTIMISED HYBRID ASAG TRAINING")
    print("=" * 60)
    print("Strategy: pre-compute DistilBERT embeddings → train FC head only")
    print()
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    run_training(cfg)
