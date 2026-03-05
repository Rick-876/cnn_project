"""
hybrid_pipeline.py
Hybrid ASAG model: Transformer (DeBERTa-v3-small / RoBERTa) + handcrafted
linguistic features fed into a shared FC regression head.

Architecture:
    ┌──────────────────────────────────────────┐
    │  Transformer backbone (DeBERTa / RoBERTa)│
    │  CLS-token hidden state  [B, H]           │
    └──────────────────────┬───────────────────┘
                           │
    ┌──────────────────────┴────────────────────┐
    │  Handcrafted feature vector  [B, 27]       │
    │  (feature_engineering.py)                 │
    └──────────────────────┬───────────────────┘
                           ↓
              Concatenate: [B, H + 27]
                           ↓
              FC(512) → ReLU → Dropout
                           ↓
              FC(128) → ReLU → Dropout
                           ↓
              FC(1) → Sigmoid → score in [0, 1]

Training strategy:
    - Stratified 5-Fold cross-validation
    - CombinedLoss: 30% Huber + 70% QWK
    - Per-sample class-weighting for imbalanced grades
    - AdamW with weight decay + linear warmup + cosine decay
    - Early stopping on fold validation QWK
    - ScoreRounder post-processing after each fold

Usage:
    python hybrid_pipeline.py                  # train with defaults
    python hybrid_pipeline.py --folds 3        # 3-fold CV
    python hybrid_pipeline.py --model roberta  # use RoBERTa backbone
    python hybrid_pipeline.py --epochs 5       # fewer epochs (quick test)
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (mean_squared_error, cohen_kappa_score,
                              accuracy_score, f1_score, confusion_matrix)
from tqdm import tqdm

from feature_engineering import batch_extract_features, FEATURE_NAMES
from training_utils import (CombinedLoss, compute_class_weights,
                             ScoreRounder, evaluate_all_metrics)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULTS = {
    'model_name': 'microsoft/deberta-v3-small',  # fallback: roberta-base
    'max_length': 256,
    'batch_size': 16,
    'epochs': 8,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'dropout': 0.3,
    'warmup_ratio': 0.1,
    'folds': 5,
    'loss_alpha': 0.3,          # 30% Huber + 70% QWK
    'num_classes': 3,
    'data_path': 'asag2024_all.csv',
    'save_dir': 'hybrid_models',
    'seed': 42,
}

MODEL_ALIASES = {
    'deberta': 'microsoft/deberta-v3-small',
    'roberta': 'roberta-base',
    'bert':    'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
}

NUM_HANDCRAFTED = len(FEATURE_NAMES)  # 27


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HybridASAGDataset(Dataset):
    """
    Combines tokenised text (for transformer) with handcrafted features.
    """

    def __init__(self, student_answers, reference_answers, labels,
                 tokenizer, max_length=256):
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Tokenise all pairs together: [SEP]-separated question+student text
        combined = [f"{s} [SEP] {r}" for s, r in
                    zip(student_answers, reference_answers)]
        self.encodings = tokenizer(
            combined,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Pre-compute handcrafted features
        print("  Extracting handcrafted features …")
        self.hc_features = torch.tensor(
            batch_extract_features(student_answers, reference_answers),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'hc_features':    self.hc_features[idx],
            'labels':         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridGrader(nn.Module):
    """
    Transformer backbone + handcrafted features → regression head.
    """

    def __init__(self, transformer_name: str, num_handcrafted: int,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size

        # Projection from transformer hidden to shared space
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Feature pathway for handcrafted features
        self.feat_proj = nn.Sequential(
            nn.Linear(num_handcrafted, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, hc_features):
        out = self.transformer(input_ids=input_ids,
                               attention_mask=attention_mask)
        # CLS token representation
        cls = out.last_hidden_state[:, 0, :]           # [B, H]
        proj = self.proj(cls)                           # [B, hidden_dim]
        feat = self.feat_proj(hc_features)              # [B, 64]
        fused = torch.cat([proj, feat], dim=1)          # [B, hidden_dim+64]
        return self.head(fused).squeeze(1)              # [B]


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion,
                device, sample_weights=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        hc        = batch['hc_features'].to(device)
        labels    = batch['labels'].to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attn_mask, hc)
        loss  = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        hc        = batch['hc_features'].to(device)
        labels    = batch['labels'].to(device)

        preds = model(input_ids, attn_mask, hc)
        loss  = criterion(preds, labels)
        total_loss += loss.item()

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    preds_arr  = np.clip(np.array(preds_all), 0, 1)
    labels_arr = np.array(labels_all)
    metrics = evaluate_all_metrics(preds_arr, labels_arr)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return preds_arr, labels_arr, metrics


# ---------------------------------------------------------------------------
# K-Fold training
# ---------------------------------------------------------------------------

def run_kfold_training(cfg: dict):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading dataset …")
    try:
        from datasets import load_dataset
        raw = load_dataset('Meyerger/ASAG2024')
        df  = raw['train'].to_pandas()
    except Exception:
        df = pd.read_csv(cfg['data_path'])

    required = ["question", "reference_answer", "provided_answer",
                "normalized_grade"]
    if not all(c in df.columns for c in required):
        df.columns = required[:len(df.columns)]
    df = df[required].fillna(
        {"question": "", "reference_answer": "", "provided_answer": "",
         "normalized_grade": 0.0}
    )

    # Build input text: question context + student answer
    student_texts = (
        "Question: " + df["question"] + " Student answer: " + df["provided_answer"]
    ).tolist()
    reference_texts = df["reference_answer"].tolist()
    labels = df["normalized_grade"].values.astype(np.float32)

    # Stratified bins for K-Fold split
    strat_bins = np.round(labels * (cfg['num_classes'] - 1)).astype(int)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {cfg['model_name']} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    # ── K-Fold loop ────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=cfg['folds'], shuffle=True,
                          random_state=cfg['seed'])
    fold_results = []
    all_oof_preds  = np.zeros(len(labels))
    all_oof_labels = labels.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(student_texts, strat_bins)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {cfg['folds']}")
        print(f"{'='*60}")

        train_students  = [student_texts[i]   for i in train_idx]
        train_refs      = [reference_texts[i] for i in train_idx]
        train_labels    = labels[train_idx]

        val_students    = [student_texts[i]   for i in val_idx]
        val_refs        = [reference_texts[i] for i in val_idx]
        val_labels      = labels[val_idx]

        # Datasets
        print("Building train dataset …")
        train_ds = HybridASAGDataset(train_students, train_refs,
                                     train_labels.tolist(), tokenizer,
                                     cfg['max_length'])
        print("Building val dataset …")
        val_ds   = HybridASAGDataset(val_students,   val_refs,
                                     val_labels.tolist(),   tokenizer,
                                     cfg['max_length'])

        # Weighted sampler to handle class imbalance
        sample_wts = compute_class_weights(train_labels, cfg['num_classes'])
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_wts, dtype=torch.float32),
            num_samples=len(train_ds),
            replacement=True
        )

        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                                  sampler=sampler)
        val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                                  shuffle=False)

        # Model, optimizer, scheduler
        model = HybridGrader(
            transformer_name=cfg['model_name'],
            num_handcrafted=NUM_HANDCRAFTED,
            dropout=cfg['dropout']
        ).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay']
        )

        total_steps  = len(train_loader) * cfg['epochs']
        warmup_steps = int(total_steps * cfg['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        criterion = CombinedLoss(
            alpha=cfg['loss_alpha'],
            num_classes=cfg['num_classes']
        )

        # ── Training loop ──────────────────────────────────────────────
        best_qwk       = -1.0
        patience_left  = 3
        best_state     = None
        best_preds     = None

        for epoch in range(cfg['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer,
                                     scheduler, criterion, device)

            val_preds, val_labels_out, val_metrics = evaluate_epoch(
                model, val_loader, criterion, device)

            qwk = val_metrics['qwk']
            print(f"  Epoch {epoch+1:02d} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | "
                  f"QWK={qwk:.4f} | "
                  f"Acc={val_metrics['accuracy']:.4f} | "
                  f"F1={val_metrics['f1']:.4f}")

            if qwk > best_qwk:
                best_qwk     = qwk
                patience_left = 3
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                best_preds    = val_preds.copy()
                model_path = os.path.join(cfg['save_dir'],
                                          f"hybrid_fold{fold_idx}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"    ✓ Best model saved (QWK={best_qwk:.4f})")
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Load best weights for post-processing
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)

        # Optimise rounding thresholds on this fold's validation set
        rounder = ScoreRounder(num_classes=cfg['num_classes'])
        rounder.fit(best_preds, val_labels_out)

        # Final fold metrics with optimised rounding
        disc_preds = rounder.predict_normalised(best_preds)
        final_metrics = evaluate_all_metrics(disc_preds, val_labels_out,
                                              cfg['num_classes'])
        print(f"\n  Fold {fold_idx+1} final | "
              f"QWK={final_metrics['qwk']:.4f} | "
              f"Acc={final_metrics['accuracy']:.4f} | "
              f"F1={final_metrics['f1']:.4f} | "
              f"MSE={final_metrics['mse']:.4f}")

        # Save rounder thresholds
        np.save(os.path.join(cfg['save_dir'],
                             f'rounder_fold{fold_idx}.npy'),
                rounder.thresholds)

        all_oof_preds[val_idx] = best_preds
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_qwk': float(best_qwk),
            'final_qwk': float(final_metrics['qwk']),
            'accuracy':    float(final_metrics['accuracy']),
            'f1':          float(final_metrics['f1']),
            'mse':         float(final_metrics['mse']),
        })

    # ── Overall OOF metrics ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("OVERALL OUT-OF-FOLD METRICS")
    print(f"{'='*60}")
    oof_metrics = evaluate_all_metrics(
        np.clip(all_oof_preds, 0, 1), all_oof_labels, cfg['num_classes'])

    print(f"  OOF QWK:      {oof_metrics['qwk']:.4f}")
    print(f"  OOF Accuracy: {oof_metrics['accuracy']:.4f}")
    print(f"  OOF F1:       {oof_metrics['f1']:.4f}")
    print(f"  OOF MSE:      {oof_metrics['mse']:.4f}")
    print(f"\nConfusion Matrix:\n{oof_metrics['confusion_matrix']}")

    print(f"\nPer-fold summary:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: QWK={r['final_qwk']:.4f} | "
              f"Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | "
              f"MSE={r['mse']:.4f}")

    avg_qwk = np.mean([r['final_qwk'] for r in fold_results])
    avg_acc = np.mean([r['accuracy']  for r in fold_results])
    avg_f1  = np.mean([r['f1']        for r in fold_results])
    avg_mse = np.mean([r['mse']        for r in fold_results])

    print(f"\nMean across folds:")
    print(f"  QWK:      {avg_qwk:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  F1:       {avg_f1:.4f}")
    print(f"  MSE:      {avg_mse:.4f}")

    # Save summary
    summary = {
        'config': {k: v for k, v in cfg.items() if isinstance(v, (str, int, float))},
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
        }
    }
    summary_path = os.path.join(cfg['save_dir'], 'hybrid_cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid ASAG training pipeline")
    p.add_argument('--model', default='deberta',
                   choices=list(MODEL_ALIASES.keys()) + list(MODEL_ALIASES.values()),
                   help='Transformer backbone alias or HuggingFace model name')
    p.add_argument('--folds',   type=int,   default=DEFAULTS['folds'])
    p.add_argument('--epochs',  type=int,   default=DEFAULTS['epochs'])
    p.add_argument('--batch',   type=int,   default=DEFAULTS['batch_size'])
    p.add_argument('--lr',      type=float, default=DEFAULTS['lr'])
    p.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    p.add_argument('--alpha',   type=float, default=DEFAULTS['loss_alpha'],
                   help='Weight for Huber loss (1-alpha for QWK loss)')
    p.add_argument('--save_dir', default=DEFAULTS['save_dir'])
    p.add_argument('--data',    default=DEFAULTS['data_path'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_name = MODEL_ALIASES.get(args.model, args.model)

    cfg = dict(DEFAULTS)
    cfg.update({
        'model_name':  model_name,
        'folds':       args.folds,
        'epochs':      args.epochs,
        'batch_size':  args.batch,
        'lr':          args.lr,
        'dropout':     args.dropout,
        'loss_alpha':  args.alpha,
        'save_dir':    args.save_dir,
        'data_path':   args.data,
    })

    print("Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    run_kfold_training(cfg)
