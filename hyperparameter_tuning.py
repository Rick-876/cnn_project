"""
hyperparameter_tuning.py
Optuna-based hyperparameter optimisation for the ASAG models.

Supports tuning:
1. CNN pipeline (asag_cnn_pipeline.py) hyperparameters
2. Hybrid transformer pipeline (hybrid_pipeline.py) hyperparameters

Usage:
    # Tune the CNN model (fast, CPU-friendly)
    python hyperparameter_tuning.py --target cnn --trials 30

    # Tune the hybrid transformer (requires GPU, ~20 min per trial)
    python hyperparameter_tuning.py --target hybrid --trials 10 --model deberta

    # Continue a previous study (resumes from storage)
    python hyperparameter_tuning.py --target cnn --trials 50 --storage optuna_cnn.db

The best hyperparameters are saved to best_params_<target>.json.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter tuning.\n"
        "Install with: pip install optuna"
    )

from training_utils import evaluate_all_metrics, ScoreRounder


# ---------------------------------------------------------------------------
# CNN objective
# ---------------------------------------------------------------------------

def cnn_objective(trial, df: pd.DataFrame, device: str) -> float:
    """
    Optuna objective function for the TextCNN pipeline.
    Returns negative QWK (Optuna minimises by default).
    """
    import re
    import torch.nn as nn
    import torch.nn.functional as F
    from collections import Counter
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from training_utils import CombinedLoss

    # ── Suggest hyperparameters ────────────────────────────────────────────
    embed_dim    = trial.suggest_categorical('embed_dim',    [100, 200, 300])
    num_filters  = trial.suggest_categorical('num_filters',  [64, 128, 256])
    filter_sizes = trial.suggest_categorical('filter_sizes', ['2,3,4', '3,4,5', '2,3,4,5'])
    dropout      = trial.suggest_float('dropout',     0.1, 0.6)
    lr           = trial.suggest_float('lr',          1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [16, 32, 64])
    loss_alpha   = trial.suggest_float('loss_alpha',  0.0, 1.0)
    max_len      = trial.suggest_categorical('max_len', [64, 100, 150])

    filter_sizes_list = [int(x) for x in filter_sizes.split(',')]

    # ── Data prep ──────────────────────────────────────────────────────────
    STOPWORDS = {
        "what","is","the","a","an","of","in","to","and","or","for","on","at",
        "with","this","that","are","it","as","be","from","by","was","were",
    }

    def tokenize(text: str):
        return re.findall(r"\w+", text.lower())

    def ref_sim(student, reference):
        r = {w for w in tokenize(reference) if w not in STOPWORDS and len(w) > 2}
        s = {w for w in tokenize(student)   if w not in STOPWORDS and len(w) > 2}
        return len(s & r) / max(len(r), 1)

    df2 = df.copy()
    df2['text'] = ("Question: " + df2['question'] + " Reference: "
                   + df2['reference_answer'] + " Student: " + df2['provided_answer'])
    df2['sim'] = df2.apply(
        lambda r: ref_sim(r['provided_answer'], r['reference_answer']), axis=1)

    train_df, val_df = train_test_split(df2, test_size=0.15,
                                        random_state=42, shuffle=True)

    all_words = [w for t in df2['text'] for w in tokenize(t)]
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(all_words).most_common())}
    vocab_size = len(vocab) + 1

    def encode(text):
        ids = [vocab.get(t, 0) for t in tokenize(text)]
        ids = ids[:max_len] + [0] * max(max_len - len(ids), 0)
        return ids

    # ── Model ──────────────────────────────────────────────────────────────
    class TunedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv2d(1, num_filters, (fs, embed_dim))
                for fs in filter_sizes_list
            ])
            self.drop = nn.Dropout(dropout)
            self.fc   = nn.Linear(num_filters * len(filter_sizes_list) + 1, 1)

        def forward(self, x, sim):
            x = self.emb(x).unsqueeze(1)
            pooled = [F.max_pool1d(F.relu(c(x)).squeeze(3),
                                   F.relu(c(x)).squeeze(3).size(2)).squeeze(2)
                      for c in self.convs]
            cat = torch.cat(pooled + [sim.unsqueeze(1)], dim=1)
            return torch.sigmoid(self.fc(self.drop(cat))).squeeze(1)

    class DS(torch.utils.data.Dataset):
        def __init__(self, sub):
            self.x = torch.tensor([encode(t) for t in sub['text']], dtype=torch.long)
            self.y = torch.tensor(sub['normalized_grade'].tolist(), dtype=torch.float)
            self.s = torch.tensor(sub['sim'].tolist(), dtype=torch.float)
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.x[i], self.y[i], self.s[i]

    train_loader = DataLoader(DS(train_df), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DS(val_df),   batch_size=batch_size)

    model     = TunedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = CombinedLoss(alpha=loss_alpha, num_classes=3)

    # Train for up to 10 quick epochs
    best_qwk = -1.0
    patience = 3

    for epoch in range(10):
        model.train()
        for bx, by, bs in train_loader:
            bx, by, bs = bx.to(device), by.to(device), bs.to(device)
            optimizer.zero_grad()
            criterion(model(bx, bs), by).backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, by, bs in val_loader:
                preds.extend(model(bx.to(device), bs.to(device)).cpu().numpy())
                trues.extend(by.numpy())

        preds = np.clip(preds, 0, 1)
        metrics = evaluate_all_metrics(np.array(preds), np.array(trues))
        qwk = metrics['qwk']
        trial.report(qwk, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if qwk > best_qwk:
            best_qwk = qwk
            patience = 3
        else:
            patience -= 1
            if patience == 0:
                break

    return -best_qwk   # minimise negative QWK


# ---------------------------------------------------------------------------
# Hybrid / transformer objective
# ---------------------------------------------------------------------------

def hybrid_objective(trial, df: pd.DataFrame, model_name: str,
                     device: str) -> float:
    """
    Optuna objective for the hybrid transformer pipeline.
    Runs a single 2-fold CV for speed; returns negative mean QWK.
    """
    from hybrid_pipeline import (HybridASAGDataset, HybridGrader,
                                  train_epoch, evaluate_epoch)
    from training_utils import CombinedLoss, compute_class_weights
    from torch.utils.data import WeightedRandomSampler
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.model_selection import StratifiedKFold
    import torch

    # ── Suggest hyperparameters ────────────────────────────────────────────
    lr          = trial.suggest_float('lr',       1e-5, 5e-5, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [8, 16])
    dropout     = trial.suggest_float('dropout',   0.1, 0.5)
    loss_alpha  = trial.suggest_float('loss_alpha', 0.0, 0.6)
    weight_decay= trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    max_length  = trial.suggest_categorical('max_length', [128, 192, 256])

    student_texts = ("Question: " + df['question'] + " Student: "
                     + df['provided_answer']).tolist()
    reference_texts = df['reference_answer'].tolist()
    labels = df['normalized_grade'].values.astype(np.float32)
    strat  = np.round(labels * 2).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    fold_qwks = []
    for fold, (ti, vi) in enumerate(skf.split(student_texts, strat)):
        t_stu  = [student_texts[i] for i in ti]
        t_ref  = [reference_texts[i] for i in ti]
        t_lab  = labels[ti].tolist()

        v_stu  = [student_texts[i] for i in vi]
        v_ref  = [reference_texts[i] for i in vi]
        v_lab  = labels[vi].tolist()

        train_ds = HybridASAGDataset(t_stu, t_ref, t_lab, tokenizer, max_length)
        val_ds   = HybridASAGDataset(v_stu, v_ref, v_lab, tokenizer, max_length)

        sw = compute_class_weights(labels[ti])
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float32),
                                        len(train_ds), replacement=True)

        train_ldr = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        from feature_engineering import FEATURE_NAMES
        model = HybridGrader(model_name, len(FEATURE_NAMES),
                              dropout=dropout).to(device)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps   = len(train_ldr) * 3
        warmup_steps  = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps,
                                                    total_steps)
        criterion = CombinedLoss(alpha=loss_alpha, num_classes=3)

        best_qwk = -1.0
        for epoch in range(3):
            train_epoch(model, train_ldr, optimizer, scheduler, criterion, device)
            preds, labs, metrics = evaluate_epoch(model, val_ldr, criterion, device)
            qwk = metrics['qwk']
            trial.report(qwk, fold * 3 + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_qwk = max(best_qwk, qwk)

        fold_qwks.append(best_qwk)

    return -float(np.mean(fold_qwks))


# ---------------------------------------------------------------------------
# Main tuning runner
# ---------------------------------------------------------------------------

def run_tuning(target: str, n_trials: int, model_name: str = None,
               storage: str = None, data_path: str = 'asag2024_all.csv'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    print("Loading data …")
    try:
        from datasets import load_dataset
        raw = load_dataset('Meyerger/ASAG2024')
        df  = raw['train'].to_pandas()
    except Exception:
        df = pd.read_csv(data_path)

    required = ['question', 'reference_answer', 'provided_answer',
                'normalized_grade']
    df = df[required].fillna({"question": "", "reference_answer": "",
                               "provided_answer": "", "normalized_grade": 0.0})

    study_name = f"asag_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_url = f"sqlite:///{storage}" if storage else None

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_url,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=bool(storage),
    )

    if target == 'cnn':
        def objective(trial):
            return cnn_objective(trial, df, device)
    elif target == 'hybrid':
        if model_name is None:
            model_name = 'microsoft/deberta-v3-small'
        def objective(trial):
            return hybrid_objective(trial, df, model_name, device)
    else:
        raise ValueError(f"Unknown target: {target!r}. Use 'cnn' or 'hybrid'.")

    print(f"Starting Optuna study: {study_name}")
    print(f"  Trials:  {n_trials}")
    print(f"  Target:  {target}")
    print(f"  Storage: {storage_url or 'in-memory'}")

    study.optimize(objective, n_trials=n_trials, timeout=None,
                   show_progress_bar=True)

    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (neg-QWK): {trial.value:.4f}")
    print(f"  Best QWK:        {-trial.value:.4f}")
    print(f"  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    # Save best params
    save_path = f"best_params_{target}.json"
    result = {
        'best_qwk': float(-trial.value),
        'params': trial.params,
        'study_name': study_name,
        'n_trials': n_trials,
    }
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nBest params saved → {save_path}")

    # Importance plot (text summary)
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter importances:")
        for k, v in sorted(importance.items(), key=lambda x: -x[1]):
            bar = '█' * int(v * 40)
            print(f"  {k:25s} {bar} {v:.4f}")
    except Exception as e:
        print(f"Could not compute importances: {e}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Optuna hyperparameter tuning for ASAG')
    p.add_argument('--target',   default='cnn', choices=['cnn', 'hybrid'])
    p.add_argument('--trials',   type=int, default=30)
    p.add_argument('--model',    default=None,
                   help='Transformer name for hybrid target '
                        '(default: microsoft/deberta-v3-small)')
    p.add_argument('--storage',  default=None,
                   help='SQLite DB file for persistent study (optional)')
    p.add_argument('--data',     default='asag2024_all.csv')
    args = p.parse_args()

    run_tuning(target=args.target, n_trials=args.trials,
               model_name=args.model, storage=args.storage,
               data_path=args.data)
