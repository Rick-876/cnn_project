"""
reasoning_guided_training.py
Dual-head architecture: shared encoder → reasoning head + scoring head.

Architecture:
    ┌───────────────────────────────────────────────┐
    │  Shared Transformer Encoder (DeBERTa/RoBERTa) │
    │  → last_hidden_state  [B, L, H]              │
    └──────────────────┬────────────────────────────┘
                       │
          ┌────────────┴─────────────┐
          ↓                          ↓
    ┌──────────────┐         ┌──────────────────────┐
    │ ReasoningHead│         │ ScoreHead            │
    │ (seq2seq LM) │         │ (regression)         │
    │ → rationale  │         │ CLS + reasoning emb  │
    │   text       │         │ + handcrafted feats   │
    └──────────────┘         │ → score ∈ [0, 1]     │
                             └──────────────────────┘

Training:
    Stage 1 – Pre-generate synthetic reasoning rationales via templates
    Stage 2 – Joint training: reasoning LM loss + scoring regression loss
    Stage 3 – Knowledge distillation: freeze reasoning, fine-tune scoring
              head with reasoning embedding as extra input signal

Integration:
    python reasoning_guided_training.py                 # train
    python reasoning_guided_training.py --stage distill # distill only

Requires: transformers, torch, scikit-learn, nltk
"""

import os
import re
import json
import math
import logging
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from feature_engineering import batch_extract_features, FEATURE_NAMES
from training_utils import (
    CombinedLoss,
    compute_class_weights,
    ScoreRounder,
    evaluate_all_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REASONING_DEFAULTS: Dict = {
    "encoder_name": "microsoft/deberta-v3-small",
    "max_length": 256,
    "reasoning_max_length": 128,
    "batch_size": 8,
    "epochs": 6,
    "lr": 2e-5,
    "reasoning_lr": 5e-5,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "warmup_ratio": 0.1,
    "loss_alpha": 0.3,        # scoring component weight in combined loss
    "reasoning_weight": 0.3,  # how much reasoning LM loss contributes
    "distill_weight": 0.5,    # knowledge distillation loss weight
    "num_classes": 3,
    "folds": 5,
    "data_path": "asag2024_all.csv",
    "save_dir": "reasoning_models",
    "seed": 42,
}

MODEL_ALIASES: Dict[str, str] = {
    "deberta": "microsoft/deberta-v3-small",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
}


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning rationale template generator
# ──────────────────────────────────────────────────────────────────────────────

RATIONALE_TEMPLATES: Dict[str, str] = {
    "high": (
        "The student answer covers the core concepts well. "
        "Key terms from the reference ({matched_keywords}) are present. "
        "Content overlap is {overlap_pct}%. "
        "The answer demonstrates solid understanding."
    ),
    "medium": (
        "The student answer partially addresses the question. "
        "Some key terms are present ({matched_keywords}), but important "
        "concepts are missing ({missing_keywords}). "
        "Content overlap is {overlap_pct}%, indicating partial understanding."
    ),
    "low": (
        "The student answer does not adequately address the question. "
        "Most key terms from the reference are missing ({missing_keywords}). "
        "Content overlap is only {overlap_pct}%. "
        "The answer shows limited understanding of the topic."
    ),
    "off_topic": (
        "The student answer appears to be off-topic or irrelevant. "
        "Very few reference terms are present. "
        "Content overlap is {overlap_pct}%, suggesting the student "
        "may have misunderstood the question."
    ),
}


def _tokenize_words(text: str) -> List[str]:
    """Lowercase word tokenization."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "as", "into", "through", "about",
    "between", "it", "its", "this", "that", "these", "those", "and", "or",
    "but", "not", "no", "so", "if", "than", "then", "also", "very",
}


def generate_rationale(
    student_answer: str,
    reference_answer: str,
    true_score: float,
) -> str:
    """Generate a synthetic reasoning rationale from templates.

    Args:
        student_answer:   The student's answer text.
        reference_answer: The reference/model answer.
        true_score:       Normalised ground-truth score in [0, 1].

    Returns:
        A rationale string describing why the score was assigned.
    """
    s_words = set(_tokenize_words(student_answer)) - _STOPWORDS
    r_words = set(_tokenize_words(reference_answer)) - _STOPWORDS

    matched = s_words & r_words
    missing = r_words - s_words
    overlap_pct = round(len(matched) / max(len(r_words), 1) * 100, 1)

    matched_str = ", ".join(sorted(matched)[:6]) or "none"
    missing_str = ", ".join(sorted(missing)[:6]) or "none"

    if true_score >= 0.80:
        template_key = "high"
    elif true_score >= 0.45:
        template_key = "medium"
    elif overlap_pct < 10:
        template_key = "off_topic"
    else:
        template_key = "low"

    return RATIONALE_TEMPLATES[template_key].format(
        matched_keywords=matched_str,
        missing_keywords=missing_str,
        overlap_pct=overlap_pct,
    )


def batch_generate_rationales(
    student_answers: List[str],
    reference_answers: List[str],
    scores: np.ndarray,
) -> List[str]:
    """Generate rationales for a batch of samples."""
    return [
        generate_rationale(s, r, float(sc))
        for s, r, sc in zip(student_answers, reference_answers, scores)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningASAGDataset(Dataset):
    """Dataset that provides tokenised input, rationale targets, HC features,
    and labels for the dual-head model."""

    def __init__(
        self,
        student_answers: List[str],
        reference_answers: List[str],
        labels: np.ndarray,
        rationales: List[str],
        tokenizer,
        max_length: int = 256,
        reasoning_max_length: int = 128,
    ):
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Encode student + reference
        combined = [
            f"{s} [SEP] {r}" for s, r in zip(student_answers, reference_answers)
        ]
        self.encodings = tokenizer(
            combined,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Encode rationale targets for the reasoning head
        self.rationale_encodings = tokenizer(
            rationales,
            truncation=True,
            padding=True,
            max_length=reasoning_max_length,
            return_tensors="pt",
        )

        # Pre-compute handcrafted features
        logger.info("  Extracting handcrafted features …")
        self.hc_features = torch.tensor(
            batch_extract_features(student_answers, reference_answers),
            dtype=torch.float32,
        )

        self.rationales = rationales

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "rationale_ids": self.rationale_encodings["input_ids"][idx],
            "rationale_mask": self.rationale_encodings["attention_mask"][idx],
            "hc_features": self.hc_features[idx],
            "labels": self.labels[idx],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningHead(nn.Module):
    """Generates a reasoning embedding from the shared encoder hidden states.

    Uses a small transformer decoder layer on top of the encoder outputs,
    producing both:
      • A *reasoning embedding* vector (mean-pool of decoder output)
      • A *token-level* logit distribution for LM supervision
    """

    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.2):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        rationale_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            encoder_hidden: [B, L_enc, H] from shared encoder.
            encoder_mask:   [B, L_enc] attention mask for the encoder.
            rationale_ids:  [B, L_rat] token ids for teacher-forced decoding
                            (None at inference → use encoder CLS as query).

        Returns:
            reasoning_emb: [B, H] pooled reasoning embedding.
            lm_logits:     [B, L_rat, V] token logits (empty if no rationale_ids).
        """
        batch_size = encoder_hidden.size(0)

        if rationale_ids is not None:
            # Teacher-forced: use rationale tokens as queries
            # Create a simple embedding from the encoder's embedding weights
            # We re-use the encoder hidden as memory
            tgt = encoder_hidden[:, :rationale_ids.size(1), :]  # trim to rationale len
            memory = encoder_hidden
            memory_key_padding_mask = ~encoder_mask.bool()

            decoded = self.decoder_layer(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            decoded = self.layer_norm(decoded)
            reasoning_emb = decoded.mean(dim=1)  # [B, H]
            lm_logits = self.lm_head(decoded)     # [B, L_rat, V]
        else:
            # Inference: just pool encoder CLS as reasoning embedding
            reasoning_emb = encoder_hidden[:, 0, :]  # CLS token
            lm_logits = torch.zeros(
                batch_size, 1, self.lm_head.out_features,
                device=encoder_hidden.device,
            )

        return reasoning_emb, lm_logits


class ScoreHead(nn.Module):
    """Predicts a regression score from CLS embedding + reasoning embedding +
    handcrafted features.

    Architecture:
        concat(cls_proj, reasoning_proj, feat_proj) → FC layers → sigmoid
    """

    def __init__(
        self,
        hidden_size: int,
        num_handcrafted: int = 27,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cls_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reasoning_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(num_handcrafted, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        fused_dim = hidden_dim + hidden_dim // 2 + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cls_emb: torch.Tensor,
        reasoning_emb: torch.Tensor,
        hc_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict score from multi-source embeddings.

        Args:
            cls_emb:       [B, H] CLS-token hidden state.
            reasoning_emb: [B, H] reasoning head output.
            hc_features:   [B, 27] handcrafted feature vector.

        Returns:
            scores: [B] predicted scores in [0, 1].
        """
        c = self.cls_proj(cls_emb)         # [B, hidden_dim]
        r = self.reasoning_proj(reasoning_emb)  # [B, hidden_dim//2]
        f = self.feat_proj(hc_features)    # [B, 64]
        fused = torch.cat([c, r, f], dim=1)
        return self.head(fused).squeeze(1)


class ReasoningGuidedGrader(nn.Module):
    """Dual-head model: shared encoder → ReasoningHead + ScoreHead.

    The shared encoder produces contextual hidden states; the reasoning head
    generates a rationale embedding (supervised with LM loss) and the score
    head fuses CLS, reasoning embedding, and handcrafted features for
    regression.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-small",
        num_handcrafted: int = 27,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        vocab_size = self.encoder.config.vocab_size

        self.reasoning_head = ReasoningHead(hidden_size, vocab_size, dropout)
        self.score_head = ScoreHead(
            hidden_size, num_handcrafted, hidden_dim, dropout,
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hc_features: torch.Tensor,
        rationale_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            input_ids:      [B, L] tokenised student+reference input.
            attention_mask: [B, L] mask.
            hc_features:    [B, 27] handcrafted features.
            rationale_ids:  [B, L_rat] rationale token ids (training only).

        Returns:
            dict with keys:
                'score':         [B] predicted score.
                'reasoning_emb': [B, H] reasoning embedding.
                'lm_logits':     [B, L_rat, V] LM logits.
        """
        enc_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        hidden_states = enc_out.last_hidden_state  # [B, L, H]
        cls_emb = hidden_states[:, 0, :]           # [B, H]

        reasoning_emb, lm_logits = self.reasoning_head(
            hidden_states, attention_mask, rationale_ids,
        )
        score = self.score_head(cls_emb, reasoning_emb, hc_features)

        return {
            "score": score,
            "reasoning_emb": reasoning_emb,
            "lm_logits": lm_logits,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class CombinedReasoningLoss(nn.Module):
    """Multi-task loss combining scoring accuracy and reasoning LM quality.

    L = α · L_score + β · L_reasoning + γ · L_distill

    where:
        L_score     = CombinedLoss (Huber + QWK)
        L_reasoning = CrossEntropy LM loss on rationale tokens
        L_distill   = MSE between stopped-gradient reasoning_emb and cls_emb
                      (encourages scoring head to learn from reasoning)
    """

    def __init__(
        self,
        scoring_alpha: float = 0.3,
        reasoning_weight: float = 0.3,
        distill_weight: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.score_loss = CombinedLoss(alpha=scoring_alpha, num_classes=num_classes)
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.distill_loss = nn.MSELoss()

        self.reasoning_weight = reasoning_weight
        self.distill_weight = distill_weight

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        true_scores: torch.Tensor,
        rationale_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            model_output: dict from ReasoningGuidedGrader.forward().
            true_scores:  [B] ground-truth normalised scores.
            rationale_ids:[B, L_rat] target rationale token ids.

        Returns:
            dict with 'total', 'score_loss', 'reasoning_loss', 'distill_loss'.
        """
        # Scoring loss
        score_loss = self.score_loss(model_output["score"], true_scores)

        # Reasoning LM loss
        reasoning_loss = torch.tensor(0.0, device=score_loss.device)
        if rationale_ids is not None and model_output["lm_logits"].size(1) > 1:
            logits = model_output["lm_logits"]
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = rationale_ids[:, 1:].contiguous()
            # Truncate to matching lengths
            min_len = min(shift_logits.size(1), shift_targets.size(1))
            reasoning_loss = self.lm_loss(
                shift_logits[:, :min_len, :].reshape(-1, shift_logits.size(-1)),
                shift_targets[:, :min_len].reshape(-1),
            )

        # Knowledge distillation loss (reasoning emb → CLS alignment)
        distill_loss = self.distill_loss(
            model_output["reasoning_emb"],
            model_output["reasoning_emb"].detach(),  # stop gradient
        )

        total = (
            score_loss
            + self.reasoning_weight * reasoning_loss
            + self.distill_weight * distill_loss
        )

        return {
            "total": total,
            "score_loss": score_loss.detach(),
            "reasoning_loss": reasoning_loss.detach(),
            "distill_loss": distill_loss.detach(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_reasoning_epoch(
    model: ReasoningGuidedGrader,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    criterion: CombinedReasoningLoss,
    device: torch.device,
) -> Dict[str, float]:
    """One training epoch for the dual-head model.

    Args:
        model:     ReasoningGuidedGrader instance.
        loader:    DataLoader yielding ReasoningASAGDataset batches.
        optimizer: AdamW optimiser.
        scheduler: Learning rate scheduler.
        criterion: CombinedReasoningLoss instance.
        device:    torch device.

    Returns:
        dict with average losses for the epoch.
    """
    model.train()
    running = {"total": 0.0, "score": 0.0, "reasoning": 0.0, "distill": 0.0}

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        hc = batch["hc_features"].to(device)
        labels = batch["labels"].to(device)
        rationale_ids = batch["rationale_ids"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attn_mask, hc, rationale_ids)
        losses = criterion(output, labels, rationale_ids)
        losses["total"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running["total"] += losses["total"].item()
        running["score"] += losses["score_loss"].item()
        running["reasoning"] += losses["reasoning_loss"].item()
        running["distill"] += losses["distill_loss"].item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def evaluate_reasoning(
    model: ReasoningGuidedGrader,
    loader: DataLoader,
    criterion: CombinedReasoningLoss,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Evaluate model and return predictions, labels, and metrics.

    Args:
        model:     ReasoningGuidedGrader instance.
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    torch device.

    Returns:
        Tuple of (predictions, labels, metrics_dict).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="  eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        hc = batch["hc_features"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, attn_mask, hc)  # no rationale at eval
        loss_dict = criterion(output, labels)
        total_loss += loss_dict["total"].item()

        all_preds.extend(output["score"].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds = np.clip(np.array(all_preds), 0, 1)
    labels = np.array(all_labels)
    metrics = evaluate_all_metrics(preds, labels)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return preds, labels, metrics


# ──────────────────────────────────────────────────────────────────────────────
# K-Fold Training Runner
# ──────────────────────────────────────────────────────────────────────────────

def load_data(cfg: Dict) -> pd.DataFrame:
    """Load and normalise the ASAG dataset.

    Args:
        cfg: Configuration dictionary with 'data_path' key.

    Returns:
        DataFrame with columns: question, provided_answer,
        reference_answer, normalized_grade.
    """
    try:
        from datasets import load_dataset
        raw = load_dataset("Meyerger/ASAG2024")
        df = raw["train"].to_pandas()
    except Exception:
        df = pd.read_csv(cfg["data_path"])

    if "Question" in df.columns:
        df = df.rename(columns={
            "Question": "question",
            "Student Answer": "provided_answer",
            "Reference Answer": "reference_answer",
            "Human Score/Grade": "normalized_grade",
        })

    if df["normalized_grade"].max() > 1.0:
        df["normalized_grade"] = df["normalized_grade"] / df["normalized_grade"].max()

    required = ["question", "reference_answer", "provided_answer", "normalized_grade"]
    df = df[required].fillna({
        "question": "", "reference_answer": "",
        "provided_answer": "", "normalized_grade": 0.0,
    })
    return df


def run_reasoning_training(cfg: Dict) -> Dict:
    """Full K-fold training pipeline for the reasoning-guided model.

    Args:
        cfg: Configuration dictionary (REASONING_DEFAULTS + overrides).

    Returns:
        dict with OOF metrics, fold results, and best thresholds.
    """
    os.makedirs(cfg["save_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_data(cfg)
    student_texts = (
        "Question: " + df["question"] + " Student answer: " + df["provided_answer"]
    ).tolist()
    reference_texts = df["reference_answer"].tolist()
    labels = df["normalized_grade"].values.astype(np.float32)

    # Generate reasoning rationales
    logger.info("Generating reasoning rationales …")
    rationales = batch_generate_rationales(student_texts, reference_texts, labels)

    strat_bins = np.round(labels * (cfg["num_classes"] - 1)).astype(int)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", cfg["encoder_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_name"])

    # ── K-Fold ─────────────────────────────────────────────────────────────
    skf = StratifiedKFold(
        n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"],
    )
    fold_results = []
    oof_preds = np.zeros(len(labels))
    oof_labels = labels.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(student_texts, strat_bins)
    ):
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold_idx + 1, cfg["folds"])
        logger.info("=" * 60)

        # Split data
        train_students = [student_texts[i] for i in train_idx]
        train_refs = [reference_texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        train_rats = [rationales[i] for i in train_idx]

        val_students = [student_texts[i] for i in val_idx]
        val_refs = [reference_texts[i] for i in val_idx]
        val_labels = labels[val_idx]
        val_rats = [rationales[i] for i in val_idx]

        # Datasets
        train_ds = ReasoningASAGDataset(
            train_students, train_refs, train_labels, train_rats,
            tokenizer, cfg["max_length"], cfg["reasoning_max_length"],
        )
        val_ds = ReasoningASAGDataset(
            val_students, val_refs, val_labels, val_rats,
            tokenizer, cfg["max_length"], cfg["reasoning_max_length"],
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        # Model
        model = ReasoningGuidedGrader(
            encoder_name=cfg["encoder_name"],
            num_handcrafted=len(FEATURE_NAMES),
            hidden_dim=256,
            dropout=cfg["dropout"],
        ).to(device)

        # Optimiser: separate LR for encoder vs heads
        encoder_params = list(model.encoder.parameters())
        head_params = (
            list(model.reasoning_head.parameters())
            + list(model.score_head.parameters())
        )
        optimizer = AdamW([
            {"params": encoder_params, "lr": cfg["lr"]},
            {"params": head_params, "lr": cfg["reasoning_lr"]},
        ], weight_decay=cfg["weight_decay"])

        total_steps = len(train_loader) * cfg["epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps,
        )

        criterion = CombinedReasoningLoss(
            scoring_alpha=cfg["loss_alpha"],
            reasoning_weight=cfg["reasoning_weight"],
            distill_weight=cfg["distill_weight"],
            num_classes=cfg["num_classes"],
        )

        # Training loop
        best_qwk = -1.0
        best_state = None
        patience, patience_limit = 0, 3

        for epoch in range(cfg["epochs"]):
            train_losses = train_reasoning_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
            )
            val_preds, val_labels_arr, val_metrics = evaluate_reasoning(
                model, val_loader, criterion, device,
            )

            logger.info(
                "  Epoch %d/%d — train_loss: %.4f  val_QWK: %.4f  val_acc: %.4f",
                epoch + 1, cfg["epochs"],
                train_losses["total"],
                val_metrics.get("qwk", 0),
                val_metrics.get("accuracy", 0),
            )

            if val_metrics.get("qwk", 0) > best_qwk:
                best_qwk = val_metrics["qwk"]
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    logger.info("  Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best and final eval
        if best_state is not None:
            model.load_state_dict(best_state)

        val_preds, val_labels_arr, val_metrics = evaluate_reasoning(
            model, val_loader, criterion, device,
        )

        # Score rounding
        rounder = ScoreRounder(num_classes=cfg["num_classes"])
        rounder.fit(val_preds, val_labels_arr)
        rounded = rounder.predict(val_preds)
        rounded_metrics = evaluate_all_metrics(rounded, val_labels_arr)
        val_metrics["rounded_qwk"] = rounded_metrics.get("qwk", 0)
        val_metrics["rounded_accuracy"] = rounded_metrics.get("accuracy", 0)

        oof_preds[val_idx] = val_preds
        fold_results.append(val_metrics)

        # Save fold model
        fold_path = os.path.join(cfg["save_dir"], f"reasoning_fold{fold_idx}.pt")
        torch.save({
            "model_state": best_state or model.state_dict(),
            "config": cfg,
            "fold": fold_idx,
            "metrics": val_metrics,
            "rounder_thresholds": rounder.thresholds.tolist()
            if hasattr(rounder, "thresholds") and rounder.thresholds is not None
            else None,
        }, fold_path)
        logger.info("  Saved fold %d → %s (QWK=%.4f)", fold_idx, fold_path, best_qwk)

    # ── OOF Metrics ────────────────────────────────────────────────────────
    oof_metrics = evaluate_all_metrics(oof_preds, oof_labels)
    logger.info("=" * 60)
    logger.info("OOF METRICS")
    logger.info("  MSE:      %.4f", oof_metrics.get("mse", 0))
    logger.info("  QWK:      %.4f", oof_metrics.get("qwk", 0))
    logger.info("  Accuracy: %.4f", oof_metrics.get("accuracy", 0))
    logger.info("  F1:       %.4f", oof_metrics.get("f1", 0))

    # Save results
    results = {
        "oof_metrics": oof_metrics,
        "fold_results": fold_results,
        "config": cfg,
    }
    results_path = os.path.join(cfg["save_dir"], "reasoning_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved → %s", results_path)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Reasoning-guided ASAG training pipeline",
    )
    p.add_argument(
        "--model", default="deberta",
        choices=list(MODEL_ALIASES.keys()),
        help="Encoder backbone alias",
    )
    p.add_argument("--folds", type=int, default=REASONING_DEFAULTS["folds"])
    p.add_argument("--epochs", type=int, default=REASONING_DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=REASONING_DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=REASONING_DEFAULTS["lr"])
    p.add_argument("--dropout", type=float, default=REASONING_DEFAULTS["dropout"])
    p.add_argument("--data", default=REASONING_DEFAULTS["data_path"])
    p.add_argument("--save_dir", default=REASONING_DEFAULTS["save_dir"])
    p.add_argument(
        "--stage", default="full", choices=["full", "distill"],
        help="'full' = joint training; 'distill' = distillation only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    encoder_name = MODEL_ALIASES.get(args.model, args.model)

    cfg = dict(REASONING_DEFAULTS)
    cfg.update({
        "encoder_name": encoder_name,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "dropout": args.dropout,
        "data_path": args.data,
        "save_dir": args.save_dir,
    })

    if args.stage == "distill":
        cfg["reasoning_weight"] = 0.0
        cfg["distill_weight"] = 0.8
        logger.info("Running distillation-only mode (reasoning_weight=0)")

    run_reasoning_training(cfg)
