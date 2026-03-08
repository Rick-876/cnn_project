"""
domain_pretraining.py
Continued MLM pre-training pipeline for domain adaptation.

Adapts a pre-trained transformer (DeBERTa/RoBERTa) to the ASAG domain by
continued Masked Language Modelling on domain-specific texts (reference
answers, textbook excerpts, lecture notes).

Pipeline:
    1. Build domain corpus from available sources
    2. Configure MLM data collator with 15 % masking
    3. Continue pre-training with small LR + warmup
    4. Save domain-adapted weights → use as backbone in hybrid_pipeline.py
    5. Two-stage fine-tuning: MLM-adapted backbone → task fine-tuning

Usage:
    python domain_pretraining.py                       # default MLM
    python domain_pretraining.py --model roberta       # use RoBERTa
    python domain_pretraining.py --epochs 5 --lr 5e-5  # custom config
    python domain_pretraining.py --compare             # compare vs base

Requires: transformers, torch, datasets (optional)
"""

import os
import re
import json
import math
import logging
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PRETRAIN_DEFAULTS: Dict = {
    "model_name": "microsoft/deberta-v3-small",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 3,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "mlm_probability": 0.15,
    "data_path": "asag2024_all.csv",
    "extra_corpus_dir": None,        # optional folder of .txt files
    "save_dir": "domain_pretrained",
    "seed": 42,
}

MODEL_ALIASES: Dict[str, str] = {
    "deberta": "microsoft/deberta-v3-small",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
}


# ──────────────────────────────────────────────────────────────────────────────
# Domain Corpus Builder
# ──────────────────────────────────────────────────────────────────────────────

class DomainCorpusBuilder:
    """Builds a domain-specific corpus from available text sources.

    Sources:
        1. Reference answers from the ASAG dataset
        2. Questions (provide context about domain)
        3. Optional external text files (textbooks, lecture notes)
    """

    def __init__(self, data_path: str, extra_dir: Optional[str] = None):
        """Initialise corpus builder.

        Args:
            data_path: Path to CSV file with questions and reference answers.
            extra_dir: Optional directory containing additional .txt files.
        """
        self.data_path = data_path
        self.extra_dir = extra_dir

    def build(self) -> List[str]:
        """Collect all domain texts.

        Returns:
            List of text passages (one per entry).
        """
        texts: List[str] = []

        # Source 1: Reference answers + questions from dataset
        logger.info("Loading domain texts from %s …", self.data_path)
        df = self._load_csv()
        for _, row in df.iterrows():
            # Combine question + reference for richer context
            q = str(row.get("question", ""))
            r = str(row.get("reference_answer", ""))
            if r.strip():
                texts.append(f"{q} {r}".strip())
            # Student answers (correct ones) are also domain text
            s = str(row.get("provided_answer", ""))
            grade = row.get("normalized_grade", 0.0)
            if grade >= 0.7 and s.strip():
                texts.append(s)

        logger.info("  Extracted %d passages from dataset", len(texts))

        # Source 2: Extra corpus files
        if self.extra_dir and os.path.isdir(self.extra_dir):
            extra_count = 0
            for fname in os.listdir(self.extra_dir):
                if fname.endswith(".txt"):
                    fpath = os.path.join(self.extra_dir, fname)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                    # Split long documents into passages
                    passages = self._split_passages(content, max_words=200)
                    texts.extend(passages)
                    extra_count += len(passages)
            logger.info(
                "  Extracted %d passages from %s", extra_count, self.extra_dir,
            )

        # Deduplicate
        seen = set()
        unique = []
        for t in texts:
            t_norm = re.sub(r"\s+", " ", t.lower().strip())
            if t_norm and t_norm not in seen:
                seen.add(t_norm)
                unique.append(t)
        logger.info("  Final corpus: %d unique passages", len(unique))
        return unique

    def _load_csv(self) -> pd.DataFrame:
        """Load and normalise the data CSV."""
        df = pd.read_csv(self.data_path)
        if "Question" in df.columns:
            df = df.rename(columns={
                "Question": "question",
                "Student Answer": "provided_answer",
                "Reference Answer": "reference_answer",
                "Human Score/Grade": "normalized_grade",
            })
        if "normalized_grade" in df.columns and df["normalized_grade"].max() > 1.0:
            df["normalized_grade"] = (
                df["normalized_grade"] / df["normalized_grade"].max()
            )
        return df

    @staticmethod
    def _split_passages(text: str, max_words: int = 200) -> List[str]:
        """Split long text into ~max_words-sized passages."""
        words = text.split()
        passages = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) >= 10:  # skip tiny fragments
                passages.append(chunk)
        return passages


# ──────────────────────────────────────────────────────────────────────────────
# MLM Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MLMDataset(Dataset):
    """Simple dataset for masked language modelling.

    Args:
        texts:      List of text passages.
        tokenizer:  HuggingFace tokenizer.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self, texts: List[str], tokenizer, max_length: int = 256,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.encodings.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Pre-training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_domain_pretraining(cfg: Dict) -> Dict:
    """Execute domain-adaptive MLM pre-training.

    Args:
        cfg: Configuration dictionary.

    Returns:
        dict with training stats and saved model path.
    """
    os.makedirs(cfg["save_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Build corpus ───────────────────────────────────────────────────────
    builder = DomainCorpusBuilder(cfg["data_path"], cfg.get("extra_corpus_dir"))
    corpus = builder.build()

    if len(corpus) < 50:
        logger.warning(
            "Domain corpus is very small (%d passages). "
            "Consider adding more domain texts.", len(corpus),
        )

    # ── Tokenizer and Model ───────────────────────────────────────────────
    logger.info("Loading model: %s", cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForMaskedLM.from_pretrained(cfg["model_name"]).to(device)

    # ── Dataset and Collator ──────────────────────────────────────────────
    dataset = MLMDataset(corpus, tokenizer, cfg["max_length"])
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg["mlm_probability"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # ── Optimiser and Scheduler ───────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = len(loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
    )

    # ── Training ──────────────────────────────────────────────────────────
    logger.info(
        "Starting MLM pre-training: %d passages, %d epochs, %d steps",
        len(corpus), cfg["epochs"], total_steps,
    )

    history: List[Dict[str, float]] = []
    best_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2),
        })
        logger.info(
            "  Epoch %d/%d — MLM loss: %.4f — perplexity: %.2f",
            epoch + 1, cfg["epochs"], avg_loss, perplexity,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best checkpoint
            save_path = os.path.join(cfg["save_dir"], "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info("  Best model saved → %s", save_path)

    # ── Final save ────────────────────────────────────────────────────────
    final_path = os.path.join(cfg["save_dir"], "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    results = {
        "model_name": cfg["model_name"],
        "corpus_size": len(corpus),
        "epochs": cfg["epochs"],
        "best_loss": round(best_loss, 4),
        "best_perplexity": round(math.exp(min(best_loss, 20)), 2),
        "history": history,
        "saved_to": final_path,
    }
    results_path = os.path.join(cfg["save_dir"], "pretraining_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", results_path)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Two-stage fine-tuning integration
# ──────────────────────────────────────────────────────────────────────────────

def load_domain_adapted_model(
    pretrained_dir: str, device: Optional[torch.device] = None,
) -> tuple:
    """Load domain-adapted model weights for downstream fine-tuning.

    This loads the encoder weights only (strips the MLM head) so they can
    be plugged into ``HybridGrader`` or ``ReasoningGuidedGrader``.

    Args:
        pretrained_dir: Path to saved domain-adapted model directory.
        device:         Target torch device.

    Returns:
        (tokenizer, encoder_state_dict) — ready to initialise downstream model.
    """
    from transformers import AutoModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

    # Load encoder portion only (discard LM head)
    model_mlm = AutoModelForMaskedLM.from_pretrained(pretrained_dir)
    # Most MaskedLM models have a .base_model attribute (e.g. .deberta)
    base_model = getattr(model_mlm, "base_model", None)
    if base_model is None:
        # Fallback: load AutoModel directly
        base_model = AutoModel.from_pretrained(pretrained_dir)

    state_dict = base_model.state_dict()
    logger.info(
        "Loaded domain-adapted encoder from %s (%d parameters)",
        pretrained_dir, sum(p.numel() for p in base_model.parameters()),
    )
    return tokenizer, state_dict


def init_hybrid_with_domain_weights(
    domain_dir: str,
    num_handcrafted: int = 27,
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> tuple:
    """Create a HybridGrader initialised with domain-adapted encoder weights.

    Args:
        domain_dir:      Path to the domain-adapted model.
        num_handcrafted: Number of handcrafted features.
        hidden_dim:      Hidden dimension for the scoring head.
        dropout:         Dropout probability.

    Returns:
        (model, tokenizer) ready for downstream fine-tuning.
    """
    from hybrid_pipeline import HybridGrader

    tokenizer, encoder_state = load_domain_adapted_model(domain_dir)

    # Determine original model name from config
    config_path = os.path.join(domain_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        model_name = config.get("_name_or_path", "microsoft/deberta-v3-small")
    else:
        model_name = "microsoft/deberta-v3-small"

    model = HybridGrader(
        transformer_name=model_name,
        num_handcrafted=num_handcrafted,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # Replace encoder weights with domain-adapted ones
    missing, unexpected = model.transformer.load_state_dict(
        encoder_state, strict=False,
    )
    if missing:
        logger.warning("Missing keys when loading domain weights: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:5])

    logger.info("HybridGrader initialised with domain-adapted encoder")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Performance comparison utility
# ──────────────────────────────────────────────────────────────────────────────

def compare_models(
    base_metrics: Dict[str, float],
    domain_metrics: Dict[str, float],
    verbose: bool = True,
) -> Dict:
    """Compare base vs domain-adapted model performance.

    Args:
        base_metrics:   Metrics dict from base model evaluation.
        domain_metrics: Metrics dict from domain-adapted model evaluation.
        verbose:        Whether to print comparison table.

    Returns:
        dict with metric deltas and improvement flags.
    """
    comparison = {}
    for key in ["mse", "qwk", "accuracy", "f1"]:
        base_val = base_metrics.get(key, 0)
        domain_val = domain_metrics.get(key, 0)
        delta = domain_val - base_val

        # For MSE, lower is better; for everything else, higher is better
        if key == "mse":
            improved = delta < 0
            delta_display = f"{delta:+.4f}"
        else:
            improved = delta > 0
            delta_display = f"{delta:+.4f}"

        comparison[key] = {
            "base": round(base_val, 4),
            "domain": round(domain_val, 4),
            "delta": round(delta, 4),
            "improved": improved,
        }

    if verbose:
        print("=" * 60)
        print("MODEL COMPARISON: Base vs Domain-Adapted")
        print("=" * 60)
        print(f"{'Metric':>12s}  {'Base':>8s}  {'Domain':>8s}  {'Delta':>8s}  Status")
        print("-" * 60)
        for key, vals in comparison.items():
            status = "✓ Better" if vals["improved"] else "✗ Worse"
            print(
                f"{key:>12s}  {vals['base']:8.4f}  {vals['domain']:8.4f}  "
                f"{vals['delta']:+8.4f}  {status}"
            )
        print("=" * 60)

    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Domain-adaptive MLM pre-training for ASAG",
    )
    p.add_argument(
        "--model", default="deberta",
        choices=list(MODEL_ALIASES.keys()),
        help="Transformer backbone alias",
    )
    p.add_argument("--epochs", type=int, default=PRETRAIN_DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=PRETRAIN_DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=PRETRAIN_DEFAULTS["lr"])
    p.add_argument("--data", default=PRETRAIN_DEFAULTS["data_path"])
    p.add_argument("--extra_dir", default=None,
                   help="Directory of extra .txt domain corpus files")
    p.add_argument("--save_dir", default=PRETRAIN_DEFAULTS["save_dir"])
    p.add_argument("--compare", action="store_true",
                   help="Run comparison between base and domain-adapted model")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_name = MODEL_ALIASES.get(args.model, args.model)

    cfg = dict(PRETRAIN_DEFAULTS)
    cfg.update({
        "model_name": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "data_path": args.data,
        "extra_corpus_dir": args.extra_dir,
        "save_dir": args.save_dir,
    })

    results = run_domain_pretraining(cfg)

    logger.info(
        "Domain pre-training complete. Best MLM loss: %.4f, perplexity: %.2f",
        results["best_loss"], results["best_perplexity"],
    )
    logger.info(
        "Use domain-adapted weights with:\n"
        "  from domain_pretraining import init_hybrid_with_domain_weights\n"
        "  model, tokenizer = init_hybrid_with_domain_weights('%s')",
        os.path.join(cfg["save_dir"], "best_model"),
    )
