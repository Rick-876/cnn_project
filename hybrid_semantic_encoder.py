"""
hybrid_semantic_encoder.py
Hybrid TextCNN + SBERT encoder with cross-attention and gated fusion.

Architecture:
    ┌───────────────────────┐   ┌────────────────────────────┐
    │ TextCNN Branch        │   │ SBERT Branch               │
    │ Conv1D n-gram filters │   │ sentence-transformers      │
    │ → local features      │   │ → global semantic emb      │
    │ [B, cnn_dim]          │   │ [B, sbert_dim]             │
    └───────────┬───────────┘   └──────────────┬─────────────┘
                │                              │
                └──────────┬───────────────────┘
                           ↓
              ┌────────────────────────────┐
              │ Cross-Attention Alignment  │
              │ student ↔ reference        │
              │ [B, align_dim]             │
              └────────────┬───────────────┘
                           ↓
              ┌────────────────────────────┐
              │ Gated Fusion               │
              │ σ(W·[cnn; sbert; attn])    │
              │ → adaptive combo [B, fused]│
              └────────────┬───────────────┘
                           ↓ + handcrafted features
              ┌────────────────────────────┐
              │ Scoring Head               │
              │ FC → sigmoid → [0, 1]      │
              └────────────────────────────┘

Usage:
    from hybrid_semantic_encoder import HybridSemanticEncoder
    model = HybridSemanticEncoder(vocab_size=10000, sbert_name='all-MiniLM-L6-v2')

Requires: torch, sentence-transformers
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# TextCNN Branch (adapted from asag_cnn_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

class TextCNNBranch(nn.Module):
    """1D convolutional branch for local n-gram pattern extraction.

    Mirrors the TextCNN from ``asag_cnn_pipeline.py`` but without the final
    classification head—outputs a fixed-size feature vector instead.

    Args:
        vocab_size:   Vocabulary size for the embedding layer.
        embed_dim:    Embedding dimensionality.
        num_filters:  Number of output channels per filter size.
        filter_sizes: List of convolution kernel widths (n-gram sizes).
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_filters: int = 128,
        filter_sizes: Optional[List[int]] = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(filter_sizes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract local n-gram features.

        Args:
            input_ids: [B, L] token id tensor.

        Returns:
            [B, out_dim] feature vector.
        """
        x = self.embedding(input_ids).unsqueeze(1)  # [B, 1, L, E]
        conv_out = [F.relu(c(x)).squeeze(3) for c in self.convs]  # [B, F, L']
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_out]
        return self.dropout(torch.cat(pooled, dim=1))  # [B, F*n_filters]


# ──────────────────────────────────────────────────────────────────────────────
# SBERT Branch
# ──────────────────────────────────────────────────────────────────────────────

class SBERTBranch(nn.Module):
    """Sentence-BERT branch for global semantic similarity.

    Encodes student and reference answers into dense embeddings using a
    pre-trained sentence-transformers model, then produces:
      • Concatenated embeddings [student_emb; reference_emb]
      • Element-wise difference |student - reference|
      • Cosine similarity scalar

    Args:
        model_name: HuggingFace sentence-transformers model name.
        proj_dim:   Dimension to project SBERT output to.
        freeze:     Whether to freeze SBERT weights (improves speed).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        proj_dim: int = 128,
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(model_name)
            self.sbert_dim = self.sbert.get_sentence_embedding_dimension()
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using a lightweight fallback."
            )
            self.sbert = None
            self.sbert_dim = 384  # MiniLM default

        if freeze and self.sbert is not None:
            for p in self.sbert.parameters():
                p.requires_grad = False

        # Projections
        # Output: proj(student) ‖ proj(reference) ‖ |diff| ‖ cosine
        self.proj = nn.Linear(self.sbert_dim, proj_dim)
        self.out_dim = proj_dim * 3 + 1  # concat + diff + cosine

    @torch.no_grad()
    def _encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode texts with SBERT, returning [N, sbert_dim]."""
        if self.sbert is not None:
            embs = self.sbert.encode(
                texts, convert_to_tensor=True, show_progress_bar=False,
            )
            return embs.to(device)
        else:
            # Fallback: random (for testing without sentence-transformers)
            return torch.randn(len(texts), self.sbert_dim, device=device)

    def forward(
        self,
        student_texts: List[str],
        reference_texts: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute SBERT-based semantic features.

        Args:
            student_texts:   List of student answer strings.
            reference_texts: List of reference answer strings.
            device:          Target torch device.

        Returns:
            [B, out_dim] semantic feature tensor.
        """
        s_emb = self._encode(student_texts, device)   # [B, sbert_dim]
        r_emb = self._encode(reference_texts, device)  # [B, sbert_dim]

        s_proj = self.proj(s_emb)  # [B, proj_dim]
        r_proj = self.proj(r_emb)  # [B, proj_dim]

        diff = torch.abs(s_proj - r_proj)  # [B, proj_dim]
        cos = F.cosine_similarity(s_emb, r_emb, dim=1, eps=1e-8).unsqueeze(1)

        return torch.cat([s_proj, r_proj, diff, cos], dim=1)  # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Alignment
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """Multi-head cross-attention between student and reference token
    sequences for fine-grained alignment features.

    Args:
        embed_dim: Dimensionality of each input token embedding.
        num_heads: Number of attention heads.
        dropout:   Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 300,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        student_emb: torch.Tensor,
        reference_emb: torch.Tensor,
        student_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend student tokens over reference tokens.

        Args:
            student_emb:   [B, L_s, E] student token embeddings.
            reference_emb: [B, L_r, E] reference token embeddings.
            student_mask:  [B, L_s] key_padding_mask (True = ignore).
            reference_mask:[B, L_r] key_padding_mask (True = ignore).

        Returns:
            [B, E] aligned representation (mean-pooled).
        """
        attn_out, attn_weights = self.cross_attn(
            query=student_emb,
            key=reference_emb,
            value=reference_emb,
            key_padding_mask=reference_mask,
        )
        attn_out = self.layer_norm(attn_out + student_emb[:, :attn_out.size(1), :])
        ffn_out = self.ffn(attn_out)
        out = self.ln2(ffn_out + attn_out)

        # Mean pool (masked)
        if student_mask is not None:
            mask = (~student_mask[:, :out.size(1)]).unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        # Store attention weights for visualization
        self._last_attn_weights = attn_weights.detach()

        return pooled  # [B, E]


# ──────────────────────────────────────────────────────────────────────────────
# Gated Fusion
# ──────────────────────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """Learnable gated fusion of multiple feature streams.

    Each stream is projected to a shared dimension, then a learned gate
    determines how much each stream contributes to the final representation:
        g_i = σ(W_g · [all_streams_concat])
        fused = Σ g_i · stream_i

    Args:
        stream_dims: List of input dimensionalities for each stream.
        fused_dim:   Output dimensionality of the fused representation.
    """

    def __init__(self, stream_dims: List[int], fused_dim: int = 256):
        super().__init__()
        self.n_streams = len(stream_dims)
        self.fused_dim = fused_dim

        # Project each stream to shared dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
            )
            for dim in stream_dims
        ])

        # Gate network: takes concatenated projected streams → gate weights
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * self.n_streams, fused_dim * self.n_streams),
            nn.ReLU(),
            nn.Linear(fused_dim * self.n_streams, self.n_streams),
            nn.Sigmoid(),
        )

    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature streams with learned gates.

        Args:
            streams: List of [B, dim_i] tensors (one per stream).

        Returns:
            [B, fused_dim] gated-fused representation.
        """
        assert len(streams) == self.n_streams

        projected = [proj(s) for proj, s in zip(self.projections, streams)]
        concat = torch.cat(projected, dim=1)  # [B, fused_dim * n]
        gates = self.gate(concat)  # [B, n_streams]

        # Weighted sum
        stacked = torch.stack(projected, dim=1)  # [B, n, fused_dim]
        gates_exp = gates.unsqueeze(-1)  # [B, n, 1]
        fused = (stacked * gates_exp).sum(dim=1)  # [B, fused_dim]

        return fused


# ──────────────────────────────────────────────────────────────────────────────
# Full Hybrid Semantic Encoder
# ──────────────────────────────────────────────────────────────────────────────

class HybridSemanticEncoder(nn.Module):
    """Complete hybrid model combining TextCNN + SBERT + cross-attention +
    gated fusion + handcrafted features for ASAG scoring.

    Args:
        vocab_size:      TextCNN vocabulary size.
        embed_dim:       TextCNN embedding dimension.
        num_filters:     TextCNN filters per kernel size.
        filter_sizes:    TextCNN kernel widths.
        sbert_name:      Sentence-transformers model name.
        sbert_proj_dim:  SBERT projection dimensionality.
        cross_attn_heads:Number of attention heads.
        fused_dim:       Gated fusion output dimensionality.
        num_handcrafted: Number of handcrafted features (default 27).
        dropout:         Dropout probability.
        freeze_sbert:    Whether to freeze SBERT weights.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_filters: int = 128,
        filter_sizes: Optional[List[int]] = None,
        sbert_name: str = "all-MiniLM-L6-v2",
        sbert_proj_dim: int = 128,
        cross_attn_heads: int = 4,
        fused_dim: int = 256,
        num_handcrafted: int = 27,
        dropout: float = 0.3,
        freeze_sbert: bool = True,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]

        # Branches
        self.cnn = TextCNNBranch(
            vocab_size, embed_dim, num_filters, filter_sizes, dropout,
        )
        self.sbert = SBERTBranch(
            sbert_name, sbert_proj_dim, freeze=freeze_sbert,
        )
        self.cross_attn = CrossAttentionLayer(
            embed_dim=embed_dim, num_heads=cross_attn_heads, dropout=dropout,
        )

        # Stream dimensions for gated fusion
        cnn_dim = self.cnn.out_dim           # num_filters * len(filter_sizes)
        sbert_dim = self.sbert.out_dim       # proj*3 + 1
        cross_dim = embed_dim                # cross-attention output

        self.fusion = GatedFusion(
            stream_dims=[cnn_dim, sbert_dim, cross_dim],
            fused_dim=fused_dim,
        )

        # Handcrafted feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(num_handcrafted, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # Scoring head
        self.head = nn.Sequential(
            nn.Linear(fused_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Store embedding for cross-attention
        self._embed_dim = embed_dim

    def forward(
        self,
        student_ids: torch.Tensor,
        reference_ids: torch.Tensor,
        student_texts: List[str],
        reference_texts: List[str],
        hc_features: torch.Tensor,
        student_mask: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            student_ids:     [B, L_s] tokenised student answer ids.
            reference_ids:   [B, L_r] tokenised reference answer ids.
            student_texts:   Raw student answer strings (for SBERT).
            reference_texts: Raw reference answer strings (for SBERT).
            hc_features:     [B, 27] handcrafted feature vector.
            student_mask:    [B, L_s] padding mask (True = pad).
            reference_mask:  [B, L_r] padding mask (True = pad).

        Returns:
            dict with 'score' and 'gate_weights'.
        """
        device = student_ids.device

        # 1. TextCNN branch: local n-gram features from student
        cnn_feats = self.cnn(student_ids)  # [B, cnn_dim]

        # 2. SBERT branch: global semantic comparison
        sbert_feats = self.sbert(
            student_texts, reference_texts, device,
        )  # [B, sbert_dim]

        # 3. Cross-attention: alignment between student and reference
        s_emb = self.cnn.embedding(student_ids)     # [B, L_s, E]
        r_emb = self.cnn.embedding(reference_ids)   # [B, L_r, E]
        cross_feats = self.cross_attn(
            s_emb, r_emb,
            student_mask=student_mask,
            reference_mask=reference_mask,
        )  # [B, cross_dim]

        # 4. Gated fusion of three streams
        fused = self.fusion([cnn_feats, sbert_feats, cross_feats])  # [B, fused_dim]

        # 5. Add handcrafted features
        feat_proj = self.feat_proj(hc_features)  # [B, 64]
        combined = torch.cat([fused, feat_proj], dim=1)  # [B, fused+64]

        # 6. Score
        score = self.head(combined).squeeze(1)  # [B]

        # Capture gate weights for interpretability
        gate_weights = self.fusion.gate(
            torch.cat([
                proj(s) for proj, s in zip(
                    self.fusion.projections,
                    [cnn_feats, sbert_feats, cross_feats],
                )
            ], dim=1)
        )

        return {
            "score": score,
            "gate_weights": gate_weights.detach(),
            "cnn_feats": cnn_feats.detach(),
            "sbert_feats": sbert_feats.detach(),
            "cross_feats": cross_feats.detach(),
        }

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Retrieve last cross-attention weights for visualization.

        Returns:
            [B, num_heads, L_s, L_r] attention weight tensor or None.
        """
        if hasattr(self.cross_attn, "_last_attn_weights"):
            return self.cross_attn._last_attn_weights
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset for Hybrid Semantic Encoder
# ──────────────────────────────────────────────────────────────────────────────

class HybridSemanticDataset(torch.utils.data.Dataset):
    """Dataset that provides tokenised ids for both student and reference,
    raw texts for SBERT, handcrafted features, and labels.

    Args:
        student_answers:   List of student answer strings.
        reference_answers: List of reference answer strings.
        labels:            Numpy array of normalised scores.
        vocab:             dict mapping word → int id.
        max_len:           Maximum sequence length for CNN tokenisation.
    """

    def __init__(
        self,
        student_answers: List[str],
        reference_answers: List[str],
        labels: np.ndarray,
        vocab: Dict[str, int],
        max_len: int = 100,
    ):
        from feature_engineering import batch_extract_features

        self.student_texts = student_answers
        self.reference_texts = reference_answers
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Tokenise for TextCNN
        self.student_ids = self._encode_all(student_answers, vocab, max_len)
        self.reference_ids = self._encode_all(reference_answers, vocab, max_len)

        # Handcrafted features
        logger.info("  Extracting handcrafted features …")
        self.hc_features = torch.tensor(
            batch_extract_features(student_answers, reference_answers),
            dtype=torch.float32,
        )

    @staticmethod
    def _encode_all(
        texts: List[str], vocab: Dict[str, int], max_len: int,
    ) -> torch.Tensor:
        """Encode list of texts to padded id tensors."""
        def _encode(text: str) -> List[int]:
            tokens = re.findall(r"\w+", text.lower())[:max_len]
            ids = [vocab.get(t, 0) for t in tokens]
            ids += [0] * (max_len - len(ids))
            return ids

        return torch.tensor([_encode(t) for t in texts], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "student_ids": self.student_ids[idx],
            "reference_ids": self.reference_ids[idx],
            "student_text": self.student_texts[idx],
            "reference_text": self.reference_texts[idx],
            "hc_features": self.hc_features[idx],
            "labels": self.labels[idx],
        }


def hybrid_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate that preserves raw text lists alongside tensors."""
    return {
        "student_ids": torch.stack([b["student_ids"] for b in batch]),
        "reference_ids": torch.stack([b["reference_ids"] for b in batch]),
        "student_texts": [b["student_text"] for b in batch],
        "reference_texts": [b["reference_text"] for b in batch],
        "hc_features": torch.stack([b["hc_features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
