"""
training_utils.py
Training utilities: QWK-aware loss, Huber loss, class weighting,
and score rounding optimisation.

These are shared by both the CNN pipeline and the hybrid transformer pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class QWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa loss.

    Based on the formulation by Yann Dauphin et al. (ICLR 2017 workshop).
    Works with continuous predictions in [0, 1] and target labels in [0, 1].

    The key idea: QWK penalizes disagreements proportionally to the square
    of the distance between ratings, so we approximate this with a continuous
    weighted quadratic penalty over soft histogram bins.

    Args:
        num_classes: Number of discrete grade classes (e.g. 3 for 0/1/2 system).
        eps:         Smoothing term to avoid division by zero.
    """

    def __init__(self, num_classes: int = 3, eps: float = 1e-10):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        # Weight matrix W[i,j] = (i-j)² / (num_classes-1)²
        wt = np.zeros((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                wt[i, j] = float((i - j) ** 2) / float((num_classes - 1) ** 2)
        self.register_buffer('weight_mat', torch.tensor(wt))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:   (B,) tensor of continuous predictions in [0, 1].
            targets: (B,) tensor of continuous targets in [0, 1].
        """
        batch_size = preds.size(0)
        nc = self.num_classes

        # Scale to [0, num_classes - 1]
        pred_scaled  = preds   * (nc - 1)
        tgt_scaled   = targets * (nc - 1)

        # Soft one-hot histograms using sigmoid gates
        # Each column k gets weight softmax(-|x - k|)
        class_centers = torch.arange(nc, dtype=torch.float32,
                                     device=preds.device)  # [nc]

        pred_scaled_exp = pred_scaled.unsqueeze(1).expand(-1, nc)   # [B, nc]
        tgt_scaled_exp  = tgt_scaled.unsqueeze(1).expand(-1, nc)    # [B, nc]

        pred_hist = torch.softmax(-torch.abs(pred_scaled_exp - class_centers), dim=1)
        tgt_hist  = torch.softmax(-torch.abs(tgt_scaled_exp  - class_centers), dim=1)

        # Confusion matrix: C[i,j] = sum over batch of pred_hist[:,i] * tgt_hist[:,j]
        # Shape [nc, nc]
        conf_mat = torch.mm(pred_hist.t(), tgt_hist) / batch_size  # [nc, nc]

        row_sum = conf_mat.sum(dim=1, keepdim=True)   # [nc, 1]
        col_sum = conf_mat.sum(dim=0, keepdim=True)   # [1, nc]
        expected = row_sum * col_sum                   # [nc, nc]
        expected = expected / (expected.sum() + self.eps)

        conf_normalised = conf_mat / (conf_mat.sum() + self.eps)

        numerator   = (self.weight_mat * conf_normalised).sum()
        denominator = (self.weight_mat * expected).sum()

        qwk_approx = 1.0 - numerator / (denominator + self.eps)

        # Return 1 – QWK so we minimise it during training
        return 1.0 - qwk_approx


class FocalMSELoss(nn.Module):
    """Focal-style regression loss that emphasises hard examples.

    Scales the squared error by |error|^gamma so that easy predictions
    (small error) contribute less gradient and hard predictions (large
    error near class boundaries) contribute more.

    Args:
        gamma: Focusing parameter (0 = standard MSE, higher = more focus
               on hard examples). Recommended range: 1.0–2.0.
    """

    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                sample_weights: torch.Tensor = None) -> torch.Tensor:
        error = (preds - targets).abs().detach()   # detach to avoid double grad
        focal_weight = (error + 1e-6) ** self.gamma
        loss = focal_weight * (preds - targets) ** 2
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines a regression loss with QWK loss.

    total = alpha * regression_loss + (1 - alpha) * qwk_loss

    The regression component can be MSE, Huber, or Focal MSE.

    Args:
        alpha:       Weight for regression loss (0 = pure QWK, 1 = pure MSE).
        num_classes: Number of grade classes for QWK calculation.
        huber_delta: If > 0 and focal_gamma == 0, use Huber loss.
        focal_gamma: If > 0, use FocalMSELoss instead of Huber/MSE.
    """

    def __init__(self, alpha: float = 0.3, num_classes: int = 3,
                 huber_delta: float = 0.5, focal_gamma: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.qwk_loss = QWKLoss(num_classes=num_classes)
        self.use_focal = focal_gamma > 0
        if self.use_focal:
            self.regression_loss = FocalMSELoss(gamma=focal_gamma)
        elif huber_delta > 0:
            self.regression_loss = nn.HuberLoss(delta=huber_delta)
        else:
            self.regression_loss = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                sample_weights: torch.Tensor = None) -> torch.Tensor:
        if self.use_focal:
            reg_loss = self.regression_loss(preds, targets, sample_weights)
        else:
            reg_loss = self.regression_loss(preds, targets)
        qwk_component = self.qwk_loss(preds, targets)
        return self.alpha * reg_loss + (1.0 - self.alpha) * qwk_component


# ---------------------------------------------------------------------------
# Class weighting (for imbalanced grade distributions)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        labels:      Array of continuous labels in [0, 1].
        num_classes: Number of discrete bins.

    Returns:
        np.ndarray of shape (N,) with per-sample weights.
    """
    bins = np.round(labels * (num_classes - 1)).astype(int)
    class_counts = np.bincount(bins, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1.0)

    # Inverse frequency, normalized to mean = 1
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.mean()

    sample_weights = class_weights[bins]
    return sample_weights


def compute_label_weights(labels: np.ndarray,
                          num_classes: int = 3) -> torch.Tensor:
    """
    Compute a weight tensor corresponding to each label in the batch.
    Used to weight the loss per sample.
    """
    sample_weights = compute_class_weights(labels, num_classes)
    return torch.tensor(sample_weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Score rounding / threshold optimisation to maximise QWK
# ---------------------------------------------------------------------------

class ScoreRounder:
    """
    Optimises the mapping from continuous predicted scores to discrete
    grade classes by tuning bin thresholds that maximise QWK.

    Usage:
        rounder = ScoreRounder(num_classes=3)
        rounder.fit(val_preds, val_true_labels)
        discretised = rounder.predict(test_preds)
    """

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        # Default thresholds: evenly spaced
        self.thresholds = np.linspace(0, 1, num_classes + 1)[1:-1]

    def _labels_from_thresholds(self, preds: np.ndarray,
                                 thresholds: np.ndarray) -> np.ndarray:
        labels = np.zeros(len(preds), dtype=int)
        for i, t in enumerate(sorted(thresholds)):
            labels[preds >= t] = i + 1
        return labels

    def _neg_qwk(self, thresholds: np.ndarray,
                 preds: np.ndarray, true_cls: np.ndarray) -> float:
        pred_cls = self._labels_from_thresholds(preds, thresholds)
        try:
            return -cohen_kappa_score(true_cls, pred_cls, weights='quadratic')
        except Exception:
            return 0.0

    def fit(self, val_preds: np.ndarray, val_labels: np.ndarray) -> 'ScoreRounder':
        """
        Tune thresholds on validation predictions to maximise QWK.

        Args:
            val_preds:  Continuous predictions in [0, 1].
            val_labels: Continuous or integer true labels (scaled to
                        [0, num_classes-1] range allowed).
        """
        # Convert continuous labels to integers if needed
        if val_labels.max() <= 1.0:
            true_cls = np.round(val_labels * (self.num_classes - 1)).astype(int)
        else:
            true_cls = np.round(val_labels).astype(int)

        x0 = np.linspace(0.0, 1.0, self.num_classes + 1)[1:-1]

        result = minimize(
            self._neg_qwk,
            x0=x0,
            args=(val_preds, true_cls),
            method='Nelder-Mead',
            options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000}
        )

        self.thresholds = np.sort(result.x)
        fitted_cls = self._labels_from_thresholds(val_preds, self.thresholds)
        qwk = cohen_kappa_score(true_cls, fitted_cls, weights='quadratic')
        print(f"[ScoreRounder] Optimised thresholds: {self.thresholds}")
        print(f"[ScoreRounder] Validation QWK after optimisation: {qwk:.4f}")
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        """Map continuous predictions to discrete class indices."""
        return self._labels_from_thresholds(preds, self.thresholds)

    def predict_normalised(self, preds: np.ndarray) -> np.ndarray:
        """Return discrete class indices normalised back to [0, 1]."""
        cls = self.predict(preds)
        return cls / (self.num_classes - 1)


# ---------------------------------------------------------------------------
# QWK evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_qwk(preds: np.ndarray, labels: np.ndarray,
                 num_classes: int = 3) -> float:
    """Compute QWK between continuous predictions and labels."""
    pred_cls = np.round(np.clip(preds, 0, 1) * (num_classes - 1)).astype(int)
    true_cls = np.round(np.clip(labels, 0, 1) * (num_classes - 1)).astype(int)
    try:
        return cohen_kappa_score(true_cls, pred_cls, weights='quadratic')
    except Exception:
        return 0.0


def evaluate_all_metrics(preds: np.ndarray, labels: np.ndarray,
                          num_classes: int = 3) -> dict:
    """Return MSE, QWK, Accuracy, F1 (all as floats)."""
    from sklearn.metrics import (mean_squared_error, accuracy_score,
                                 f1_score, confusion_matrix)
    preds_c = np.clip(preds, 0, 1)
    mse = mean_squared_error(labels, preds_c)
    qwk = evaluate_qwk(preds_c, labels, num_classes)

    p_cls = np.round(preds_c * (num_classes - 1)).astype(int)
    t_cls = np.round(np.clip(labels, 0, 1) * (num_classes - 1)).astype(int)

    acc = accuracy_score(t_cls, p_cls)
    f1  = f1_score(t_cls, p_cls, average='weighted', zero_division=0)
    cm  = confusion_matrix(t_cls, p_cls,
                           labels=list(range(num_classes)))

    return {'mse': mse, 'qwk': qwk, 'accuracy': acc, 'f1': f1,
            'confusion_matrix': cm, 'num_classes': num_classes}
