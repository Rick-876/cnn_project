"""
score_calibration.py
Post-processing calibration to improve QWK after model inference.

Techniques implemented:
1. PlattScaler  — Sigmoid calibration to align predicted probabilities
                  with empirical class frequencies (reduces overconfidence).
2. IsotonicCalibrator — Non-parametric monotone mapping via sklearn
                        IsotonicRegression (stronger but needs more data).
3. TemperatureScaler — Single-parameter scaling of logit distribution.
4. ScoreRounder (re-exported from training_utils for convenience).

Usage example:
    from score_calibration import PlattScaler, IsotonicCalibrator

    cal = PlattScaler()
    cal.fit(val_preds, val_labels)
    calibrated_preds = cal.transform(test_preds)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_array


class PlattScaler:
    """
    Platt scaling: fit a logistic regression on (raw_score → label)
    to align continuous predictions with empirical proportions.

    Works best when the raw model output is already monotonically
    correlated with the true label (which is true for regression models).
    """

    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver='lbfgs',
                                      max_iter=1000)
        self.fitted = False

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> 'PlattScaler':
        """
        Args:
            preds:  Raw model predictions in [0, 1], shape (N,).
            labels: True continuous labels in [0, 1], shape (N,).
        """
        preds  = np.clip(preds,  0, 1).reshape(-1, 1)
        # Binarize: above median = 1, else 0 (used only for fitting shape)
        # For regression targets we use continuous labels directly via isotonic
        binary = (labels >= np.median(labels)).astype(int)
        self._lr.fit(preds, binary)
        self.fitted = True
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        """Map raw predictions to calibrated predictions in [0, 1]."""
        if not self.fitted:
            raise RuntimeError("PlattScaler must be fit before transform.")
        preds = np.clip(preds, 0, 1).reshape(-1, 1)
        return self._lr.predict_proba(preds)[:, 1]


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    More flexible than Platt — learns any monotone mapping.
    Requires a reasonable-sized validation set (≥ 200 samples).
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self._iso = IsotonicRegression(y_min=y_min, y_max=y_max,
                                       out_of_bounds='clip')
        self.fitted = False

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """
        Args:
            preds:  Raw predictions in [0, 1].
            labels: True labels in [0, 1].
        """
        self._iso.fit(np.clip(preds, 0, 1), np.clip(labels, 0, 1))
        self.fitted = True
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator must be fit before transform.")
        return self._iso.transform(np.clip(preds, 0, 1))


class TemperatureScaler:
    """
    Temperature scaling: divides the logit (log-odds) of the prediction
    by a learned temperature T.

    For score [0,1] predictions we treat logit = log(p/(1-p)).
    T > 1 → softer, more central predictions.
    T < 1 → sharper predictions pushed toward extremes.
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def _logit(self, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> 'TemperatureScaler':
        """Find T that minimises MSE between calibrated preds and labels."""
        from scipy.optimize import minimize_scalar

        logits = self._logit(np.clip(preds, 1e-7, 1 - 1e-7))
        labels = np.clip(labels, 0, 1)

        def mse_at_temp(t):
            calibrated = self._sigmoid(logits / t)
            return np.mean((calibrated - labels) ** 2)

        result = minimize_scalar(mse_at_temp, bounds=(0.1, 10.0),
                                 method='bounded')
        self.temperature = float(result.x)
        self.fitted = True
        print(f"[TemperatureScaler] Fitted temperature: {self.temperature:.4f}")
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureScaler must be fit before transform.")
        logits = self._logit(np.clip(preds, 1e-7, 1 - 1e-7))
        return self._sigmoid(logits / self.temperature)


class EnsembleCalibrator:
    """
    Combines multiple calibration methods and selects the one
    that achieves the highest QWK on the validation set.

    Supported methods: 'platt', 'isotonic', 'temperature', 'none'
    """

    def __init__(self, methods=('platt', 'isotonic', 'temperature', 'none'),
                 num_classes: int = 3):
        self.methods = methods
        self.num_classes = num_classes
        self._calibrators = {}
        self.best_method = 'none'
        self.fitted = False

    def _qwk(self, preds, labels):
        from sklearn.metrics import cohen_kappa_score
        nc = self.num_classes
        p = np.round(np.clip(preds, 0, 1) * (nc - 1)).astype(int)
        t = np.round(np.clip(labels, 0, 1) * (nc - 1)).astype(int)
        try:
            return cohen_kappa_score(t, p, weights='quadratic')
        except Exception:
            return -1.0

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> 'EnsembleCalibrator':
        best_qwk   = -999.0
        best_cal   = None
        best_name  = 'none'

        for name in self.methods:
            try:
                if name == 'platt':
                    cal = PlattScaler().fit(preds, labels)
                    cal_preds = cal.transform(preds)
                elif name == 'isotonic':
                    cal = IsotonicCalibrator().fit(preds, labels)
                    cal_preds = cal.transform(preds)
                elif name == 'temperature':
                    cal = TemperatureScaler().fit(preds, labels)
                    cal_preds = cal.transform(preds)
                else:  # 'none'
                    cal = None
                    cal_preds = np.clip(preds, 0, 1)

                qwk = self._qwk(cal_preds, labels)
                print(f"  [{name}] Val QWK = {qwk:.4f}")

                if qwk > best_qwk:
                    best_qwk  = qwk
                    best_cal  = cal
                    best_name = name
            except Exception as e:
                print(f"  [{name}] Failed: {e}")

        self._calibrators[best_name] = best_cal
        self.best_method = best_name
        self.fitted = True
        print(f"  Best calibration method: {best_name} (QWK={best_qwk:.4f})")
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("EnsembleCalibrator must be fit.")
        cal = self._calibrators.get(self.best_method)
        if cal is None:
            return np.clip(preds, 0, 1)
        return cal.transform(preds)


# ---------------------------------------------------------------------------
# Re-export ScoreRounder for convenience
# ---------------------------------------------------------------------------
from training_utils import ScoreRounder  # noqa – re-export


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(0)

    # Simulate biased model predictions (overconfident toward middle)
    true_labels = np.random.beta(2, 2, 500)
    raw_preds   = true_labels + np.random.normal(0, 0.1, 500)
    raw_preds   = np.clip(raw_preds, 0, 1)

    from training_utils import evaluate_all_metrics

    base = evaluate_all_metrics(raw_preds, true_labels)
    print(f"Before calibration: QWK={base['qwk']:.4f}, "
          f"Acc={base['accuracy']:.4f}, MSE={base['mse']:.4f}")

    cal = EnsembleCalibrator()
    cal.fit(raw_preds, true_labels)
    cal_preds = cal.transform(raw_preds)

    after = evaluate_all_metrics(cal_preds, true_labels)
    print(f"After  calibration: QWK={after['qwk']:.4f}, "
          f"Acc={after['accuracy']:.4f}, MSE={after['mse']:.4f}")

    # Optimised rounding
    rounder = ScoreRounder(num_classes=3)
    rounder.fit(cal_preds, true_labels)
    rounded = rounder.predict_normalised(cal_preds)
    final = evaluate_all_metrics(rounded, true_labels)
    print(f"After  rounding:    QWK={final['qwk']:.4f}, "
          f"Acc={final['accuracy']:.4f}, MSE={final['mse']:.4f}")
