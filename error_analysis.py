"""
error_analysis.py
Post-training error analysis for the ASAG model.

Generates:
1. Per-class performance breakdown (precision, recall, F1 per grade)
2. High-error examples table (largest |pred - true| cases)
3. Score distribution comparison (true vs predicted)
4. Confusion matrix with human-readable summary
5. Bias analysis — does model over/under-score for any score class?
6. Feature correlation with error magnitude (uses feature_engineering.py)
7. SHAP summary (optional — requires shap package)

Usage:
    from error_analysis import ErrorAnalyser

    analyser = ErrorAnalyser(num_classes=3)
    analyser.fit(preds, true_labels, texts=student_texts,
                 reference_texts=ref_texts)
    analyser.report()                  # print full report
    analyser.save('error_report.json') # save JSON summary
    analyser.top_errors(n=10)          # show worst predictions
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, mean_squared_error,
    cohen_kappa_score, accuracy_score, f1_score,
    precision_recall_fscore_support
)


class ErrorAnalyser:
    """
    Comprehensive error analysis for regression-based short-answer grading.

    Args:
        num_classes: Number of discrete grade classes (default 3: 0, 1, 2).
    """

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.fitted = False

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self, preds: np.ndarray, true_labels: np.ndarray,
            texts: list = None,
            reference_texts: list = None,
            question_ids: list = None) -> 'ErrorAnalyser':
        """
        Store predictions and compute all error metrics.

        Args:
            preds:           Continuous predictions in [0, 1].
            true_labels:     Continuous true labels in [0, 1].
            texts:           Optional list of student answer texts.
            reference_texts: Optional list of reference answer texts.
            question_ids:    Optional list of question identifiers.
        """
        nc = self.num_classes
        self.preds       = np.clip(preds, 0, 1)
        self.true_labels = np.clip(true_labels, 0, 1)

        # Discrete class labels
        self.pred_cls = np.round(self.preds * (nc - 1)).astype(int)
        self.true_cls = np.round(self.true_labels * (nc - 1)).astype(int)

        self.texts           = texts or [''] * len(preds)
        self.reference_texts = reference_texts or [''] * len(preds)
        self.question_ids    = question_ids

        # Absolute error per sample
        self.errors = np.abs(self.preds - self.true_labels)

        # Build a DataFrame for easy slicing
        self._df = pd.DataFrame({
            'pred':      self.preds,
            'true':      self.true_labels,
            'pred_cls':  self.pred_cls,
            'true_cls':  self.true_cls,
            'error':     self.errors,
            'student':   self.texts,
            'reference': self.reference_texts,
        })
        if question_ids:
            self._df['question_id'] = question_ids

        self.fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Core metrics                                                        #
    # ------------------------------------------------------------------ #

    def overall_metrics(self) -> dict:
        """Return dict of MSE, QWK, Accuracy, F1."""
        mse = float(mean_squared_error(self.true_labels, self.preds))
        try:
            qwk = float(cohen_kappa_score(self.true_cls, self.pred_cls,
                                           weights='quadratic'))
        except Exception:
            qwk = 0.0
        acc = float(accuracy_score(self.true_cls, self.pred_cls))
        f1  = float(f1_score(self.true_cls, self.pred_cls,
                              average='weighted', zero_division=0))
        return {'mse': mse, 'qwk': qwk, 'accuracy': acc, 'f1': f1}

    def per_class_report(self) -> dict:
        """Per-class precision, recall, F1, support."""
        labels = list(range(self.num_classes))
        prec, rec, f1, sup = precision_recall_fscore_support(
            self.true_cls, self.pred_cls,
            labels=labels, zero_division=0
        )
        return {
            f'class_{i}': {
                'precision': float(prec[i]),
                'recall':    float(rec[i]),
                'f1':        float(f1[i]),
                'support':   int(sup[i]),
            }
            for i in labels
        }

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.true_cls, self.pred_cls,
                                 labels=list(range(self.num_classes)))

    # ------------------------------------------------------------------ #
    #  Bias analysis                                                       #
    # ------------------------------------------------------------------ #

    def bias_by_class(self) -> dict:
        """
        For each true class: mean prediction, mean error, over/under-scoring.

        Returns:
            dict mapping true_class → {'mean_pred', 'mean_error', 'bias'}
        """
        result = {}
        for cls in range(self.num_classes):
            mask = self.true_cls == cls
            if mask.sum() == 0:
                continue
            mean_pred  = float(self.preds[mask].mean())
            mean_true  = float(self.true_labels[mask].mean())
            mean_error = float(self.errors[mask].mean())
            bias = mean_pred - mean_true  # + = over-grading, - = under-grading
            result[f'class_{cls}'] = {
                'n': int(mask.sum()),
                'mean_true_score': round(mean_true, 4),
                'mean_pred_score': round(mean_pred, 4),
                'mean_error':      round(mean_error, 4),
                'bias':            round(bias, 4),
                'direction':       'over-grades' if bias > 0.05 else
                                   ('under-grades' if bias < -0.05 else 'balanced'),
            }
        return result

    # ------------------------------------------------------------------ #
    #  High-error examples                                                 #
    # ------------------------------------------------------------------ #

    def top_errors(self, n: int = 20) -> pd.DataFrame:
        """
        Return the n examples with the highest absolute prediction error.
        """
        top = self._df.nlargest(n, 'error')[
            ['error', 'true', 'pred', 'true_cls', 'pred_cls',
             'student', 'reference']
        ].copy()
        top['error'] = top['error'].round(4)
        top['true']  = top['true'].round(4)
        top['pred']  = top['pred'].round(4)
        return top.reset_index(drop=True)

    def low_recall_classes(self, threshold: float = 0.5) -> list:
        """Return class indices where recall < threshold."""
        report = self.per_class_report()
        return [
            int(k.split('_')[1])
            for k, v in report.items()
            if v['recall'] < threshold
        ]

    # ------------------------------------------------------------------ #
    #  Feature correlation with error (uses handcrafted features)         #
    # ------------------------------------------------------------------ #

    def feature_error_correlation(self) -> dict:
        """
        Compute Pearson correlation between each handcrafted feature and
        the absolute prediction error.  Highlights features where
        high/low values predict model mistakes.
        """
        try:
            from feature_engineering import (batch_extract_features,
                                              FEATURE_NAMES)
        except ImportError:
            return {}

        if not any(self.texts):
            return {}

        print("Extracting features for correlation analysis …")
        feats = batch_extract_features(self.texts, self.reference_texts)
        errors = self.errors

        correlations = {}
        for i, name in enumerate(FEATURE_NAMES):
            col = feats[:, i]
            if col.std() < 1e-9:
                correlations[name] = 0.0
                continue
            # Pearson correlation
            r = float(np.corrcoef(col, errors)[0, 1])
            correlations[name] = round(r, 4)

        # Sort by absolute correlation
        return dict(sorted(correlations.items(),
                            key=lambda x: abs(x[1]), reverse=True))

    # ------------------------------------------------------------------ #
    #  SHAP analysis (optional)                                            #
    # ------------------------------------------------------------------ #

    def shap_summary(self, model, val_texts, val_refs,
                     sample_size: int = 100) -> None:
        """
        Generate a SHAP feature importance summary for a sklearn-compatible
        model that accepts handcrafted feature vectors.

        Args:
            model:      Any sklearn-compatible model with predict().
            val_texts:  List of student answer texts (for feature extraction).
            val_refs:   List of reference answer texts.
            sample_size: Number of samples to use for SHAP (keep small for speed).
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return

        from feature_engineering import batch_extract_features, FEATURE_NAMES

        indices = np.random.choice(len(val_texts), size=min(sample_size,
                                   len(val_texts)), replace=False)
        sample_texts = [val_texts[i] for i in indices]
        sample_refs  = [val_refs[i]  for i in indices]

        X = batch_extract_features(sample_texts, sample_refs)
        explainer = shap.Explainer(model.predict, X,
                                    feature_names=FEATURE_NAMES)
        shap_vals = explainer(X)

        print("\nSHAP Feature Importance Summary:")
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        for name, val in sorted(zip(FEATURE_NAMES, mean_abs),
                                  key=lambda x: -x[1])[:15]:
            bar = '█' * int(val * 200)
            print(f"  {name:30s} {bar} {val:.4f}")

    # ------------------------------------------------------------------ #
    #  Full report                                                         #
    # ------------------------------------------------------------------ #

    def report(self, verbose: bool = True) -> dict:
        """Print and return a comprehensive error analysis report."""
        if not self.fitted:
            raise RuntimeError("Call fit() before report().")

        overall  = self.overall_metrics()
        per_cls  = self.per_class_report()
        cm       = self.confusion_matrix()
        bias     = self.bias_by_class()
        low_rec  = self.low_recall_classes()

        if verbose:
            nc = self.num_classes
            sep = '=' * 60
            print(sep)
            print("ASAG ERROR ANALYSIS REPORT")
            print(sep)

            print("\n── Overall Metrics ──────────────────────────────────")
            print(f"  MSE:       {overall['mse']:.4f}  (target < 0.08)")
            print(f"  QWK:       {overall['qwk']:.4f}  (target > 0.65)")
            print(f"  Accuracy:  {overall['accuracy']:.4f}  (target > 0.70)")
            print(f"  F1:        {overall['f1']:.4f}  (target > 0.70)")

            print("\n── Confusion Matrix ─────────────────────────────────")
            class_labels = [f"C{i}" for i in range(nc)]
            header = "      " + "  ".join(f"{l:>5}" for l in class_labels)
            print(header)
            for i, row in enumerate(cm):
                print(f"  {class_labels[i]}: " +
                      "  ".join(f"{v:>5}" for v in row))

            print("\n── Per-Class Performance ────────────────────────────")
            for cls_name, vals in per_cls.items():
                print(f"  {cls_name}: P={vals['precision']:.3f}  "
                      f"R={vals['recall']:.3f}  "
                      f"F1={vals['f1']:.3f}  "
                      f"n={vals['support']}")

            if low_rec:
                print(f"\n  ⚠ Low-recall classes (< 0.5): {low_rec}")
                print("    → Consider oversampling or class weighting for these.")

            print("\n── Bias Analysis ────────────────────────────────────")
            for cls_name, vals in bias.items():
                arrow = '↑' if vals['direction'] == 'over-grades' else \
                        ('↓' if vals['direction'] == 'under-grades' else '≈')
                print(f"  {cls_name}: mean_true={vals['mean_true_score']:.3f}  "
                      f"mean_pred={vals['mean_pred_score']:.3f}  "
                      f"bias={vals['bias']:+.3f}  {arrow} {vals['direction']}")

            print("\n── Top-5 Worst Predictions ──────────────────────────")
            top5 = self.top_errors(n=5)
            for _, row in top5.iterrows():
                print(f"  error={row['error']:.3f}  "
                      f"true={row['true']:.3f}  "
                      f"pred={row['pred']:.3f}")
                stu_preview = str(row['student'])[:80]
                print(f"    student: \"{stu_preview}\"")

            print("\n── Feature–Error Correlations ───────────────────────")
            corrs = self.feature_error_correlation()
            if corrs:
                top_corrs = list(corrs.items())[:8]
                for name, r in top_corrs:
                    direction = 'longer/richer = more errors' if r > 0 \
                                else 'shorter/simpler = more errors'
                    print(f"  {name:30s}: r={r:+.4f}  ({direction})")
            else:
                print("  (Feature texts not provided — skipped)")

            print(f"\n{sep}")

        full_report = {
            'overall': overall,
            'per_class': per_cls,
            'confusion_matrix': cm.tolist(),
            'bias': bias,
            'low_recall_classes': low_rec,
        }
        return full_report

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #

    def save(self, path: str = 'error_report.json') -> None:
        """Save the error report to a JSON file."""
        report = self.report(verbose=False)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Error report saved → {path}")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(0)
    n = 500
    true  = np.random.beta(2, 2, n)
    preds = np.clip(true + np.random.normal(0, 0.15, n), 0, 1)

    texts = [
        "The mitochondria produces ATP energy for the cell through respiration"
    ] * n

    refs = [
        "The mitochondria is the powerhouse of the cell producing ATP via cellular respiration"
    ] * n

    analyser = ErrorAnalyser(num_classes=3)
    analyser.fit(preds, true, texts=texts, reference_texts=refs)
    analyser.report()
    analyser.save('error_report_demo.json')
    print("\nTop-10 errors:")
    print(analyser.top_errors(n=10)[['error', 'true', 'pred']].to_string())
