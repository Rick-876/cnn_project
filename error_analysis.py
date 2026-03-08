"""
error_analysis.py
Post-training error analysis for the ASAG model.

Generates:
1.  Per-class performance breakdown (precision, recall, F1 per grade)
2.  High-error examples table (largest |pred - true| cases)
3.  Score distribution comparison (true vs predicted)
4.  Confusion matrix with human-readable summary
5.  Bias analysis — does model over/under-score for any score class?
6.  Feature correlation with error magnitude (uses feature_engineering.py)
7.  SHAP summary (optional — requires shap package)
8.  Reasoning quality metrics (BLEU, ROUGE, BERTScore)
9.  Length-category performance breakdown
10. Cross-attention alignment visualization (heatmaps)
11. A/B testing framework with statistical significance

Usage:
    from error_analysis import ErrorAnalyser

    analyser = ErrorAnalyser(num_classes=3)
    analyser.fit(preds, true_labels, texts=student_texts,
                 reference_texts=ref_texts)
    analyser.report()                  # print full report
    analyser.save('error_report.json') # save JSON summary
    analyser.top_errors(n=10)          # show worst predictions

    # New capabilities:
    analyser.reasoning_quality(generated_rationales, expected_rationales)
    analyser.length_category_report()
    analyser.visualize_cross_attention(attn_weights, tokens_a, tokens_b)
    ABTestFramework.compare(metrics_a, metrics_b)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, mean_squared_error,
    cohen_kappa_score, accuracy_score, f1_score,
    precision_recall_fscore_support
)

logger = logging.getLogger(__name__)


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
    #  8. Reasoning quality metrics                                        #
    # ------------------------------------------------------------------ #

    def reasoning_quality(
        self,
        generated: List[str],
        expected: List[str],
        use_bertscore: bool = True,
    ) -> Dict[str, float]:
        """Evaluate quality of generated reasoning rationales.

        Computes BLEU, ROUGE-L, and optionally BERTScore between
        generated and expected rationale texts.

        Args:
            generated:     List of generated rationale strings.
            expected:      List of expected/reference rationale strings.
            use_bertscore: Whether to also compute BERTScore (slower).

        Returns:
            dict with bleu, rouge_l_f1, and optionally bert_f1.
        """
        results: Dict[str, float] = {}

        # BLEU (corpus-level)
        try:
            from nltk.translate.bleu_score import (
                corpus_bleu, SmoothingFunction,
            )
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            refs_tok = [[ref.split()] for ref in expected]
            hyps_tok = [gen.split() for gen in generated]
            smooth = SmoothingFunction().method1
            bleu = corpus_bleu(
                refs_tok, hyps_tok, smoothing_function=smooth,
            )
            results["bleu"] = round(bleu, 4)
        except ImportError:
            logger.warning("nltk not installed — BLEU skipped")

        # ROUGE-L
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [
                scorer.score(ref, gen)["rougeL"].fmeasure
                for gen, ref in zip(generated, expected)
            ]
            results["rouge_l_f1"] = round(float(np.mean(scores)), 4)
        except ImportError:
            # Fallback: simple LCS-based ROUGE-L
            def _lcs_len(a: str, b: str) -> int:
                wa, wb = a.split(), b.split()
                m, n = len(wa), len(wb)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if wa[i - 1].lower() == wb[j - 1].lower():
                            dp[i][j] = dp[i - 1][j - 1] + 1
                        else:
                            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                return dp[m][n]

            rouge_scores = []
            for gen, ref in zip(generated, expected):
                lcs = _lcs_len(gen, ref)
                prec = lcs / max(len(gen.split()), 1)
                rec = lcs / max(len(ref.split()), 1)
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
                rouge_scores.append(f1)
            results["rouge_l_f1"] = round(float(np.mean(rouge_scores)), 4)

        # BERTScore
        if use_bertscore:
            try:
                from bert_score import score as bert_score_fn

                P, R, F1 = bert_score_fn(
                    generated, expected, lang="en", verbose=False,
                )
                results["bert_precision"] = round(float(P.mean()), 4)
                results["bert_recall"] = round(float(R.mean()), 4)
                results["bert_f1"] = round(float(F1.mean()), 4)
            except ImportError:
                logger.warning("bert-score not installed — BERTScore skipped")

        if results:
            print("\n── Reasoning Quality Metrics ────────────────────────")
            for k, v in results.items():
                print(f"  {k:20s}: {v:.4f}")

        return results

    # ------------------------------------------------------------------ #
    #  9. Length-category performance breakdown                             #
    # ------------------------------------------------------------------ #

    def length_category_report(self) -> Dict[str, Dict]:
        """Break down performance by answer length category.

        Categories mirror length_adaptive_processor.py:
            very_short (<30% of ref), short (30-60%), medium (60-100%),
            long (>100%).

        Returns:
            dict mapping category → {mse, qwk, accuracy, f1, n}.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before length_category_report().")

        # Compute length ratio for every sample
        ratios = []
        for stu, ref in zip(self.texts, self.reference_texts):
            ref_len = max(len(str(ref).split()), 1)
            stu_len = len(str(stu).split())
            ratios.append(stu_len / ref_len)
        self._df["length_ratio"] = ratios

        # Categorise
        def _cat(r: float) -> str:
            if r < 0.30:
                return "very_short"
            if r < 0.60:
                return "short"
            if r <= 1.00:
                return "medium"
            return "long"

        self._df["length_cat"] = [_cat(r) for r in ratios]

        results: Dict[str, Dict] = {}
        nc = self.num_classes

        print("\n── Length-Category Performance ──────────────────────")
        print(f"{'Category':>12s}  {'N':>5s}  {'MSE':>7s}  {'QWK':>7s}  "
              f"{'Acc':>7s}  {'F1':>7s}")
        print("-" * 58)

        for cat in ["very_short", "short", "medium", "long"]:
            mask = self._df["length_cat"] == cat
            n = int(mask.sum())
            if n == 0:
                results[cat] = {"n": 0, "mse": None, "qwk": None,
                                "accuracy": None, "f1": None}
                continue

            sub = self._df[mask]
            mse_val = float(mean_squared_error(sub["true"], sub["pred"]))
            pred_cls = sub["pred_cls"].values
            true_cls = sub["true_cls"].values
            try:
                qwk_val = float(cohen_kappa_score(
                    true_cls, pred_cls, weights="quadratic",
                ))
            except Exception:
                qwk_val = 0.0
            acc_val = float(accuracy_score(true_cls, pred_cls))
            f1_val = float(f1_score(
                true_cls, pred_cls, average="weighted", zero_division=0,
            ))

            results[cat] = {
                "n": n,
                "mse": round(mse_val, 4),
                "qwk": round(qwk_val, 4),
                "accuracy": round(acc_val, 4),
                "f1": round(f1_val, 4),
            }

            print(f"{cat:>12s}  {n:5d}  {mse_val:7.4f}  {qwk_val:7.4f}  "
                  f"{acc_val:7.4f}  {f1_val:7.4f}")

        return results

    # ------------------------------------------------------------------ #
    #  10. Cross-attention alignment visualization                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def visualize_cross_attention(
        attn_weights: np.ndarray,
        tokens_a: List[str],
        tokens_b: List[str],
        title: str = "Cross-Attention Alignment",
        save_path: Optional[str] = None,
    ) -> None:
        """Generate a heatmap of cross-attention weights.

        Args:
            attn_weights: 2-D array of shape (len_a, len_b) with attention
                          weights from model_a tokens attending to model_b
                          tokens. If 3-D (multi-head), averages over heads.
            tokens_a:     Token strings for the query side (rows).
            tokens_b:     Token strings for the key side (columns).
            title:        Plot title.
            save_path:    If given, save figure to this path.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning(
                "matplotlib/seaborn not installed — heatmap skipped. "
                "Install with: pip install matplotlib seaborn"
            )
            return

        # Handle multi-head: average across head dimension
        if attn_weights.ndim == 3:
            attn_weights = attn_weights.mean(axis=0)

        # Truncate for readability
        max_tok = 40
        attn_weights = attn_weights[:max_tok, :max_tok]
        tokens_a = tokens_a[:max_tok]
        tokens_b = tokens_b[:max_tok]

        fig, ax = plt.subplots(figsize=(
            max(8, len(tokens_b) * 0.45),
            max(6, len(tokens_a) * 0.35),
        ))
        sns.heatmap(
            attn_weights,
            xticklabels=tokens_b,
            yticklabels=tokens_a,
            cmap="YlOrRd",
            linewidths=0.3,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Reference tokens (keys)")
        ax.set_ylabel("Student tokens (queries)")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Attention heatmap saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)

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

            # Length-category breakdown (auto-included)
            if any(self.texts) and any(self.reference_texts):
                self.length_category_report()

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


# ======================================================================== #
#  A/B Testing Framework                                                    #
# ======================================================================== #

class ABTestFramework:
    """Compare two models (A and B) with statistical significance testing.

    Supports paired t-tests and bootstrap confidence intervals to determine
    whether observed performance differences are statistically significant.

    Usage:
        from error_analysis import ABTestFramework

        result = ABTestFramework.compare(
            metrics_a={"qwk": [...], "accuracy": [...]},  # per-fold
            metrics_b={"qwk": [...], "accuracy": [...]},
            model_a_name="Base DeBERTa",
            model_b_name="Domain-Adapted DeBERTa",
        )
    """

    @staticmethod
    def paired_ttest(
        scores_a: np.ndarray, scores_b: np.ndarray,
    ) -> Tuple[float, float]:
        """Paired t-test for dependent samples.

        Args:
            scores_a: Per-fold metric values for model A.
            scores_b: Per-fold metric values for model B.

        Returns:
            (t_statistic, p_value).
        """
        from scipy import stats

        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        return float(t_stat), float(p_val)

    @staticmethod
    def bootstrap_ci(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict:
        """Bootstrap confidence interval for mean difference (B - A).

        Args:
            scores_a:    Per-fold metric values for model A.
            scores_b:    Per-fold metric values for model B.
            n_bootstrap: Number of bootstrap samples.
            alpha:       Significance level (e.g. 0.05 for 95% CI).
            seed:        Random seed.

        Returns:
            dict with mean_diff, ci_lower, ci_upper, significant.
        """
        rng = np.random.default_rng(seed)
        diffs = scores_b - scores_a
        n = len(diffs)

        boot_means = np.array([
            rng.choice(diffs, size=n, replace=True).mean()
            for _ in range(n_bootstrap)
        ])

        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        mean_diff = float(diffs.mean())

        # Significant if CI does not include 0
        significant = (ci_lower > 0) or (ci_upper < 0)

        return {
            "mean_diff": round(mean_diff, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "significant": significant,
        }

    @classmethod
    def compare(
        cls,
        metrics_a: Dict[str, List[float]],
        metrics_b: Dict[str, List[float]],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> Dict:
        """Full A/B comparison across multiple metrics.

        Args:
            metrics_a:    dict mapping metric_name → list of per-fold values
                          for model A.
            metrics_b:    dict mapping metric_name → list of per-fold values
                          for model B.
            model_a_name: Display name for model A.
            model_b_name: Display name for model B.
            alpha:        Significance level.
            verbose:      Whether to print report.

        Returns:
            dict with per-metric comparison results.
        """
        results: Dict = {}
        common_keys = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))

        if verbose:
            print("=" * 70)
            print(f"A/B TEST: {model_a_name}  vs  {model_b_name}")
            print("=" * 70)
            print(f"  Significance level: α = {alpha}")
            print()

        for key in common_keys:
            a = np.array(metrics_a[key], dtype=float)
            b = np.array(metrics_b[key], dtype=float)

            if len(a) != len(b):
                logger.warning("Metric '%s': unequal lengths, skipped", key)
                continue
            if len(a) < 2:
                logger.warning("Metric '%s': need ≥2 folds, skipped", key)
                continue

            mean_a = float(a.mean())
            mean_b = float(b.mean())

            # Paired t-test
            try:
                t_stat, p_val = cls.paired_ttest(a, b)
            except ImportError:
                t_stat, p_val = float("nan"), float("nan")

            # Bootstrap CI
            boot = cls.bootstrap_ci(a, b, alpha=alpha)

            # For MSE, lower is better → flip improvement check
            higher_is_better = key.lower() != "mse"
            if higher_is_better:
                b_better = mean_b > mean_a
            else:
                b_better = mean_b < mean_a

            significant = (not np.isnan(p_val) and p_val < alpha) or \
                boot["significant"]

            results[key] = {
                "mean_a": round(mean_a, 4),
                "mean_b": round(mean_b, 4),
                "t_stat": round(t_stat, 4) if not np.isnan(t_stat) else None,
                "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
                "bootstrap_ci": boot,
                "significant": significant,
                "b_better": b_better,
                "winner": model_b_name if (significant and b_better) else
                    (model_a_name if (significant and not b_better) else "Tie"),
            }

            if verbose:
                sig_marker = "*" if significant else ""
                winner = results[key]["winner"]
                print(
                    f"  {key:>10s}: {model_a_name}={mean_a:.4f}  "
                    f"{model_b_name}={mean_b:.4f}  "
                    f"Δ={boot['mean_diff']:+.4f}  "
                    f"p={p_val:.4f}{sig_marker}  → {winner}"
                )
                print(
                    f"{'':>14s}95% CI: [{boot['ci_lower']:+.4f}, "
                    f"{boot['ci_upper']:+.4f}]"
                )

        if verbose:
            print("=" * 70)
            winners = [v["winner"] for v in results.values()]
            a_wins = winners.count(model_a_name)
            b_wins = winners.count(model_b_name)
            ties = winners.count("Tie")
            print(
                f"  Summary: {model_a_name} wins {a_wins}, "
                f"{model_b_name} wins {b_wins}, Ties {ties}"
            )
            print("=" * 70)

        return results


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
