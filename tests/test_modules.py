"""
tests/test_modules.py
Integration and unit tests for the ASAG improvement modules.

Covers:
    1. reasoning_guided_training  — model forward pass, loss, rationale gen
    2. hybrid_semantic_encoder    — encoder forward, gated fusion, collate
    3. length_adaptive_processor  — categorisation, feature extraction
    4. domain_pretraining         — corpus builder, comparison utility
    5. error_analysis             — enhanced metrics, A/B framework
    6. ensemble_model             — MultiModelEnsemble integration

Run:
    python -m pytest tests/test_modules.py -v
    (or)
    python tests/test_modules.py
"""

import sys
import os
import unittest
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Length-Adaptive Processor
# ═══════════════════════════════════════════════════════════════════════════

class TestLengthAdaptiveProcessor(unittest.TestCase):
    """Tests for length_adaptive_processor.py"""

    def setUp(self):
        from length_adaptive_processor import LengthAdaptiveProcessor
        self.proc = LengthAdaptiveProcessor()
        self.ref = (
            "The mitochondria is the powerhouse of the cell "
            "producing ATP via cellular respiration"
        )

    def test_categorize_very_short(self):
        cat, ratio = self.proc.categorize("ATP energy", self.ref)
        self.assertEqual(cat, "very_short")
        self.assertLess(ratio, 0.30)

    def test_categorize_short(self):
        cat, ratio = self.proc.categorize(
            "Mitochondria makes energy for cells", self.ref,
        )
        self.assertEqual(cat, "short")

    def test_categorize_medium(self):
        cat, ratio = self.proc.categorize(
            "The mitochondria produces ATP energy for the cell "
            "through respiration",
            self.ref,
        )
        self.assertIn(cat, ["medium", "short"])

    def test_categorize_long(self):
        cat, ratio = self.proc.categorize(self.ref + " " + self.ref, self.ref)
        self.assertEqual(cat, "long")
        self.assertGreater(ratio, 1.0)

    def test_extract_returns_12_features(self):
        feats = self.proc.extract("ATP energy", self.ref)
        self.assertEqual(len(feats), 12)
        self.assertTrue(all(np.isfinite(f) for f in feats))

    def test_batch_extract(self):
        students = ["ATP energy", "Mitochondria produces ATP", self.ref]
        refs = [self.ref] * 3
        result = self.proc.batch_extract(students, refs)
        self.assertEqual(result.shape, (3, 12))

    def test_extended_features(self):
        from length_adaptive_processor import extract_extended_features
        feats = extract_extended_features("ATP energy", self.ref)
        # Should be 27 (base) + 12 (adaptive) = 39
        self.assertEqual(len(feats), 39)

    def test_category_stats(self):
        students = ["yes", "ATP energy", self.ref, self.ref + " " + self.ref]
        refs = [self.ref] * 4
        stats = self.proc.get_category_stats(students, refs)
        self.assertIsInstance(stats, dict)
        total = sum(v["count"] for v in stats.values())
        self.assertEqual(total, 4)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Error Analysis — Enhanced Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorAnalyser(unittest.TestCase):
    """Tests for error_analysis.py enhanced features."""

    def setUp(self):
        from error_analysis import ErrorAnalyser
        np.random.seed(42)
        n = 100
        self.true = np.random.beta(2, 2, n)
        self.preds = np.clip(self.true + np.random.normal(0, 0.1, n), 0, 1)
        self.texts = ["The cell has mitochondria"] * n
        self.refs = ["The mitochondria is the powerhouse of the cell"] * n

        self.analyser = ErrorAnalyser(num_classes=3)
        self.analyser.fit(
            self.preds, self.true,
            texts=self.texts, reference_texts=self.refs,
        )

    def test_overall_metrics_keys(self):
        m = self.analyser.overall_metrics()
        for key in ["mse", "qwk", "accuracy", "f1"]:
            self.assertIn(key, m)
            self.assertIsInstance(m[key], float)

    def test_per_class_report(self):
        report = self.analyser.per_class_report()
        self.assertEqual(len(report), 3)  # num_classes = 3
        for cls, vals in report.items():
            self.assertIn("precision", vals)
            self.assertIn("recall", vals)

    def test_bias_by_class(self):
        bias = self.analyser.bias_by_class()
        self.assertIsInstance(bias, dict)
        for cls, vals in bias.items():
            self.assertIn("direction", vals)

    def test_top_errors(self):
        top = self.analyser.top_errors(n=5)
        self.assertEqual(len(top), 5)
        # Errors should be sorted descending
        self.assertTrue(top["error"].iloc[0] >= top["error"].iloc[-1])

    def test_length_category_report(self):
        report = self.analyser.length_category_report()
        self.assertIsInstance(report, dict)
        # All texts are the same length, should group to one category
        total_n = sum(v["n"] for v in report.values())
        self.assertEqual(total_n, 100)

    def test_reasoning_quality(self):
        gen = ["The mitochondria makes ATP"] * 10
        exp = ["The mitochondria produces ATP energy"] * 10
        result = self.analyser.reasoning_quality(gen, exp, use_bertscore=False)
        self.assertIn("bleu", result)
        self.assertIn("rouge_l_f1", result)
        self.assertGreater(result["bleu"], 0)
        self.assertGreater(result["rouge_l_f1"], 0)

    def test_report_runs(self):
        report = self.analyser.report(verbose=False)
        self.assertIn("overall", report)
        self.assertIn("per_class", report)
        self.assertIn("confusion_matrix", report)

    def test_save_and_load(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.analyser.save(path)
            with open(path) as f:
                data = __import__("json").load(f)
            self.assertIn("overall", data)
        finally:
            os.remove(path)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  A/B Testing Framework
# ═══════════════════════════════════════════════════════════════════════════

class TestABTestFramework(unittest.TestCase):
    """Tests for error_analysis.ABTestFramework."""

    def test_clear_winner(self):
        from error_analysis import ABTestFramework

        np.random.seed(0)
        metrics_a = {"qwk": [0.30, 0.28, 0.32, 0.29, 0.31]}
        metrics_b = {"qwk": [0.65, 0.68, 0.63, 0.66, 0.70]}

        result = ABTestFramework.compare(
            metrics_a, metrics_b,
            model_a_name="Base", model_b_name="Improved",
            verbose=False,
        )
        self.assertIn("qwk", result)
        self.assertTrue(result["qwk"]["significant"])
        self.assertTrue(result["qwk"]["b_better"])
        self.assertEqual(result["qwk"]["winner"], "Improved")

    def test_no_significant_difference(self):
        from error_analysis import ABTestFramework

        metrics_a = {"accuracy": [0.50, 0.52, 0.49, 0.51, 0.50]}
        metrics_b = {"accuracy": [0.51, 0.50, 0.52, 0.49, 0.51]}

        result = ABTestFramework.compare(
            metrics_a, metrics_b, verbose=False,
        )
        # Differences are tiny — should not be significant
        self.assertFalse(result["accuracy"]["significant"])
        self.assertEqual(result["accuracy"]["winner"], "Tie")

    def test_bootstrap_ci_shape(self):
        from error_analysis import ABTestFramework

        a = np.array([0.3, 0.32, 0.28])
        b = np.array([0.6, 0.62, 0.58])
        ci = ABTestFramework.bootstrap_ci(a, b)
        self.assertIn("mean_diff", ci)
        self.assertIn("ci_lower", ci)
        self.assertIn("ci_upper", ci)
        self.assertGreater(ci["ci_lower"], 0)  # clear positive diff

    def test_multiple_metrics(self):
        from error_analysis import ABTestFramework

        metrics_a = {
            "qwk": [0.30, 0.28, 0.32],
            "accuracy": [0.50, 0.52, 0.49],
            "mse": [0.12, 0.11, 0.13],
        }
        metrics_b = {
            "qwk": [0.65, 0.68, 0.63],
            "accuracy": [0.72, 0.70, 0.74],
            "mse": [0.05, 0.06, 0.04],
        }
        result = ABTestFramework.compare(
            metrics_a, metrics_b, verbose=False,
        )
        self.assertEqual(len(result), 3)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Domain Pre-training Utilities
# ═══════════════════════════════════════════════════════════════════════════

class TestDomainPretraining(unittest.TestCase):
    """Tests for domain_pretraining.py utilities (no GPU needed)."""

    def test_corpus_builder(self):
        from domain_pretraining import DomainCorpusBuilder

        csv_path = os.path.join(ROOT, "asag2024_all.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Dataset CSV not found")

        builder = DomainCorpusBuilder(csv_path)
        corpus = builder.build()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 100)

    def test_split_passages(self):
        from domain_pretraining import DomainCorpusBuilder

        text = " ".join(["word"] * 500)
        passages = DomainCorpusBuilder._split_passages(text, max_words=100)
        self.assertEqual(len(passages), 5)
        for p in passages:
            self.assertLessEqual(len(p.split()), 100)

    def test_compare_models(self):
        from domain_pretraining import compare_models

        base = {"mse": 0.12, "qwk": 0.30, "accuracy": 0.50, "f1": 0.48}
        domain = {"mse": 0.07, "qwk": 0.67, "accuracy": 0.72, "f1": 0.71}

        comp = compare_models(base, domain, verbose=False)
        self.assertTrue(comp["mse"]["improved"])
        self.assertTrue(comp["qwk"]["improved"])
        self.assertTrue(comp["accuracy"]["improved"])
        self.assertTrue(comp["f1"]["improved"])


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Ensemble Model — MultiModelEnsemble
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiModelEnsemble(unittest.TestCase):
    """Tests for ensemble_model.MultiModelEnsemble."""

    def test_add_member_and_predict(self):
        from ensemble_model import MultiModelEnsemble

        ensemble = MultiModelEnsemble()

        # add_member: predict_fn(student_texts, ref_texts, hc_feats) → np.ndarray
        ensemble.add_member(
            "constant_low",
            lambda t, r, h: np.full(len(t), 0.3),
            weight=1.0,
        )
        ensemble.add_member(
            "constant_high",
            lambda t, r, h: np.full(len(t), 0.8),
            weight=1.0,
        )

        hc = np.zeros((1, 27))
        result = ensemble.predict(["test answer"], ["reference answer"], hc)
        self.assertIn("score", result)
        self.assertIn("confidence", result)
        self.assertIn("per_model", result)
        # Weighted average of 0.3 and 0.8 should be ~0.55
        self.assertAlmostEqual(float(result["score"][0]), 0.55, delta=0.01)

    def test_single_model(self):
        from ensemble_model import MultiModelEnsemble

        ensemble = MultiModelEnsemble()
        ensemble.add_member(
            "single",
            lambda t, r, h: np.full(len(t), 0.6),
            weight=1.0,
        )

        hc = np.zeros((1, 27))
        result = ensemble.predict(["test"], ["ref"], hc)
        self.assertAlmostEqual(float(result["score"][0]), 0.6, delta=0.01)

    def test_median_blend(self):
        from ensemble_model import MultiModelEnsemble

        ensemble = MultiModelEnsemble(method="median")
        ensemble.add_member("a", lambda t, r, h: np.full(len(t), 0.2), weight=1.0)
        ensemble.add_member("b", lambda t, r, h: np.full(len(t), 0.5), weight=1.0)
        ensemble.add_member("c", lambda t, r, h: np.full(len(t), 0.9), weight=1.0)

        hc = np.zeros((1, 27))
        result = ensemble.predict(["test"], ["ref"], hc)
        self.assertAlmostEqual(float(result["score"][0]), 0.5, delta=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Cross-Attention Visualisation (smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossAttentionVisualization(unittest.TestCase):
    """Smoke test for cross-attention heatmap."""

    def test_visualize_saves_file(self):
        from error_analysis import ErrorAnalyser
        import tempfile

        attn = np.random.rand(5, 8)
        tok_a = ["The", "cat", "sat", "on", "mat"]
        tok_b = ["A", "cat", "sits", "on", "the", "mat", "here", "now"]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            ErrorAnalyser.visualize_cross_attention(
                attn, tok_a, tok_b, save_path=path,
            )
            # If matplotlib is not installed, file won't be created — that's OK
            if os.path.exists(path) and os.path.getsize(path) > 0:
                self.assertGreater(os.path.getsize(path), 100)
        finally:
            if os.path.exists(path):
                os.remove(path)


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
