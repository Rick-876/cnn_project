"""
length_adaptive_processor.py
Categorises student answers by length relative to reference and applies
strategy-specific feature extraction.

Length categories:
    very_short  — < 30 % of reference word count
    short       — 30–60 %
    medium      — 60–100 %
    long        — > 100 %

Each category triggers a different feature extraction strategy:
    very_short → keyword density + concept coverage (sparse signal)
    short      → balanced keyword + light semantic
    medium     → full semantic + structural features
    long       → summarise first, then compare key points

Usage:
    from length_adaptive_processor import LengthAdaptiveProcessor
    processor = LengthAdaptiveProcessor()
    feats = processor.extract(student_answer, reference_answer)

Requires: torch, nltk, (transformers for long-answer summarisation)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LENGTH_THRESHOLDS = {
    "very_short": 0.30,
    "short": 0.60,
    "medium": 1.00,
    # everything above medium → long
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "as", "into", "through", "about",
    "between", "it", "its", "this", "that", "these", "those", "and", "or",
    "but", "not", "no", "so", "if", "than", "then", "also", "very",
}

ADAPTIVE_FEATURE_NAMES = [
    # Core (always present)
    "length_ratio",
    "length_category_code",  # 0=very_short, 1=short, 2=medium, 3=long
    # Keyword features
    "keyword_density",
    "keyword_recall",
    "concept_coverage",
    # Semantic features
    "unigram_overlap",
    "bigram_overlap",
    "trigram_overlap",
    # Structural features
    "sentence_count_ratio",
    "avg_word_length_ratio",
    # Strategy-specific
    "strategy_confidence",     # how well the strategy fits this sample
    "compression_ratio",       # for long answers: orig / summarised length
]

CATEGORY_CODE = {"very_short": 0, "short": 1, "medium": 2, "long": 3}


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenization."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def _content_words(text: str) -> List[str]:
    """Extract content words (non-stopword)."""
    return [w for w in _tokenize(text) if w not in _STOPWORDS]


def _ngrams(tokens: List[str], n: int) -> set:
    """Compute n-gram set from token list."""
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def categorize_length(
    student_answer: str, reference_answer: str,
) -> Tuple[str, float]:
    """Classify answer length relative to reference.

    Args:
        student_answer:   Student's text.
        reference_answer: Reference/model answer.

    Returns:
        (category_name, length_ratio).
    """
    s_len = len(_tokenize(student_answer))
    r_len = max(len(_tokenize(reference_answer)), 1)
    ratio = s_len / r_len

    if ratio < LENGTH_THRESHOLDS["very_short"]:
        return "very_short", ratio
    elif ratio < LENGTH_THRESHOLDS["short"]:
        return "short", ratio
    elif ratio <= LENGTH_THRESHOLDS["medium"]:
        return "medium", ratio
    else:
        return "long", ratio


# ──────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────────────────────────────────────

def _strategy_very_short(
    student: str, reference: str, ratio: float,
) -> Dict[str, float]:
    """Very short answers: focus on keyword density and concept coverage.

    Because the answer is much shorter than expected, we emphasise
    whether the few words present are high-value content words.
    """
    s_content = _content_words(student)
    r_content = _content_words(reference)

    s_set = set(s_content)
    r_set = set(r_content)

    keyword_recall = len(s_set & r_set) / max(len(r_set), 1)
    keyword_density = len(s_set) / max(len(_tokenize(student)), 1)
    concept_coverage = keyword_recall  # same for very short

    # No useful bigrams/trigrams in very short answers
    return {
        "keyword_density": keyword_density,
        "keyword_recall": keyword_recall,
        "concept_coverage": concept_coverage,
        "unigram_overlap": keyword_recall,
        "bigram_overlap": 0.0,
        "trigram_overlap": 0.0,
        "strategy_confidence": min(1.0, keyword_density + keyword_recall),
        "compression_ratio": 1.0,
    }


def _strategy_short(
    student: str, reference: str, ratio: float,
) -> Dict[str, float]:
    """Short answers: balanced keyword + light semantic analysis."""
    s_content = _content_words(student)
    r_content = _content_words(reference)

    s_set = set(s_content)
    r_set = set(r_content)

    keyword_recall = len(s_set & r_set) / max(len(r_set), 1)
    keyword_density = len(s_set) / max(len(_tokenize(student)), 1)

    # Light n-gram overlap
    s_bi = _ngrams(s_content, 2)
    r_bi = _ngrams(r_content, 2)
    bigram_overlap = len(s_bi & r_bi) / max(len(r_bi), 1)

    # Concept coverage: fraction of reference concepts mentioned
    concept_coverage = keyword_recall * 0.7 + bigram_overlap * 0.3

    return {
        "keyword_density": keyword_density,
        "keyword_recall": keyword_recall,
        "concept_coverage": concept_coverage,
        "unigram_overlap": keyword_recall,
        "bigram_overlap": bigram_overlap,
        "trigram_overlap": 0.0,
        "strategy_confidence": min(1.0, concept_coverage + ratio),
        "compression_ratio": 1.0,
    }


def _strategy_medium(
    student: str, reference: str, ratio: float,
) -> Dict[str, float]:
    """Medium answers: full semantic + structural analysis."""
    s_content = _content_words(student)
    r_content = _content_words(reference)

    s_set = set(s_content)
    r_set = set(r_content)

    keyword_recall = len(s_set & r_set) / max(len(r_set), 1)
    keyword_density = len(s_set) / max(len(_tokenize(student)), 1)

    s_bi = _ngrams(s_content, 2)
    r_bi = _ngrams(r_content, 2)
    bigram_overlap = len(s_bi & r_bi) / max(len(r_bi), 1)

    s_tri = _ngrams(s_content, 3)
    r_tri = _ngrams(r_content, 3)
    trigram_overlap = len(s_tri & r_tri) / max(len(r_tri), 1)

    concept_coverage = (
        keyword_recall * 0.4 + bigram_overlap * 0.35 + trigram_overlap * 0.25
    )

    return {
        "keyword_density": keyword_density,
        "keyword_recall": keyword_recall,
        "concept_coverage": concept_coverage,
        "unigram_overlap": keyword_recall,
        "bigram_overlap": bigram_overlap,
        "trigram_overlap": trigram_overlap,
        "strategy_confidence": min(1.0, concept_coverage + 0.3),
        "compression_ratio": 1.0,
    }


def _strategy_long(
    student: str, reference: str, ratio: float,
) -> Dict[str, float]:
    """Long answers: summarise key points, then compare.

    For answers significantly longer than the reference, we extract key
    sentences by TF-IDF relevance to the reference, then run medium-strategy
    comparison on the extracted summary.
    """
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        sent_tokenize = lambda t: t.split(".")  # noqa: E731

    sentences = sent_tokenize(student)
    if len(sentences) <= 2:
        # Not long enough to meaningfully summarise
        return _strategy_medium(student, reference, ratio)

    # Score each sentence by keyword overlap with reference
    r_set = set(_content_words(reference))
    scored = []
    for sent in sentences:
        s_set = set(_content_words(sent))
        overlap = len(s_set & r_set) / max(len(r_set), 1)
        scored.append((overlap, sent))

    # Take top 60% of sentences by relevance
    scored.sort(key=lambda x: -x[0])
    keep_n = max(1, int(len(scored) * 0.6))
    summary = " ".join(s for _, s in scored[:keep_n])

    orig_len = len(_tokenize(student))
    summary_len = max(len(_tokenize(summary)), 1)
    compression_ratio = summary_len / max(orig_len, 1)

    # Run medium strategy on summarised text
    feats = _strategy_medium(summary, reference, ratio)
    feats["compression_ratio"] = compression_ratio
    feats["strategy_confidence"] = min(
        1.0, feats["concept_coverage"] + 0.2 * compression_ratio,
    )
    return feats


_STRATEGY_MAP = {
    "very_short": _strategy_very_short,
    "short": _strategy_short,
    "medium": _strategy_medium,
    "long": _strategy_long,
}


# ──────────────────────────────────────────────────────────────────────────────
# LengthAdaptiveProcessor main class
# ──────────────────────────────────────────────────────────────────────────────

class LengthAdaptiveProcessor:
    """Processes student answers adaptively based on length category.

    Provides a uniform feature vector regardless of length category,
    with category-specific internal processing.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialise processor.

        Args:
            thresholds: Optional custom length thresholds dict.
        """
        self.thresholds = thresholds or dict(LENGTH_THRESHOLDS)

    def categorize(
        self, student_answer: str, reference_answer: str,
    ) -> Tuple[str, float]:
        """Classify answer length relative to reference.

        Args:
            student_answer:   Student's text.
            reference_answer: Reference answer text.

        Returns:
            (category, length_ratio).
        """
        return categorize_length(student_answer, reference_answer)

    def extract(
        self, student_answer: str, reference_answer: str,
    ) -> np.ndarray:
        """Extract length-adaptive features for one sample.

        Args:
            student_answer:   Student answer text.
            reference_answer: Reference answer text.

        Returns:
            np.ndarray of shape (len(ADAPTIVE_FEATURE_NAMES),).
        """
        category, ratio = self.categorize(student_answer, reference_answer)

        # Core features
        try:
            from nltk.tokenize import sent_tokenize
            s_sents = len(sent_tokenize(student_answer)) if student_answer.strip() else 1
            r_sents = max(len(sent_tokenize(reference_answer)), 1) if reference_answer.strip() else 1
        except Exception:
            s_sents = max(student_answer.count("."), 1)
            r_sents = max(reference_answer.count("."), 1)

        s_words = _tokenize(student_answer)
        r_words = _tokenize(reference_answer)
        avg_s = np.mean([len(w) for w in s_words]) if s_words else 0.0
        avg_r = np.mean([len(w) for w in r_words]) if r_words else 1.0

        # Strategy-specific features
        strategy_fn = _STRATEGY_MAP[category]
        strategy_feats = strategy_fn(student_answer, reference_answer, ratio)

        feature_dict = {
            "length_ratio": min(ratio, 2.0),
            "length_category_code": float(CATEGORY_CODE[category]),
            "keyword_density": strategy_feats["keyword_density"],
            "keyword_recall": strategy_feats["keyword_recall"],
            "concept_coverage": strategy_feats["concept_coverage"],
            "unigram_overlap": strategy_feats["unigram_overlap"],
            "bigram_overlap": strategy_feats["bigram_overlap"],
            "trigram_overlap": strategy_feats["trigram_overlap"],
            "sentence_count_ratio": min(s_sents / r_sents, 3.0),
            "avg_word_length_ratio": min(avg_s / max(avg_r, 1.0), 2.0),
            "strategy_confidence": strategy_feats["strategy_confidence"],
            "compression_ratio": strategy_feats["compression_ratio"],
        }

        return np.array(
            [feature_dict[k] for k in ADAPTIVE_FEATURE_NAMES], dtype=np.float32,
        )

    def batch_extract(
        self,
        student_answers: List[str],
        reference_answers: List[str],
    ) -> np.ndarray:
        """Extract features for a batch of samples.

        Args:
            student_answers:   List of student texts.
            reference_answers: List of reference texts.

        Returns:
            np.ndarray of shape (N, len(ADAPTIVE_FEATURE_NAMES)).
        """
        return np.vstack([
            self.extract(s, r)
            for s, r in zip(student_answers, reference_answers)
        ])

    def get_category_stats(
        self,
        student_answers: List[str],
        reference_answers: List[str],
    ) -> Dict[str, Dict]:
        """Compute category distribution statistics.

        Args:
            student_answers:   List of student texts.
            reference_answers: List of reference texts.

        Returns:
            dict mapping category → {count, pct, avg_ratio}.
        """
        cats = [self.categorize(s, r) for s, r in
                zip(student_answers, reference_answers)]

        stats: Dict[str, Dict] = {}
        total = len(cats)
        for cat_name in CATEGORY_CODE:
            entries = [(c, r) for c, r in cats if c == cat_name]
            count = len(entries)
            ratios = [r for _, r in entries]
            stats[cat_name] = {
                "count": count,
                "pct": round(count / max(total, 1) * 100, 1),
                "avg_ratio": round(float(np.mean(ratios)), 3) if ratios else 0.0,
                "min_ratio": round(float(np.min(ratios)), 3) if ratios else 0.0,
                "max_ratio": round(float(np.max(ratios)), 3) if ratios else 0.0,
            }
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# Integration: extend feature_engineering.py's feature vector
# ──────────────────────────────────────────────────────────────────────────────

def extract_extended_features(
    student_answer: str,
    reference_answer: str = "",
    normalize: bool = True,
) -> np.ndarray:
    """Extract base 27 features + 12 length-adaptive features = 39 total.

    Args:
        student_answer:   Student answer text.
        reference_answer: Reference answer text.
        normalize:        Whether to normalise the base features.

    Returns:
        np.ndarray of shape (39,).
    """
    from feature_engineering import extract_all_features

    base = extract_all_features(student_answer, reference_answer, normalize)
    processor = LengthAdaptiveProcessor()
    adaptive = processor.extract(student_answer, reference_answer)
    return np.concatenate([base, adaptive])


def batch_extract_extended_features(
    student_answers: List[str],
    reference_answers: Optional[List[str]] = None,
) -> np.ndarray:
    """Batch extraction of extended features (39-dim).

    Args:
        student_answers:   List of student texts.
        reference_answers: List of reference texts.

    Returns:
        np.ndarray of shape (N, 39).
    """
    if reference_answers is None:
        reference_answers = [""] * len(student_answers)
    return np.vstack([
        extract_extended_features(s, r)
        for s, r in zip(student_answers, reference_answers)
    ])


EXTENDED_FEATURE_NAMES = None  # populated lazily

def get_extended_feature_names() -> List[str]:
    """Get the full list of 39 feature names."""
    global EXTENDED_FEATURE_NAMES
    if EXTENDED_FEATURE_NAMES is None:
        from feature_engineering import FEATURE_NAMES
        EXTENDED_FEATURE_NAMES = list(FEATURE_NAMES) + list(ADAPTIVE_FEATURE_NAMES)
    return EXTENDED_FEATURE_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    processor = LengthAdaptiveProcessor()

    ref = (
        "The mitochondria is the powerhouse of the cell, producing ATP "
        "through cellular respiration via electron transport chain and "
        "oxidative phosphorylation."
    )

    test_cases = [
        ("ATP energy", "very_short"),
        ("Mitochondria produce ATP for the cell", "short"),
        (
            "The mitochondria produces energy for the cell through "
            "cellular respiration and ATP synthesis",
            "medium",
        ),
        (
            "The mitochondria is often referred to as the powerhouse of the "
            "cell because it is responsible for producing the majority of the "
            "cell's energy currency known as ATP or adenosine triphosphate. "
            "This process occurs through cellular respiration which includes "
            "glycolysis, the citric acid cycle also known as the Krebs cycle, "
            "and the electron transport chain during oxidative phosphorylation."
            " The mitochondria has a double membrane structure with cristae "
            "that increase the surface area for these reactions to take place.",
            "long",
        ),
    ]

    print("=" * 60)
    print("Length-Adaptive Processor Self-Test")
    print("=" * 60)

    for answer, expected_cat in test_cases:
        cat, ratio = processor.categorize(answer, ref)
        feats = processor.extract(answer, ref)
        status = "✓" if cat == expected_cat else "✗"
        print(f"\n{status} Category: {cat:12s} (expected: {expected_cat})")
        print(f"  Length ratio: {ratio:.2f}")
        print(f"  Features ({len(feats)}d):")
        for name, val in zip(ADAPTIVE_FEATURE_NAMES, feats):
            print(f"    {name:30s}: {val:.4f}")

    # Category stats
    students = [t[0] for t in test_cases]
    refs = [ref] * len(students)
    stats = processor.get_category_stats(students, refs)
    print("\nCategory distribution:")
    for cat, s in stats.items():
        print(f"  {cat:12s}: {s['count']} ({s['pct']}%)")
