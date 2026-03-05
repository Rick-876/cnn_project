"""
feature_engineering.py
Handcrafted linguistic feature extraction for ASAG.

Extracts:
- Essay length features (word count, sentence count, avg sentence length)
- Readability metrics (Flesch Reading Ease, Flesch-Kincaid Grade Level)
- Lexical diversity (type-token ratio, vocabulary richness)
- Grammar indicators (verb/noun density, modifier ratio)
- Structural features (paragraph count, intro/conclusion presence)
- Similarity features (n-gram overlap, TF-IDF cosine similarity)
- Content alignment with reference answer
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

STOP_WORDS = set(stopwords.words('english'))

# ---------------------------------------------------------------------------
# Syllable estimation (pure Python, no extra deps)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Approximate syllable count for a word."""
    word = word.lower().strip(".,!?;:'\"()-")
    if not word:
        return 0
    vowels = 'aeiouy'
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    # Silent 'e' at end
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def extract_length_features(text: str) -> Dict[str, float]:
    """
    Extract essay length and structural features.

    Returns:
        word_count, sentence_count, avg_sentence_length,
        char_count, avg_word_length, paragraph_count
    """
    if not text or not text.strip():
        return {k: 0.0 for k in ['word_count', 'sentence_count',
                                  'avg_sentence_length', 'char_count',
                                  'avg_word_length', 'paragraph_count']}

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    sentences = sent_tokenize(text) if text.strip() else []
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    char_count = sum(len(w) for w in words)

    return {
        'word_count': float(word_count),
        'sentence_count': float(sentence_count),
        'avg_sentence_length': float(word_count) / sentence_count,
        'char_count': float(char_count),
        'avg_word_length': float(char_count) / max(word_count, 1),
        'paragraph_count': float(max(len(paragraphs), 1)),
    }


def extract_readability_features(text: str) -> Dict[str, float]:
    """
    Compute readability metrics.

    Returns:
        flesch_reading_ease  – higher = easier (0–100 scale)
        flesch_kincaid_grade – US grade level
        gunning_fog          – years of education needed
    """
    if not text or not text.strip():
        return {'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'gunning_fog': 0.0}

    words = re.findall(r'\b[a-zA-Z]+\b', text)
    sentences = sent_tokenize(text)

    word_count = max(len(words), 1)
    sentence_count = max(len(sentences), 1)
    syllable_count = sum(_count_syllables(w) for w in words)
    complex_words = sum(1 for w in words if _count_syllables(w) >= 3)

    # Flesch Reading Ease
    fre = (206.835
           - 1.015 * (word_count / sentence_count)
           - 84.6 * (syllable_count / word_count))
    fre = max(0.0, min(100.0, fre))

    # Flesch–Kincaid Grade Level
    fkg = (0.39 * (word_count / sentence_count)
           + 11.8 * (syllable_count / word_count)
           - 15.59)
    fkg = max(0.0, fkg)

    # Gunning Fog Index
    fog = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))

    return {
        'flesch_reading_ease': fre,
        'flesch_kincaid_grade': fkg,
        'gunning_fog': fog,
    }


def extract_lexical_diversity(text: str) -> Dict[str, float]:
    """
    Compute lexical diversity metrics.

    Returns:
        type_token_ratio     – unique types / total tokens
        vocab_richness       – log(unique) / log(total)  [Herdan's C]
        content_word_ratio   – non-stopword fraction
    """
    if not text or not text.strip():
        return {'type_token_ratio': 0.0,
                'vocab_richness': 0.0,
                'content_word_ratio': 0.0}

    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not tokens:
        return {'type_token_ratio': 0.0,
                'vocab_richness': 0.0,
                'content_word_ratio': 0.0}

    unique_tokens = set(tokens)
    n = len(tokens)
    v = len(unique_tokens)
    content_words = [t for t in tokens if t not in STOP_WORDS]

    ttr = v / n
    richness = math.log(v) / math.log(n) if n > 1 and v > 1 else 0.0
    cwr = len(content_words) / n

    return {
        'type_token_ratio': ttr,
        'vocab_richness': richness,
        'content_word_ratio': cwr,
    }


def extract_pos_features(text: str) -> Dict[str, float]:
    """
    Extract part-of-speech density features.

    Returns:
        verb_ratio, noun_ratio, adj_ratio, adv_ratio,
        modifier_ratio  (adj + adv together)
    """
    if not text or not text.strip():
        return {k: 0.0 for k in ['verb_ratio', 'noun_ratio',
                                   'adj_ratio', 'adv_ratio', 'modifier_ratio']}
    try:
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
    except Exception:
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        tagged = [(t, 'NN') for t in tokens]

    n = max(len(tagged), 1)
    verbs = sum(1 for _, tag in tagged if tag.startswith('VB'))
    nouns = sum(1 for _, tag in tagged if tag.startswith('NN'))
    adjs  = sum(1 for _, tag in tagged if tag.startswith('JJ'))
    advs  = sum(1 for _, tag in tagged if tag.startswith('RB'))

    return {
        'verb_ratio': verbs / n,
        'noun_ratio': nouns / n,
        'adj_ratio': adjs / n,
        'adv_ratio': advs / n,
        'modifier_ratio': (adjs + advs) / n,
    }


def extract_structural_features(text: str) -> Dict[str, float]:
    """
    Detect structural elements: intro, body, conclusion.

    Returns:
        has_introduction, has_conclusion, has_examples,
        uses_connectors, connector_density
    """
    if not text or not text.strip():
        return {k: 0.0 for k in ['has_introduction', 'has_conclusion',
                                   'has_examples', 'uses_connectors',
                                   'connector_density']}

    text_lower = text.lower()
    sentences = sent_tokenize(text)

    intro_phrases = ['first', 'firstly', 'to begin', 'introduction', 'in summary',
                     'the answer', 'this is', 'one of', 'a key']
    conclusion_phrases = ['therefore', 'in conclusion', 'thus', 'hence', 'overall',
                          'in summary', 'finally', 'to conclude', 'consequently']
    example_phrases = ['for example', 'for instance', 'such as', 'e.g.', 'i.e.',
                       'including', 'like', 'namely']
    connectors = ['however', 'moreover', 'furthermore', 'additionally', 'although',
                  'because', 'while', 'whereas', 'therefore', 'thus', 'since',
                  'as a result', 'in contrast', 'on the other hand']

    has_intro = float(any(p in text_lower for p in intro_phrases))
    has_conc  = float(any(p in text_lower for p in conclusion_phrases))
    has_ex    = float(any(p in text_lower for p in example_phrases))

    connector_count = sum(text_lower.count(c) for c in connectors)
    word_count = max(len(re.findall(r'\b\w+\b', text)), 1)

    return {
        'has_introduction': has_intro,
        'has_conclusion': has_conc,
        'has_examples': has_ex,
        'uses_connectors': float(connector_count > 0),
        'connector_density': min(connector_count / word_count * 100, 1.0),
    }


def extract_similarity_features(student: str, reference: str) -> Dict[str, float]:
    """
    Compute similarity between student answer and reference answer.

    Returns:
        unigram_overlap, bigram_overlap, trigram_overlap,
        keyword_recall, length_ratio
    """
    if not student or not reference:
        return {k: 0.0 for k in ['unigram_overlap', 'bigram_overlap',
                                   'trigram_overlap', 'keyword_recall',
                                   'length_ratio']}

    def tokenize_content(t):
        return [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', t)
                if w.lower() not in STOP_WORDS and len(w) > 2]

    def ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    s_toks = tokenize_content(student)
    r_toks = tokenize_content(reference)

    if not r_toks:
        return {k: 0.0 for k in ['unigram_overlap', 'bigram_overlap',
                                   'trigram_overlap', 'keyword_recall',
                                   'length_ratio']}

    s_set = set(s_toks)
    r_set = set(r_toks)

    uni_overlap = len(s_set & r_set) / len(r_set)

    s_bi = ngrams(s_toks, 2)
    r_bi = ngrams(r_toks, 2)
    bi_overlap = len(s_bi & r_bi) / max(len(r_bi), 1)

    s_tri = ngrams(s_toks, 3)
    r_tri = ngrams(r_toks, 3)
    tri_overlap = len(s_tri & r_tri) / max(len(r_tri), 1)

    s_words_all = set(re.findall(r'\b[a-zA-Z]+\b', student.lower()))
    r_words_all = set(re.findall(r'\b[a-zA-Z]+\b', reference.lower()))
    keyword_recall = len(s_words_all & r_words_all) / max(len(r_words_all), 1)

    s_len = len(re.findall(r'\b\w+\b', student))
    r_len = max(len(re.findall(r'\b\w+\b', reference)), 1)
    length_ratio = min(s_len / r_len, 2.0)  # cap at 2× reference length

    return {
        'unigram_overlap': uni_overlap,
        'bigram_overlap': bi_overlap,
        'trigram_overlap': tri_overlap,
        'keyword_recall': keyword_recall,
        'length_ratio': length_ratio,
    }


# ---------------------------------------------------------------------------
# Composite feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Length
    'word_count', 'sentence_count', 'avg_sentence_length',
    'char_count', 'avg_word_length', 'paragraph_count',
    # Readability
    'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
    # Lexical diversity
    'type_token_ratio', 'vocab_richness', 'content_word_ratio',
    # POS
    'verb_ratio', 'noun_ratio', 'adj_ratio', 'adv_ratio', 'modifier_ratio',
    # Structure
    'has_introduction', 'has_conclusion', 'has_examples',
    'uses_connectors', 'connector_density',
    # Similarity
    'unigram_overlap', 'bigram_overlap', 'trigram_overlap',
    'keyword_recall', 'length_ratio',
]


def extract_all_features(student_answer: str,
                         reference_answer: str = "",
                         normalize: bool = True) -> np.ndarray:
    """
    Extract the full handcrafted feature vector for one student answer.

    Args:
        student_answer:   Student's text.
        reference_answer: Reference/model answer (used for similarity features).
        normalize:        Clip extreme values to reasonable ranges.

    Returns:
        np.ndarray of shape (27,) – order matches FEATURE_NAMES.
    """
    feats = {}
    feats.update(extract_length_features(student_answer))
    feats.update(extract_readability_features(student_answer))
    feats.update(extract_lexical_diversity(student_answer))
    feats.update(extract_pos_features(student_answer))
    feats.update(extract_structural_features(student_answer))
    feats.update(extract_similarity_features(student_answer, reference_answer))

    vec = np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)

    if normalize:
        # Clip outlier-prone features to sane ranges
        # word_count (idx 0): cap at 200
        vec[0] = min(vec[0], 200.0) / 200.0
        # sentence_count (idx 1): cap at 20
        vec[1] = min(vec[1], 20.0) / 20.0
        # avg_sentence_length (idx 2): cap at 50
        vec[2] = min(vec[2], 50.0) / 50.0
        # char_count (idx 3): cap at 1500
        vec[3] = min(vec[3], 1500.0) / 1500.0
        # avg_word_length (idx 4): cap at 15
        vec[4] = min(vec[4], 15.0) / 15.0
        # paragraph_count (idx 5): cap at 10
        vec[5] = min(vec[5], 10.0) / 10.0
        # flesch_reading_ease (idx 6): already 0–100, divide by 100
        vec[6] = vec[6] / 100.0
        # flesch_kincaid_grade (idx 7): cap at 20
        vec[7] = min(vec[7], 20.0) / 20.0
        # gunning_fog (idx 8): cap at 25
        vec[8] = min(vec[8], 25.0) / 25.0
        # indices 9–26 are already in [0, 1] range by construction

    return vec


def batch_extract_features(student_answers: List[str],
                            reference_answers: Optional[List[str]] = None) -> np.ndarray:
    """
    Extract features for a list of student answers.

    Args:
        student_answers:   List of student texts.
        reference_answers: Optional list of corresponding reference answers.

    Returns:
        np.ndarray of shape (N, 27).
    """
    if reference_answers is None:
        reference_answers = [''] * len(student_answers)

    return np.vstack([
        extract_all_features(s, r)
        for s, r in zip(student_answers, reference_answers)
    ])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ref = "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration."
    stu = "Mitochondria produce ATP energy for the cell using cellular respiration processes."

    vec = extract_all_features(stu, ref)
    print(f"Feature vector length: {len(vec)}")
    print("\nFeature values:")
    for name, val in zip(FEATURE_NAMES, vec):
        print(f"  {name:30s}: {val:.4f}")
