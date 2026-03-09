"""
backend.py
FastAPI backend for the ASAG CNN grading system.
Exposes POST /predict — accepts a question + student answer,
returns predicted score, confidence, and AI feedback.

Run with:
    C:/Users/fsmith/Documents/cnn_project/.venv/Scripts/uvicorn backend:app --reload --port 8000
"""

import re
import os
import numpy as np
import pandas as pd
import torch
import asyncio
from contextlib import asynccontextmanager
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, DistilBertForMaskedLM
from content_correctness import ContentCorrectnessChecker
from grammar_detection import GrammarDetector
from feature_engineering import extract_all_features

# ── Config (must match training) ───────────────────────────────────────────────
FULL_DATA    = os.path.join(os.path.dirname(__file__), "asag2024_all.csv")
MODEL_FILE   = os.path.join(os.path.dirname(__file__), "textcnn_model.pth")
MAX_LEN      = 100
EMBED_DIM    = 300
NUM_FILTERS  = 256
FILTER_SIZES = [2, 3, 4, 5]

# ── Helpers & global placeholders ─────────────────────────────────────────────
STOPWORDS = {
    "what","is","the","a","an","of","in","to","and","or","for",
    "on","at","with","this","that","are","it","as","be","from",
    "by","was","were","has","have","had","its","do","does","did",
    "how","why","when","where","which","who","can","could","would",
    "should","may","might","will","shall","used","use","using",
    "also","about","between","into","they","them","their","your",
    "our","we","you","he","she","i","me","my","his","her","not",
    "no","so","if","than","then","such","any","all","each",
}

vocab = {}
vocab_size = 0
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REF_LOOKUP = {}
QUESTION_LIST = []  # List of all questions for fuzzy matching
tfidf_vectorizer = None

# Semantic similarity encoder (loaded at startup)
semantic_tokenizer = None
semantic_encoder = None
lemmatizer = WordNetLemmatizer()

# Content correctness and grammar detection
content_checker = None
grammar_detector = None


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def compute_ngram_overlap(student: str, reference: str, n: int) -> float:
    """Compute n-gram overlap between texts."""
    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    s_tokens = [w for w in tokenize(student) if w not in STOPWORDS]
    r_tokens = [w for w in tokenize(reference) if w not in STOPWORDS]
    
    if len(s_tokens) < n or len(r_tokens) < n:
        return 0.0
    
    s_ngrams = get_ngrams(s_tokens, n)
    r_ngrams = get_ngrams(r_tokens, n)
    
    if not r_ngrams:
        return 0.0
    
    return len(s_ngrams & r_ngrams) / len(r_ngrams)


def _get_synonyms(word: str) -> set:
    """Get all synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            if name != word and ' ' not in name:  # skip multi-word synonyms
                synonyms.add(name)
    return synonyms


def _get_related_words(word: str) -> set:
    """Get synonyms + hypernyms + hyponyms for richer matching."""
    related = _get_synonyms(word)
    for syn in wordnet.synsets(word)[:3]:  # limit to top 3 senses
        # Hypernyms (more general: "dog" → "animal")
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                if ' ' not in name:
                    related.add(name)
        # Hyponyms (more specific: "animal" → "dog")
        for hypo in syn.hyponyms()[:5]:
            for lemma in hypo.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                if ' ' not in name:
                    related.add(name)
    related.discard(word)
    return related


def synonym_overlap(student: str, reference: str) -> dict:
    """Compute semantic word overlap using WordNet synonyms.

    For each content word in the reference, checks if the student answer
    contains either the exact word or any of its synonyms/related words.
    Returns both the overlap score and which concepts were matched.
    """
    s_tokens = [w for w in tokenize(student) if w not in STOPWORDS and len(w) > 2]
    r_tokens = [w for w in tokenize(reference) if w not in STOPWORDS and len(w) > 2]

    if not r_tokens:
        return {'score': 0.0, 'matched': [], 'missed': [], 'synonym_matches': []}

    # Lemmatize for better matching ("produces" → "produce", "cells" → "cell")
    s_lemmas = {lemmatizer.lemmatize(w) for w in s_tokens}
    s_set = set(s_tokens) | s_lemmas

    # Expand student words with their synonyms for matching
    s_expanded = set(s_set)
    for w in s_tokens:
        s_expanded |= _get_synonyms(w)
        s_expanded.add(lemmatizer.lemmatize(w))

    matched = []
    missed = []
    synonym_matches = []  # track which words were matched via synonym

    for ref_word in set(r_tokens):
        ref_lemma = lemmatizer.lemmatize(ref_word)

        # Direct match (exact or lemma)
        if ref_word in s_set or ref_lemma in s_set:
            matched.append(ref_word)
            continue

        # Synonym match: check if any student word is a synonym of the ref word
        ref_synonyms = _get_related_words(ref_word) | _get_related_words(ref_lemma)
        if ref_synonyms & s_expanded:
            matched.append(ref_word)
            match_via = ref_synonyms & s_expanded
            synonym_matches.append({
                'reference_word': ref_word,
                'matched_by': list(match_via)[:3]
            })
            continue

        missed.append(ref_word)

    unique_ref = set(r_tokens)
    score = len(matched) / len(unique_ref) if unique_ref else 0.0

    return {
        'score': score,
        'matched': matched,
        'missed': missed,
        'synonym_matches': synonym_matches
    }


@torch.no_grad()
def semantic_similarity(student: str, reference: str) -> float:
    """Compute sentence-level semantic similarity using DistilBERT.

    Encodes both texts with DistilBERT, mean-pools the token embeddings,
    and returns cosine similarity. This captures meaning even when
    different words are used (e.g. 'produces energy' vs 'generates power').
    """
    if semantic_tokenizer is None or semantic_encoder is None:
        return 0.0

    try:
        enc = semantic_tokenizer(
            [student, reference],
            truncation=True, padding=True, max_length=128,
            return_tensors='pt',
        )
        out = semantic_encoder(**enc)
        # Mean pool over tokens (excluding padding)
        mask = enc['attention_mask'].unsqueeze(-1).float()  # [2, L, 1]
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)  # [2, H]
        # Cosine similarity
        sim = F.cosine_similarity(pooled[0:1], pooled[1:2]).item()
        return max(0.0, sim)  # clamp negatives
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.0


def enhanced_similarity(student: str, reference: str) -> dict:
    """Compute multiple similarity features between student answer and reference."""
    # 1. Unigram overlap (original simple method)
    r_words = {w for w in tokenize(reference) if w not in STOPWORDS and len(w) > 2}
    s_words = {w for w in tokenize(student) if w not in STOPWORDS and len(w) > 2}
    unigram_overlap = len(s_words & r_words) / len(r_words) if r_words else 0.0
    
    # 2. Bigram overlap (captures phrase-level similarity)
    bigram_overlap = compute_ngram_overlap(student, reference, 2)
    
    # 3. Trigram overlap (captures longer phrase similarity)
    trigram_overlap = compute_ngram_overlap(student, reference, 3)
    
    # 4. TF-IDF weighted similarity
    tfidf_sim = 0.0
    if tfidf_vectorizer is not None:
        try:
            vectors = tfidf_vectorizer.transform([student, reference])
            tfidf_sim = float(cosine_similarity(vectors[0:1], vectors[1:2])[0, 0])
        except:
            pass
    
    # 5. Length ratio (penalize very short answers)
    length_ratio = min(len(s_words) / max(len(r_words), 1), 1.0)

    # 6. Synonym overlap (WordNet) — matches words with similar meanings
    syn_result = synonym_overlap(student, reference)
    synonym_score = syn_result['score']

    # 7. Sentence-level semantic similarity (DistilBERT)
    sem_sim = semantic_similarity(student, reference)

    return {
        'unigram': unigram_overlap,
        'bigram': bigram_overlap,
        'trigram': trigram_overlap,
        'tfidf': tfidf_sim,
        'length_ratio': length_ratio,
        'synonym': synonym_score,
        'semantic': sem_sim,
        'synonym_detail': syn_result,
    }


# Weights used by reference_similarity (shared constant)
_SIM_WEIGHTS = {
    'unigram': 0.10,
    'bigram': 0.15,
    'trigram': 0.10,
    'tfidf': 0.15,
    'length_ratio': 0.05,
    'synonym': 0.20,
    'semantic': 0.25,
}


def reference_similarity_from_features(features: dict) -> float:
    """Compute weighted combination from pre-computed similarity features.

    Avoids re-calling enhanced_similarity() when the dict is already available.
    """
    return sum(features.get(k, 0) * w for k, w in _SIM_WEIGHTS.items())


def reference_similarity(student: str, reference: str) -> float:
    """Convenience wrapper: compute features then combine."""
    return reference_similarity_from_features(enhanced_similarity(student, reference))


def find_closest_question(query: str) -> str:
    """Find closest matching question using simple token overlap."""
    if query in REF_LOOKUP:
        return query
    
    query_tokens = {w for w in tokenize(query) if len(w) > 2}
    
    best_match = ""
    best_score = 0.0
    
    for q in QUESTION_LIST:
        q_tokens = {w for w in tokenize(q) if len(w) > 2}
        if query_tokens and q_tokens:
            overlap = len(query_tokens & q_tokens) / len(query_tokens | q_tokens)
            if overlap > best_score:
                best_score = overlap
                best_match = q
    
    # Only return match if overlap is significant
    return best_match if best_score > 0.4 else ""


# ── Model definition (must match training) ───────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=FILTER_SIZES, num_hc=27, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        conv_out_dim = num_filters * len(filter_sizes) + 1 + num_hc  # 256*4+1+27=1052
        self.fc1 = nn.Linear(conv_out_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, sim, hc):
        x = self.embedding(x).unsqueeze(1)
        convs = [F.relu(c(x)).squeeze(3) for c in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        cat = torch.cat(pooled, 1)
        cat = torch.cat([cat, sim.unsqueeze(1), hc], dim=1)
        cat = self.dropout(cat)
        cat = F.relu(self.fc1(cat))
        cat = F.dropout(cat, p=0.3, training=self.training)
        return torch.sigmoid(self.fc2(cat)).squeeze(1)


# ── Initialization (runs at app startup) ──────────────────────────────────────
def encode_text(text: str):
    ids = [vocab.get(tok, 0) for tok in tokenize(text)]
    ids = ids[:MAX_LEN] if len(ids) >= MAX_LEN else ids + [0] * (MAX_LEN - len(ids))
    return ids


def relevance_score(question: str, answer: str, reference: str = "") -> float:
    """Check if the answer is on-topic.

    Uses three signals so synonym-rich answers aren't penalised:
      1. Exact-word overlap with the question (original method)
      2. Exact-word overlap with the reference answer
      3. WordNet-expanded overlap with the reference answer
    Returns the *maximum* of these three, so a single 'hit' is enough.
    """
    q_words = {w for w in tokenize(question) if w not in STOPWORDS and len(w) > 2}
    a_words = {w for w in tokenize(answer)   if w not in STOPWORDS and len(w) > 2}
    if not q_words or not a_words:
        return 0.0

    # Signal 1: question ↔ answer word overlap (original)
    question_overlap = len(q_words & a_words) / len(q_words | a_words)

    # Signal 2: reference ↔ answer word overlap
    ref_overlap = 0.0
    if reference:
        r_words = {w for w in tokenize(reference) if w not in STOPWORDS and len(w) > 2}
        if r_words:
            ref_overlap = len(r_words & a_words) / len(r_words | a_words)

    # Signal 3: synonym-expanded overlap with reference
    syn_overlap = 0.0
    if reference:
        syn_result = synonym_overlap(answer, reference)
        syn_overlap = syn_result['score']

    return max(question_overlap, ref_overlap, syn_overlap * 0.5)




def initialize_backend():
    global vocab, vocab_size, model, REF_LOOKUP, QUESTION_LIST, tfidf_vectorizer
    global content_checker, grammar_detector
    print("Initializing backend: loading dataset, building vocab, loading model...")
    df = pd.read_csv(FULL_DATA)
    # Handle actual CSV column names
    if list(df.columns[:4]) == ['Question', 'Student Answer', 'Reference Answer', 'Human Score/Grade']:
        df = df.iloc[:, :4].copy()
        df.columns = ["question", "provided_answer", "reference_answer", "normalized_grade"]
    else:
        # fallback: assume columns are already correct
        pass
    df = df.fillna({"question": "", "reference_answer": "", "provided_answer": "", "normalized_grade": 0.0})
    df["text"] = (
        "Question: " + df["question"] + " "
        "Reference: " + df["reference_answer"] + " "
        "Student: " + df["provided_answer"]
    )

    all_words = [w for t in df["text"] for w in tokenize(t)]
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(all_words).most_common())}
    vocab_size = len(vocab) + 1
    print(f"Built vocab size: {vocab_size}")

    REF_LOOKUP = df.groupby("question")["reference_answer"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    QUESTION_LIST = list(REF_LOOKUP.keys())
    print(f"Built REF_LOOKUP for {len(REF_LOOKUP)} questions")

    # Build TF-IDF vectorizer on all reference answers
    print("Building TF-IDF vectorizer...")
    all_texts = df["reference_answer"].tolist() + df["provided_answer"].tolist()
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_vectorizer.fit(all_texts)
    print("TF-IDF vectorizer ready")

    # Load DistilBERT for semantic similarity
    global semantic_tokenizer, semantic_encoder
    print("Loading semantic encoder (DistilBERT) …")
    try:
        semantic_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        # Load with the full MLM class (matches the checkpoint exactly),
        # then keep only the base transformer — avoids UNEXPECTED-key warnings.
        _mlm = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        semantic_encoder = _mlm.distilbert
        del _mlm
        semantic_encoder.eval()
        print("  ✓ Semantic encoder ready")
    except Exception as e:
        print(f"  ✗ Semantic encoder failed: {e}")
        semantic_tokenizer = None
        semantic_encoder = None

    # Initialize model and try to load weights
    m = TextCNN(vocab_size=vocab_size).to(device)
    try:
        state = torch.load(MODEL_FILE, map_location=device)
        m.load_state_dict(state)
        m.eval()
        model = m
        print(f"Loaded model weights from {MODEL_FILE}")
    except Exception as e:
        model = m
        print(f"Warning: unable to load model weights: {e}")
    
    # Initialize content correctness checker and grammar detector
    print("Initializing content correctness checker...")
    content_checker = ContentCorrectnessChecker()
    print("  ✓ Content checker ready")
    
    print("Initializing grammar detector...")
    grammar_detector = GrammarDetector()
    print("  ✓ Grammar detector ready")


# ── Feedback & thresholds ─────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.04

FEEDBACK = {
    "off_topic": [
        "Your answer does not appear to address this question. Please read the question carefully and try again.",
        "The response seems unrelated to the question asked. Review the topic and submit a relevant answer.",
        "Off-topic response detected. Your answer should address the specific concept in the question.",
    ],
    "high": [
        "Your answer demonstrates a strong understanding of the concept. Key terms and ideas are well addressed.",
        "Excellent response. You have clearly explained the core idea with accurate details.",
        "Well done! Your answer covers the main points accurately and concisely.",
    ],
    "mid": [
        "Your answer shows some understanding but is missing key details. Try to expand on the core concept.",
        "Partially correct. Consider including more specific terminology or examples.",
        "Good start, but your answer could be more precise. Review the reference material for additional detail.",
    ],
    "low": [
        "Your answer does not address the question adequately. Please review the topic and try again.",
        "The response lacks key concepts required for this question. Consider revising your understanding.",
        "Insufficient detail. Try to address the specific mechanisms or processes asked about.",
    ],
}


def get_feedback(score_norm: float, question: str, off_topic: bool = False) -> str:
    import hashlib
    idx = int(hashlib.md5(question.encode()).hexdigest(), 16) % 3
    if off_topic:
        return FEEDBACK["off_topic"][idx]
    if score_norm >= 0.67:
        return FEEDBACK["high"][idx]
    elif score_norm >= 0.33:
        return FEEDBACK["mid"][idx]
    else:
        return FEEDBACK["low"][idx]


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Run heavy initialisation in a thread so the event-loop stays responsive."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, initialize_backend)
    yield  # app is running


app = FastAPI(title="ASAG CNN Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    question: str
    answer: str


class PredictResponse(BaseModel):
    score: float
    confidence: float
    feedback: str
    reference_answer: str


@app.get("/")
def root():
    return {"status": "ASAG backend running", "endpoint": "POST /predict"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Look up reference with fuzzy matching fallback
    question_key = find_closest_question(req.question.strip())
    reference = REF_LOOKUP.get(question_key, "")

    # If no reference found, warn but continue with empty reference
    if not reference:
        print(f"Warning: No reference found for question: {req.question[:100]}...")
        reference = ""

    # ── Relevance guard (cheap check BEFORE expensive features) ────────────
    rel = relevance_score(req.question, req.answer, reference)
    if rel < RELEVANCE_THRESHOLD:
        return PredictResponse(
            score=0.0, confidence=0.97,
            feedback=get_feedback(0.0, req.question, off_topic=True),
            reference_answer=reference,
        )

    # ── Encode input for TextCNN ───────────────────────────────────────────
    combined = (
        f"Question: {req.question} "
        f"Reference: {reference} "
        f"Student: {req.answer}"
    )
    input_ids = torch.tensor([encode_text(combined)], dtype=torch.long).to(device)

    # Compute enhanced similarity features (computed once, reused below)
    sim_features = enhanced_similarity(req.answer, reference)
    sim_score = reference_similarity_from_features(sim_features)
    sim_tensor = torch.tensor([sim_score], dtype=torch.float).to(device)

    # Compute handcrafted features (27-dim) to match training architecture
    hc_feats = extract_all_features(req.answer, reference)
    hc_tensor = torch.tensor([hc_feats], dtype=torch.float).to(device)

    with torch.no_grad():
        model_prediction = model(input_ids, sim_tensor, hc_tensor)

    model_score = float(torch.clamp(model_prediction, 0.0, 1.0).item())
    
    # ── Content Correctness Check ──────────────────────────────────────────────
    content_correct_check = None
    content_override = None
    grammar_assessment = None
    content_boost = 0.0
    grammar_tolerance = 1.0
    
    if content_checker and grammar_detector and reference:
        try:
            # Check if key words from reference appear in student answer
            content_correct_check = content_checker.check_keyword_presence(
                req.answer, reference, threshold=0.6
            )
            
            # Check for content-override (scrambled but correct answers)
            content_override = content_checker.compute_content_override_score(
                req.answer, reference
            )
            
            # Assess grammar and get tolerance factor
            grammar_assessment = grammar_detector.assess_grammar_tolerance(req.answer)
            grammar_tolerance = grammar_assessment['tolerance_score']
            
            # Apply content override for scrambled-but-correct answers
            # This is the key improvement for "energy the mitochondria cell produces for" cases
            if content_override['should_boost']:
                # Student knows the content! Give them credit
                override_score = content_override['override_score']
                
                # If model scored much lower than content suggests, apply boost
                if model_score < override_score:
                    content_boost = (override_score - model_score) * 0.6
                    print(f"Content override applied: +{content_boost:.2f} (content: {override_score:.2f}, model: {model_score:.2f})")
            
            # Also apply grammar tolerance (but don't double-penalize well-written answers)
            if grammar_tolerance < 1.0:
                model_score = model_score * grammar_tolerance
            
        except Exception as e:
            print(f"Content/grammar check error: {e}")
    
    # Calibrated scoring: blend model prediction with similarity features
    # When similarity is high, trust it more; when low, trust model more
    # This corrects for model underconfidence on high-quality answers
    similarity_weight = 0.65  # Trust similarity features significantly
    model_weight = 0.35
    
    # Use the average of top similarity features as direct evidence
    # Now includes semantic and synonym scores for meaning-level matching
    top_features = [
        sim_features['tfidf'],
        sim_features['bigram'],
        sim_features['unigram'],
        sim_features.get('synonym', 0.0),
        sim_features.get('semantic', 0.0),
    ]
    valid_features = [f for f in top_features if f > 0 and not np.isnan(f)]
    avg_similarity = np.mean(valid_features) if valid_features else 0.0

    # Log synonym matches for transparency
    syn_detail = sim_features.get('synonym_detail', {})
    if syn_detail.get('synonym_matches'):
        matches_str = "; ".join(
            f"{m['reference_word']} ← {', '.join(m['matched_by'])}"
            for m in syn_detail['synonym_matches']
        )
        print(f"Synonym matches: {matches_str}")
    
    # Ensure no NaN values
    if np.isnan(model_score):
        model_score = 0.0
    if np.isnan(avg_similarity):
        avg_similarity = 0.0
    
    # Blend scores with adaptive weighting based on agreement
    agreement = 1.0 - abs(model_score - avg_similarity)
    # When model and similarity agree, use model more; when they disagree, trust similarity
    adaptive_model_weight = model_weight + (similarity_weight - model_weight) * (1 - agreement)
    adaptive_sim_weight = 1.0 - adaptive_model_weight
    
    score_norm = (adaptive_model_weight * model_score) + (adaptive_sim_weight * avg_similarity)
    score_norm = float(np.clip(score_norm, 0.0, 1.0))
    
    # Apply content correctness boost if applicable
    if content_boost > 0:
        score_norm = min(1.0, score_norm + content_boost)
    
    # Final NaN check
    if np.isnan(score_norm) or np.isinf(score_norm):
        score_norm = 0.0
    
    # Confidence based on agreement between signals
    confidence = round(0.55 + 0.35 * agreement, 3)
    confidence = min(confidence, 0.99)
    
    score_display = round(score_norm * 2) / 2
    
    # Generate intelligent feedback with content and grammar insights
    feedback = get_feedback(score_norm, req.question)
    
    # Enhance feedback based on content correctness and grammar analysis
    if content_override and content_override['should_boost']:
        if content_override['order_check']['is_scrambled']:
            feedback = "Your answer demonstrates understanding of the key concepts. " + \
                       "The main ideas are present, though the sentence structure could be clearer."
        elif content_override['recommendation'] == 'full_credit':
            feedback = "Excellent! Your answer correctly addresses the main concepts."
    elif content_correct_check and content_correct_check['content_correct'] and score_norm < 0.7:
        feedback += " Your answer contains the key concepts correctly."
    
    if grammar_assessment and not (content_override and content_override['should_boost']):
        if grammar_assessment['recommendation'] == 'be_lenient':
            pass  # Don't add noise if already lenient
        elif grammar_assessment['violations']['severity'] >= 2:
            feedback += " Note: While your grammar could be stronger, your content is clear."

    return PredictResponse(score=score_display, confidence=confidence, feedback=feedback, reference_answer=reference)
