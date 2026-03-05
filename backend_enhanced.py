"""
backend_enhanced.py
Improved FastAPI backend with:
- DistilBERT or ensemble model predictions
- Question-specific scoring
- Advanced reference answer scoring
- Calibrated confidence scores
- Better preprocessing

Run with:
    .venv/Scripts/uvicorn backend_enhanced:app --reload --port 8000
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import Counter

# Import improved modules
try:
    from preprocessing import preprocess_for_model, jaccard_similarity
    from reference_answers import SemanticSimilarityScorer, load_reference_database
    from scoring_improvements import FinalScorer
    from question_models import QuestionSpecificEnsemble
    from ensemble_model import EnsemblePredictor
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    print("Falling back to basic backend functionality")

# ── Configuration ──────────────────────────────────────────────────────────────
FULL_DATA = os.path.join(os.path.dirname(__file__), "asag2024_all.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "textcnn_model.pth")
REFERENCE_DB_PATH = "reference_database.json"
QUESTION_MODELS_DIR = "question_models"

# Enhanced configuration
USE_DISTILBERT = False  # Set to True once DistilBERT models are trained
USE_ENSEMBLE = False    # Set to True once ensemble models are trained
USE_QUESTION_MODELS = False  # Set to True once question models are trained
USE_REFERENCE_DB = True

MAX_LEN = 100
EMBED_DIM = 300
NUM_FILTERS = 128
FILTER_SIZES = [2, 3, 4, 5]
BATCH_SIZE = 32

# ── Global state ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original model components
vocab = {}
vocab_size = 0
model = None

# Enhanced components
reference_database = {}
question_models = None
ensemble_predictor = None
semantic_scorer = None
final_scorer = None

STOPWORDS = {
    "what","is","the","a","an","of","in","to","and","or","for","on","at","with",
    "this","that","are","it","as","be","from","by","was","were","has","have","had",
    "its","do","does","did","how","why","when","where","which","who","can","could",
    "would","should","may","might","will","shall","used","use","using","also",
    "about","between","into","they","them","their","your","our","we","you","he",
    "she","i","me","my","his","her","not","no","so","if","than","then","such",
    "any","all","each",
}


def tokenize(text: str):
    """Basic tokenization."""
    return re.findall(r"\w+", text.lower())


def encode_text(text: str):
    """Encode text using vocabulary."""
    ids = [vocab.get(tok, 0) for tok in tokenize(text)]
    ids = ids[:MAX_LEN] if len(ids) >= MAX_LEN else ids + [0] * (MAX_LEN - len(ids))
    return ids


# ── Original TextCNN model ─────────────────────────────────────────────────────
class TextCNN(torch.nn.Module):
    """Original TextCNN architecture (unchanged for backward compatibility)."""
    
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 filter_sizes=FILTER_SIZES, output_dim=1, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(num_filters * len(filter_sizes) + 1, output_dim)

    def forward(self, x, sim):
        x = self.embedding(x).unsqueeze(1)
        convs = [torch.nn.functional.relu(c(x)).squeeze(3) for c in self.convs]
        pooled = [torch.nn.functional.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        cat = torch.cat(pooled, 1)
        cat = torch.cat([cat, sim.unsqueeze(1)], dim=1)
        return self.fc(self.dropout(cat)).squeeze(1)


def compute_similarity_features(student_answer: str, reference_answer: str) -> dict:
    """Compute detailed similarity features between answers."""
    
    if not reference_answer or not student_answer:
        return {
            'unigram': 0.0,
            'bigram': 0.0,
            'trigram': 0.0,
            'semantic': 0.0,
            'tfidf': 0.0,
        }
    
    # Tokenization
    r_tokens = [w for w in tokenize(reference_answer) if w not in STOPWORDS and len(w) > 2]
    s_tokens = [w for w in tokenize(student_answer) if w not in STOPWORDS and len(w) > 2]
    r_words = set(r_tokens)
    s_words = set(s_tokens)
    
    # 1. Unigram overlap
    unigram = len(s_words & r_words) / len(r_words) if r_words else 0.0
    
    # 2. Bigram overlap
    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    bigram = 0.0
    if len(s_tokens) >= 2 and len(r_tokens) >= 2:
        s_bigrams = get_ngrams(s_tokens, 2)
        r_bigrams = get_ngrams(r_tokens, 2)
        bigram = len(s_bigrams & r_bigrams) / len(r_bigrams) if r_bigrams else 0.0
    
    # 3. Trigram overlap
    trigram = 0.0
    if len(s_tokens) >= 3 and len(r_tokens) >= 3:
        s_trigrams = get_ngrams(s_tokens, 3)
        r_trigrams = get_ngrams(r_tokens, 3)
        trigram = len(s_trigrams & r_trigrams) / len(r_trigrams) if r_trigrams else 0.0
    
    # 4. TF-IDF would require vectorizer; using weighted combination instead
    tfidf = (unigram + bigram * 0.5) / 1.5
    
    return {
        'unigram': float(unigram),
        'bigram': float(bigram),
        'trigram': float(trigram),
        'tfidf': float(tfidf),
    }


def compute_blended_similarity(features: dict) -> tuple:
    """Compute blended similarity and confidence."""
    weights = {
        'unigram': 0.15,
        'bigram': 0.20,
        'trigram': 0.15,
        'semantic': 0.35,
        'tfidf': 0.15,
    }
    
    similarity = sum(features.get(k, 0) * w for k, w in weights.items())
    
    # Higher agreement = higher confidence
    values = [features.get(k, 0) for k in ['unigram', 'bigram', 'semantic']]
    variance = np.var(values) if values else 0
    confidence = 1.0 - np.tanh(variance)  # Lower variance = higher confidence
    
    return float(np.clip(similarity, 0, 1)), float(np.clip(confidence, 0.5, 0.95))


async def initialize_backend():
    """Initialize backend at startup."""
    global vocab, vocab_size, model, reference_database, question_models, final_scorer, semantic_scorer
    
    print("="*60)
    print("Initializing Enhanced ASAG Backend")
    print("="*60)
    
    # Load dataset
    print("Loading dataset…")
    df = pd.read_csv(FULL_DATA)
    
    # Handle column names
    if list(df.columns[:4]) == ['Question', 'Student Answer', 'Reference Answer', 'Human Score/Grade']:
        df = df.iloc[:, :4].copy()
        df.columns = ["question", "provided_answer", "reference_answer", "normalized_grade"]
    
    df = df.fillna({"question": "", "reference_answer": "", "provided_answer": "", "normalized_grade": 0.0})
    df["text"] = (
        "Question: " + df["question"] + " "
        "Reference: " + df["reference_answer"] + " "
        "Student: " + df["provided_answer"]
    )
    
    # Build vocabulary
    print("Building vocabulary…")
    all_words = [w for t in df["text"] for w in tokenize(t)]
    vocab = {w: i + 1 for i, (w, _) in enumerate(Counter(all_words).most_common())}
    vocab_size = len(vocab) + 1
    print(f"  Vocabulary size: {vocab_size}")
    
    # Initialize original model
    print("Loading TextCNN model…")
    m = TextCNN(vocab_size=vocab_size).to(device)
    try:
        state = torch.load(MODEL_FILE, map_location=device)
        m.load_state_dict(state)
        m.eval()
        model = m
        print(f"  ✓ Model loaded from {MODEL_FILE}")
    except Exception as e:
        model = m
        print(f"  ⚠ Warning: Could not load model weights: {e}")
    
    # Initialize enhanced components
    print("\nInitializing enhanced components…")
    
    # Reference database
    if USE_REFERENCE_DB and os.path.exists(REFERENCE_DB_PATH):
        print("Loading reference database…")
        reference_database = load_reference_database(REFERENCE_DB_PATH)
        print(f"  ✓ Loaded {len(reference_database)} questions with references")
    else:
        print("  Reference database disabled or not found (optional)")
    
    # Question-specific models
    if USE_QUESTION_MODELS:
        print("Loading question-specific models…")
        question_models = QuestionSpecificEnsemble(save_dir=QUESTION_MODELS_DIR)
        if question_models.load_all() > 0:
            stats = question_models.get_coverage()
            print(f"  ✓ Coverage: {stats['coverage_pct']:.1f}% ({stats['trained_questions']}/{stats['total_questions']})")
        else:
            print("  ⚠ Warning: No question models found")
            question_models = None
    else:
        print("  Question-specific models disabled")
    
    # Semantic similarity scorer
    print("Initializing semantic similarity scorer…")
    semantic_scorer = SemanticSimilarityScorer(use_bert_score=False)  # BERTScore optional
    print(f"  ✓ Scorer ready")
    
    # Final scorer with calibration
    print("Initializing final scoring system…")
    final_scorer = FinalScorer()
    print(f"  ✓ Scoring system ready")
    
    print("\n" + "="*60)
    print("Backend initialization complete!")
    print("="*60)


# ── FastAPI Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ASAG Enhanced Backend",
    version="2.0.0",
    description="Automatic Short Answer Grading with improved models and scoring"
)

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


@app.on_event("startup")
async def startup_event():
    """Initialize backend on startup."""
    await initialize_backend()


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ASAG enhanced backend running",
        "version": "2.0.0",
        "endpoint": "POST /predict"
    }


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vocab_size": vocab_size,
        "reference_database": len(reference_database),
        "question_models_enabled": USE_QUESTION_MODELS,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Grade a student answer.
    
    Enhanced pipeline:
    1. Preprocessing
    2. CNN model prediction
    3. Similarity feature computation
    4. Question-specific model prediction (if available)
    5. Ensemble blending (if available)
    6. Final scoring with calibration
    """
    
    # Preprocess inputs
    question = req.question.strip()
    answer = req.answer.strip()
    
    if not question or not answer:
        return PredictResponse(
            score=0.0,
            confidence=0.95,
            feedback="Please provide both a question and an answer."
        )
    
    # Find reference answer
    reference = ""
    if reference_database and question in reference_database:
        refs = reference_database[question].get('references', [])
        reference = refs[0] if refs else ""
    
    # If no reference in DB, use simple matching
    if not reference:
        # This would need a question lookup table; for now using first reference
        pass
    
    # ──  Original CNN prediction ────────────────────────────────────────────────
    combined_text = f"Question: {question} Reference: {reference} Student: {answer}"
    input_ids = torch.tensor([encode_text(combined_text)], dtype=torch.long).to(device)
    
    # Compute similarity score for model input
    sim_features = compute_similarity_features(answer, reference)
    sim_score = sim_features['unigram']  # Use unigram for original model
    sim_tensor = torch.tensor([sim_score], dtype=torch.float).to(device)
    
    with torch.no_grad():
        cnn_pred_raw = model(input_ids, sim_tensor)
    
    cnn_pred = float(torch.clamp(cnn_pred_raw, 0.0, 1.0).item())
    
    # ──  Similarity-based scoring ──────────────────────────────────────────────
    similarity_score, similarity_confidence = compute_blended_similarity(sim_features)
    
    # CNN confidence (simple: based on prediction magnitude)
    cnn_confidence = 0.5 + abs(cnn_pred - 0.5) * 0.3 + 0.2
    cnn_confidence = float(np.clip(cnn_confidence, 0.5, 0.95))
    
    # ──  Question-specific model (if available) ────────────────────────────────
    question_specific_pred = None
    if USE_QUESTION_MODELS and question_models is not None:
        try:
            question_specific_pred = question_models.predict(question, answer)
        except:
            question_specific_pred = None
    
    # ──  Ensemble prediction (if available) ──────────────────────────────────
    ensemble_pred = None
    if USE_ENSEMBLE and ensemble_predictor is not None:
        try:
            # Would need actual ensemble predictions
            pass
        except:
            ensemble_pred = None
    
    # ──  Final scoring with calibration ───────────────────────────────────────
    if final_scorer:
        result = final_scorer.score_answer(
            question=question,
            answer=answer,
            cnn_pred=cnn_pred,
            cnn_confidence=cnn_confidence,
            similarity_score=similarity_score,
            similarity_confidence=similarity_confidence,
            question_specific_pred=question_specific_pred,
            ensemble_pred=ensemble_pred
        )
        
        # Handle off-topic responses
        if result.get('relevance_score', 1.0) < 0.04:
            return PredictResponse(
                score=0.0,
                confidence=0.97,
                feedback="Off-topic response detected. Your answer should address the specific concept in the question."
            )
        
        final_score = result['score']
        final_confidence = result['confidence']
        feedback = result['feedback']
    else:
        # Fallback if final scorer not initialized
        final_score = (0.35 * cnn_pred + 0.65 * similarity_score)
        final_confidence = (0.35 * cnn_confidence + 0.65 * similarity_confidence)
        
        if final_score >= 0.67:
            feedback = "Well done! Your answer covers the main points accurately."
        elif final_score >= 0.33:
            feedback = "Good start, but your answer could be more precise. Review the reference material."
        else:
            feedback = "Your answer does not adequately address the question."
    
    # Round score for display
    score_display = round(final_score * 2) / 2
    
    return PredictResponse(
        score=score_display,
        confidence=round(final_confidence, 2),
        feedback=feedback
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
