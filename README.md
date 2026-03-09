# Automatic Short Answer Grading (ASAG) using CNN

## Overview

This project implements an **Automatic Short Answer Grading (ASAG)** system using a **Text Convolutional Neural Network (TextCNN)** enhanced with semantic similarity features. It takes a student's short answer and a question, compares the answer against a reference answer using multiple NLP techniques, and predicts a normalised grade (0.0–1.0).

The system consists of:
- **Backend**: FastAPI REST API that combines a TextCNN model with WordNet synonym matching, DistilBERT semantic similarity, content correctness checking, and grammar tolerance scoring
- **Frontend**: React web interface with 20 sample questions spanning Computer Science, Biology, and Science domains
- **Training Pipeline**: TextCNN training with 5-epoch cross-validation and 80/20 split

**Dataset**: [ASAG2024](https://huggingface.co/datasets/Meyerger/ASAG2024) — 15,190 graded short answers across 284 unique questions from university-level exams.

---

## Architecture

```
┌──────────────────────────────┐
│   React Frontend (:3000)     │   User selects question,
│   20 sample questions        │   types answer, submits
│   Score + confidence display │──────────────────┐
│   Reference answer display   │                  │
│   Score distribution chart   │                  │
└──────────────────────────────┘                  │
         ▲                                        ▼
         │                              POST /predict
         │                    ┌──────────────────────────────┐
         │  JSON response     │   FastAPI Backend (:8000)    │
         │  {score,           │                              │
         │   confidence,      │  ┌────────────────────────┐  │
         │   feedback,        │  │ Relevance Guard        │  │
         │   reference}       │  │ 3 signals: question,   │  │
         │                    │  │ reference, synonym     │  │
         └────────────────────│  └──────────┬─────────────┘  │
                              │             │ on-topic?      │
                              │             ▼                │
                              │  ┌────────────────────────┐  │
                              │  │ TextCNN Model          │  │
                              │  │ 512 conv + 1 sim +     │  │
                              │  │ 27 handcrafted feats   │  │
                              │  └──────────┬─────────────┘  │
                              │             │                │
                              │             ▼                │
                              │  ┌────────────────────────┐  │
                              │  │ Semantic Similarity     │  │
                              │  │ • WordNet synonyms      │  │
                              │  │ • DistilBERT cosine     │  │
                              │  │ • TF-IDF + n-grams      │  │
                              │  └──────────┬─────────────┘  │
                              │             │                │
                              │             ▼                │
                              │  ┌────────────────────────┐  │
                              │  │ Content & Grammar      │  │
                              │  │ • Keyword checking     │  │
                              │  │ • Word-order analysis   │  │
                              │  │ • Grammar tolerance     │  │
                              │  └──────────┬─────────────┘  │
                              │             │                │
                              │             ▼                │
                              │  ┌────────────────────────┐  │
                              │  │ Adaptive Score Blend   │  │
                              │  │ model(35%) + sim(65%)  │  │
                              │  │ → round to 0, 0.5, 1  │  │
                              │  └────────────────────────┘  │
                              └──────────────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────────────┐
                              │   ASAG2024 Dataset           │
                              │   15,190 answers             │
                              │   284 questions              │
                              │   Reference answer lookup    │
                              └──────────────────────────────┘
```

---

## Features

### Scoring Engine (Backend)

- **TextCNN model** — 4-kernel CNN (sizes 2–5, 128 filters each) trained on ASAG2024 with 300-dim embeddings
- **27 handcrafted features** — Lexical, syntactic, and semantic features extracted per answer (length, n-gram overlap, POS ratios, keyword density, etc.)
- **WordNet synonym matching** — Expands vocabulary using synonyms, hypernyms, and hyponyms so answers with alternative phrasing are matched correctly
- **DistilBERT semantic similarity** — Sentence-level cosine similarity via mean-pooled DistilBERT embeddings to capture meaning beyond exact words
- **TF-IDF similarity** — Weighted term overlap using a vectorizer fit on the full corpus
- **N-gram overlap** — Unigram, bigram, and trigram overlap with reference answers
- **Content correctness checking** — Verifies key concepts from the reference appear in the student answer, with WordNet synonym tolerance
- **Word-order analysis** — Detects scrambled-but-correct answers (content present, structure broken) and applies content override scoring
- **Grammar tolerance** — POS-based grammar assessment that avoids harsh penalties for ESL students or non-native speakers
- **Relevance gate** — 3-signal relevance check (question overlap, reference overlap, synonym-expanded overlap) to reject off-topic answers before expensive computation
- **Adaptive score blending** — Model prediction (35%) + similarity features (65%) with adaptive weights based on agreement between signals
- **Reference answer display** — Returns the best reference answer alongside the score for student learning
- **Modern FastAPI patterns** — Lifespan context manager for startup, async-safe initialisation, CORS middleware

### Frontend (React)

- **20 pre-loaded sample questions** spanning neural networks, DHCP, photosynthesis, TCP/IP, DNA, machine learning, cybersecurity, and more
- **Interactive grading** — Type an answer, submit, and receive a score, confidence, AI feedback, and reference answer in real time
- **Score visualisation** — Confidence circle, score progress bar, and score distribution chart (Recharts)
- **Model insights panel** — Displays key metrics (MSE, QWK, Accuracy, F1, sample counts)
- **Character counter** — Live character count with 1,000-character limit and colour-coded bar
- **Error handling** — Clear error banners when the backend is unreachable or returns an error

### Training Pipeline

- **TextCNN pipeline** (`asag_cnn_pipeline.py`) — 5-epoch training with vocab building and 80/20 split
- **Hyperparameter tuning** (`hyperparameter_tuning.py`) — Optuna TPE search for CNN configurations
- **Score calibration** (`score_calibration.py`) — Platt, isotonic, and temperature scaling with Nelder-Mead-optimised rounding thresholds

### Evaluation & Analysis

- **Error analysis** — Length-category breakdown (MSE/QWK/Accuracy/F1), reasoning quality (BLEU, ROUGE-L, BERTScore)
- **A/B testing framework** — Paired t-tests + bootstrap confidence intervals for model comparison
- **Cross-question testing** — Held-out question evaluation to measure generalisation to unseen questions
- **44 integration tests** — Covering all backend functions (synonym overlap, semantic similarity, relevance scoring, enhanced similarity, reference similarity) and all v3.0+ modules

---

## Model Performance

### TextCNN Results (Random 80/20 Split)

| Metric | Value |
|--------|-------|
| **QWK** | 0.261 |
| **Accuracy** | 49.6% |
| **MSE** | 0.140 |
| **F1 (weighted)** | 0.493 |
| **Test Samples** | 3,038 |
| **Training Time** | ~5 min |

---

## Dataset

- **Source**: [ASAG2024 on HuggingFace](https://huggingface.co/datasets/Meyerger/ASAG2024)
- **Rows**: 15,190 graded short answers
- **Questions**: 284 unique questions
- **Source Domains**: Computer Science, Biology, Science
- **Score Range**: 0.0 – 1.0 (normalised)
- **Test Split**: 20% (3,038 samples)

---

## Project Structure

```
cnn_project/
│
├── backend.py                   # FastAPI server — TextCNN + semantic scoring + content checks
├── asag_cnn_pipeline.py         # Original TextCNN training pipeline (5 epochs, 80/20 split)
├── asag_data.py                 # Dataset loading, vocab building, TF-IDF
│
├── preprocessing.py             # Advanced text preprocessing (lemmatization, stemming, POS)
├── distilbert_pipeline.py       # DistilBERT 5-fold cross-validation training
├── ensemble_model.py            # Multi-model ensemble with calibrated confidence
├── question_models.py           # Per-question Ridge regression models
├── reference_answers.py         # Reference answer database + BERTScore
├── scoring_improvements.py      # Calibrated scoring blend with adaptive weights
├── content_correctness.py       # Keyword checking, word-order analysis, content override
├── grammar_detection.py         # POS-based grammar tolerance scoring
│
├── feature_engineering.py       # 27 handcrafted features (lexical, syntactic, semantic)
├── training_utils.py            # CombinedLoss, CosineScheduler, training helpers
├── score_calibration.py         # Platt/Isotonic/Temperature scaling + ScoreRounder
├── hyperparameter_tuning.py     # Optuna TPE hyperparameter search
├── error_analysis.py            # Error analysis + A/B testing framework
│
├── test_model.py                # Standard model evaluation (random 80/20 split)
├── cross_question_test.py       # Cross-question generalisation evaluation
├── save_test_set.py             # Export test_set.csv from dataset
│
├── asag2024_all.csv             # Full ASAG2024 dataset (15,190 rows)
├── textcnn_model.pth            # Trained TextCNN model weights
├── test_set.csv                 # Exported test set (3,038 rows)
│
├── tests/
│   └── test_modules.py          # 44 integration tests
│
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main React component (20 questions, grading UI)
│   │   ├── App.css              # Styles
│   │   └── index.js             # Entry point
│   ├── public/index.html        # HTML shell
│   └── package.json             # Node dependencies
│
└── README.md
```

---

## Installation

### Prerequisites

- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Node.js 16+** — [nodejs.org](https://nodejs.org/) (includes npm)

### Step 1: Set up Python virtual environment

```bash
cd c:\Users\user\Documents\cnn_project
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Python backend dependencies

```bash
# Core ML & data processing
pip install torch scikit-learn pandas numpy

# FastAPI backend server
pip install fastapi "uvicorn[standard]" pydantic

# NLP & transformer models
pip install nltk transformers

# Optional: for calibration, tuning, and evaluation
pip install bert_score optuna scipy tqdm openpyxl
```

**Full list of Python packages:**
```
torch                 # PyTorch deep learning framework
scikit-learn          # ML utilities (train_test_split, metrics, etc.)
pandas                # Data manipulation (CSV loading, DataFrame operations)
numpy                 # Numerical computing
fastapi               # Web framework for REST API
uvicorn[standard]     # ASGI server to run FastAPI
pydantic              # Data validation (API request/response models)
nltk                  # WordNet synonyms, lemmatization, POS tagging
transformers          # HuggingFace DistilBERT for semantic similarity
bert_score            # Semantic evaluation using BERT (optional)
optuna                # Hyperparameter optimisation (optional)
scipy                 # Score rounding optimisation (optional)
tqdm                  # Progress bars (optional)
```

### Step 3: Install Node.js dependencies (Frontend)

```bash
cd c:\Users\user\Documents\cnn_project\frontend
npm install
```

**Frontend packages:**
- `react@18.2.0` — UI library
- `react-dom@18.2.0` — React DOM rendering
- `axios@1.13.5` — HTTP client for API calls
- `recharts@2.10.4` — Charting library
- `react-scripts@5.0.1` — Build tooling

### Step 4: Verify required data files

| File | Approx. Size | Purpose |
|------|--------------:|---------|
| `asag2024_all.csv` | ~2.1 MB | Full ASAG2024 dataset (15,190 rows) |
| `textcnn_model.pth` | ~55 MB | Pre-trained TextCNN model weights |

These files are **not** included in the repository. To obtain them:

- **`asag2024_all.csv`**: Download from [HuggingFace](https://huggingface.co/datasets/Meyerger/ASAG2024) or export using:
  ```python
  from datasets import load_dataset
  ds = load_dataset('Meyerger/ASAG2024')
  df = ds['train'].to_pandas()
  df.to_csv('asag2024_all.csv', index=False)
  ```

- **`textcnn_model.pth`**: Train locally using `asag_cnn_pipeline.py` (see training section below).

### Step 5 (Optional): Export test set

```bash
.venv\Scripts\python save_test_set.py
```

Creates `test_set.csv` with 3,038 test samples (20% of dataset).

---

## Quick Start

After completing installation, run in separate terminals:

**Terminal 1 — Backend API:**
```bash
cd c:\Users\user\Documents\cnn_project
.venv\Scripts\activate
.venv\Scripts\uvicorn backend:app --reload --port 8000
```
Visit **http://localhost:8000/docs** for interactive API docs.

**Terminal 2 — Frontend:**
```bash
cd c:\Users\user\Documents\cnn_project\frontend
node_modules\.bin\react-scripts start
```
Opens **http://localhost:3000** automatically.

**Terminal 3 (optional) — Evaluation:**
```bash
cd c:\Users\user\Documents\cnn_project
.venv\Scripts\activate
.venv\Scripts\python test_model.py                 # Random 80/20 split
.venv\Scripts\python cross_question_test.py         # Held-out question split
```

---

## Training

### Train the TextCNN Model (~5 min)

```bash
.venv\Scripts\python asag_cnn_pipeline.py
```

Trains for 5 epochs, saves weights to `textcnn_model.pth` and writes `test_set.csv`.

### Hyperparameter Tuning (Optional)

```bash
# Tune the CNN pipeline (30 trials ≈ 20 min)
.venv\Scripts\python hyperparameter_tuning.py --target cnn --trials 30

# Resume a study using a persistent SQLite database
.venv\Scripts\python hyperparameter_tuning.py --target cnn --trials 50 --storage optuna_cnn.db
```

---

## API Reference

### `POST /predict`

**Request:**
```json
{
  "question": "Explain backpropagation in neural networks.",
  "answer": "It uses gradient descent to propagate errors backward through the layers."
}
```

**Response:**
```json
{
  "score": 0.5,
  "confidence": 0.87,
  "feedback": "Your answer shows some understanding but is missing key details.",
  "reference_answer": "Backpropagation is a supervised learning algorithm that computes the gradient of the loss function with respect to each weight by propagating errors backward through the network layers using the chain rule."
}
```

### `GET /`

Health check — returns `{"status": "ASAG backend running", "endpoint": "POST /predict"}`.

---

## Scoring System

### Score Scale

| Score | Meaning | Interpretation |
|-------|---------|----------------|
| **0.0** | Incorrect / Off-topic | Answer does not address the question or is completely wrong |
| **0.5** | Partially Correct | Answer shows some understanding but is incomplete or lacks key details |
| **1.0** | Excellent / Complete | Answer demonstrates full understanding with all key concepts covered |

### How Scoring Works

The backend uses a **calibrated hybrid approach** combining multiple scoring signals:

1. **TextCNN Model Prediction** — TextCNN trained on 15,190 graded answers with 4 convolutional kernels (sizes 2–5), 128 filters each, plus a similarity score and 27 handcrafted features

2. **Semantic Similarity Features** (8 features computed per answer):
   - **Unigram overlap** (weight: 10%) — Direct word overlap with reference
   - **Bigram overlap** (weight: 15%) — Phrase-level 2-gram matching
   - **Trigram overlap** (weight: 10%) — Longer phrase matching
   - **TF-IDF similarity** (weight: 15%) — Weighted term overlap using corpus-wide IDF scores
   - **Length ratio** (weight: 5%) — Penalizes extremely short answers
   - **WordNet synonym overlap** (weight: 20%) — Matches words via synonyms, hypernyms, and hyponyms (e.g., "produces" ≈ "generates", "energy" ≈ "power")
   - **DistilBERT semantic similarity** (weight: 25%) — Sentence-level meaning comparison using DistilBERT mean-pooled embeddings + cosine similarity

3. **Content Correctness & Grammar Tolerance**:
   - Checks whether key concepts from the reference appear in the student answer (using WordNet synonyms)
   - Detects scrambled-but-correct answers (all words present, order wrong) and applies a content override boost
   - Assesses grammar quality via POS tagging and applies tolerance for minor issues (fair for ESL students)

4. **Adaptive Score Blending**:
   - Model prediction weight: **35%**, Similarity features weight: **65%**
   - Weights adapt based on agreement between signals
   - High agreement → trust model more; disagreement → trust similarity features more
   - Content override boost applied when content is correct despite poor structure

### Relevance Gate

Before scoring, answers are checked for **topical relevance** using three signals:

1. **Question overlap** — Jaccard similarity between question and answer tokens
2. **Reference overlap** — Jaccard similarity between reference and answer tokens
3. **Synonym-expanded overlap** — WordNet-expanded overlap with the reference answer

The maximum of these three signals is compared against a threshold (0.04). If below threshold, the answer is scored 0.0 with an "off-topic" feedback. This three-signal approach prevents synonym-rich but valid answers from being incorrectly flagged as off-topic.

### Confidence Score

| Range | Interpretation |
|-------|---------------|
| 0.80–0.99 | Model and similarity features strongly agree |
| 0.60–0.79 | Moderate agreement or clear patterns |
| 0.55–0.59 | Uncertainty — conflicting signals |

Off-topic answers return confidence = **0.97** (very confident they're wrong).

### Example Scoring

**Question:** "What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?"

| Student Answer | Score | Feedback |
|----------------|-------|----------|
| Full DHCP explanation with uses, IP assignment, and related protocols | **1.0** | "Well done! Your answer covers the main points accurately and concisely." |
| "DHCP is a protocol that assigns IP addresses automatically" | **0.5** | "Good start, but your answer could be more precise." |
| "AI, or Artificial Intelligence, is a field of computer science." | **0.0** | "Off-topic response detected." |

---

## Model Architecture

### TextCNN

| Layer | Details |
|-------|---------|
| Embedding | 300-dim, vocab-size × 300, padding_idx=0 |
| Conv2D ×4 | Kernel sizes [2, 3, 4, 5], 128 filters each |
| Activation | ReLU |
| Pooling | Max-over-time (per filter) |
| Dropout | p = 0.4 |
| Fully Connected | 540 → 1 (512 conv + 1 sim + 27 HC features) |
| Output | Sigmoid (regression, 0–1) |
| Loss | MSELoss |
| Optimiser | Adam (lr = 1e-3) |

### Semantic Encoder (DistilBERT)

| Component | Details |
|-----------|---------|
| Model | `distilbert-base-uncased` (loaded via `DistilBertForMaskedLM` → `.distilbert` base, avoids UNEXPECTED key warnings) |
| Purpose | Sentence-level semantic similarity between student and reference answers |
| Pooling | Mean pooling over token embeddings (excluding padding) |
| Similarity | Cosine similarity, clamped to [0, 1] |

### Classification Scheme

Continuous 0–1 grades are binned into 3 classes for classification metrics:

| Class | Grade Range | Label |
|-------|-------------|-------|
| 0 | 0.00 – 0.32 | Low |
| 1 | 0.33 – 0.66 | Medium |
| 2 | 0.67 – 1.00 | High |

---

## Module Reference

### Content Correctness (`content_correctness.py`)

Verifies key concepts from reference answers appear in student responses:

```python
from content_correctness import ContentCorrectnessChecker

checker = ContentCorrectnessChecker()

# Check keyword presence with WordNet synonyms
result = checker.check_keyword_presence(student_answer, reference_answer, threshold=0.6)
# → {'match_score': 0.8, 'content_correct': True, 'matched_keywords': [...]}

# Detect scrambled-but-correct answers
override = checker.compute_content_override_score(student_answer, reference_answer)
# → {'override_score': 0.85, 'should_boost': True, 'boost_amount': 0.35}
```

### Grammar Tolerance (`grammar_detection.py`)

POS-based grammar assessment that avoids harsh penalties for imperfect structure:

```python
from grammar_detection import GrammarDetector

detector = GrammarDetector()
assessment = detector.assess_grammar_tolerance(
    "The system it uses gradients for learning the weights in layers"
)
# → {'tolerance_score': 0.85, 'recommendation': 'be_lenient', 'violations': {...}}
```

### Feature Engineering (`feature_engineering.py`)

Extracts 27 handcrafted features per answer:

```python
from feature_engineering import extract_all_features, FEATURE_NAMES

vec = extract_all_features(student_answer, reference_answer)  # shape (27,)
print(dict(zip(FEATURE_NAMES, vec)))
```

### Score Calibration (`score_calibration.py`)

```python
from score_calibration import EnsembleCalibrator, ScoreRounder

cal = EnsembleCalibrator(methods=['platt', 'isotonic', 'temperature'])
cal.fit(val_preds, val_labels)
calibrated = cal.transform(test_preds)

rounder = ScoreRounder(num_classes=3)
rounder.fit(calibrated, val_labels)
final_scores = rounder.predict_normalised(calibrated)
```

### Error Analysis (`error_analysis.py`)

```python
from error_analysis import ABTestFramework

result = ABTestFramework.compare(
    metrics_a={"qwk": [0.30, 0.28, 0.32]},  # baseline per-fold
    metrics_b={"qwk": [0.48, 0.50, 0.48]},  # improved per-fold
    model_a_name="TextCNN",
    model_b_name="DistilBERT+HC",
)
# → prints winner with p-value and 95% CI
```

---

## Testing

Run the full integration test suite:

```bash
.venv\Scripts\python -m pytest tests/test_modules.py -v
# 44 tests covering all modules
```

**Test coverage includes:**
- `LengthAdaptiveProcessor` — short/medium/long answer handling
- `ErrorAnalyser` — metric computation and error categorisation
- `ABTestFramework` — statistical comparison between models
- `DomainPretraining` — corpus building and training configuration
- `MultiModelEnsemble` — prediction blending and method selection
- `CrossAttentionVisualization` — attention heatmap generation
- `synonym_overlap()` — WordNet exact, synonym, hypernym/hyponym, and unrelated-word matching
- `semantic_similarity()` — DistilBERT sentence similarity (identical/related/unrelated pairs)
- `relevance_score()` — On-topic vs off-topic classification with synonym-rich answers
- `enhanced_similarity()` — Full 8-feature similarity computation
- `reference_similarity()` — Weighted combination and pre-computed feature reuse

---

## Performance Summary

| Metric | TextCNN |
|--------|--------|
| QWK | 0.261 |
| Accuracy | 49.6% |
| MSE | 0.140 |
| F1-score | 0.493 |
| Training time | ~5 min |
| Off-topic detection | Yes (3-signal) |
| Synonym matching | WordNet + DistilBERT |
| Feature input | 300-dim embeddings + 1 sim + 27 HC |

---

## Troubleshooting

### "No module named 'torch'"
Ensure virtual environment is activated:
```bash
.venv\Scripts\activate
.venv\Scripts\pip install torch scikit-learn pandas fastapi uvicorn pydantic nltk transformers
```

### "Could not reach the grading server"
Backend must be running:
```bash
.venv\Scripts\uvicorn backend:app --reload --port 8000
```

### Port 3000 already in use
```bash
cd frontend
node_modules\.bin\react-scripts start --port 3001
```

### Model file too large / slow to load
The TextCNN model (~55 MB) plus DistilBERT (~260 MB) load on first request. This is normal and takes ~5–10 seconds on CPU.

### UNEXPECTED key warnings from DistilBERT
The backend loads DistilBERT via `DistilBertForMaskedLM.from_pretrained()` and extracts the `.distilbert` base transformer. This avoids warnings about `vocab_projector`/`vocab_transform`/`vocab_layer_norm` keys that exist in the MLM checkpoint but aren't needed for embeddings.

---

## License

This project is for educational and research purposes.
