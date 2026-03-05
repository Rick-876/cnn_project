# Automatic Short Answer Grading System (ASAG) Using CNN

## Overview

This project implements a complete end-to-end pipeline for **Automatic Short Answer Grading (ASAG)** using the [ASAG2024 dataset](https://huggingface.co/datasets/Meyerger/ASAG2024). It trains a **TextCNN** model to predict normalised grades for student answers and exposes a **FastAPI backend** consumed by a **React web frontend** — enabling live, interactive grading through a browser.

---

## Architecture

```
asag2024_all.csv
      │
      ▼
asag_cnn_pipeline.py   ←─── Train TextCNN model
      │ saves
      ▼
textcnn_model.pth
      │ loaded by
      ├──► test_model.py            (random 80/20 split evaluation)
      ├──► cross_question_test.py   (cross-question generalisation test)
      └──► backend.py               (FastAPI REST API — POST /predict)
                │
                ▼
        frontend/  (React SPA)
        http://localhost:3000
```

---

## Features

- **TextCNN Model** — 4 parallel conv filters (sizes 2, 3, 4, 5), 128 feature maps, 300-dim embeddings, max-over-time pooling
- **80/20 Train/Test Split** — reproducible with `random_state=42`
- **Persistent Test Set** — `test_set.csv` saved for offline analysis
- **Standard Evaluation** — MSE, QWK, Accuracy, Precision, Recall, F1, Confusion Matrix
- **Cross-Question Testing** — held-out questions unseen during training; per-question F1 breakdown
- **FastAPI Backend** — CORS-enabled REST API serving live predictions at `http://localhost:8000/predict`
- **React Frontend** — academic-style single-page grading interface with score visualisation and model insights

---

## Model Performance

### Random 80/20 Split (`test_model.py`)

| Metric | Value |
|--------|-------|
| Test MSE | 0.1403 |
| Quadratic Weighted Kappa (QWK) | 0.2607 |
| Accuracy | 49.6% |
| F1-score (weighted) | 0.493 |
| Test samples | 3,038 |

### Cross-Question Split (`cross_question_test.py`)

| Metric | Value |
|--------|-------|
| Test MSE | 0.1267 |
| QWK | 0.2916 |
| Accuracy | 50.7% |
| F1-score (weighted) | 0.504 |
| Held-out questions | 57 of 284 |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Meyerger/ASAG2024](https://huggingface.co/datasets/Meyerger/ASAG2024) |
| Local file | `asag2024_all.csv` |
| Total rows | 15,190 |
| Unique questions | 284 |
| Average answers/question | ~54 |
| Grade format | Normalised float 0.0 – 1.0 |

---

## Project Structure

```
cnn_project/
├── asag_cnn_pipeline.py       # Full training pipeline (load → preprocess → train → save)
├── asag_data.py               # Alternative data loading utilities
├── test_model.py              # Evaluate saved model — random split, full metrics
├── cross_question_test.py     # Cross-question generalisation evaluation
├── save_test_set.py           # Export persistent test_set.csv
├── backend.py                 # FastAPI grading server (POST /predict)
├── asag2024_all.csv           # Full dataset
├── test_set.csv               # Saved 20% test split (3,038 rows)
├── textcnn_model.pth          # Trained model weights
├── view_questions_answers.py  # Browse dataset samples, export to Excel
├── subject_frequency.py       # Subject distribution analysis
├── count_questions.py         # Count unique questions
├── frontend/                  # React web application
│   ├── src/App.js             # All React components
│   ├── src/App.css            # Full CSS styling
│   ├── public/index.html      # HTML shell
│   └── package.json           # Node dependencies (React, Axios, Recharts)
└── README.md
```

---

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.10+** — Download from [python.org](https://www.python.org/downloads/)
- **Node.js 16+** — Download from [nodejs.org](https://nodejs.org/) (includes npm)
- **Git** (optional, for version control)

### Step 1: Set up Python virtual environment

```bash
cd c:\Users\fsmith\Documents\cnn_project
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Python backend dependencies

```bash
# Core ML & data processing
pip install torch scikit-learn pandas numpy

# FastAPI backend server
pip install fastapi "uvicorn[standard]" pydantic

# Optional: for dataset exploration
pip install tqdm openpyxl
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
tqdm                  # Progress bars (optional, for training)
openpyxl              # Excel export (optional, for data analysis scripts)
```

### Step 3: Install Node.js dependencies (Frontend)

```bash
cd c:\Users\fsmith\Documents\cnn_project\frontend
npm install
```

**Frontend packages installed:**
- `react@18.2.0` — UI library
- `react-dom@18.2.0` — React DOM rendering
- `axios@1.13.5` — HTTP client for API calls
- `recharts@2.10.4` — Charting library
- `react-scripts@5.0.1` — Build tooling

### Step 4: Verify required data files

Ensure these files exist in the project root:

| File | Size | Purpose |
|------|------|---------|
| `asag2024_all.csv` | ~2.1 MB | Full ASAG2024 dataset (15,190 rows) |
| `textcnn_model.pth` | ~55 MB | Pre-trained model weights |

**Data files location:**
```
c:\Users\fsmith\Documents\cnn_project\
├── asag2024_all.csv         ✓ Required
└── textcnn_model.pth        ✓ Required
```

### Step 5 (Optional): Export test set

If `test_set.csv` doesn't exist, regenerate it:

```bash
.venv\Scripts\python save_test_set.py
```

This creates `test_set.csv` with 3,038 test samples (20% of dataset).

---

## Quick Start (5 minutes)

After completing installation, run all three components in separate terminals:

**Terminal 1: Backend API**
```bash
cd c:\Users\fsmith\Documents\cnn_project
.venv\Scripts\activate
.venv\Scripts\uvicorn backend:app --reload --port 8000
```
✓ If successful, visit **http://localhost:8000/docs** to see interactive API docs

**Terminal 2: Frontend**
```bash
cd c:\Users\fsmith\Documents\cnn_project\frontend
node_modules\.bin\react-scripts start
```
✓ If successful, browser opens **http://localhost:3000** automatically

**Terminal 3 (optional): Run evaluations**
```bash
cd c:\Users\fsmith\Documents\cnn_project
.venv\Scripts\activate

# Standard evaluation (random 80/20 split)
.venv\Scripts\python test_model.py

# Cross-question evaluation (held-out questions)
.venv\Scripts\python cross_question_test.py
```

---

## Troubleshooting

### "No module named 'torch'"
**Solution:** Ensure virtual environment is activated and pip install completed:
```bash
.venv\Scripts\activate
.venv\Scripts\pip install torch scikit-learn pandas fastapi uvicorn pydantic
```

### "Could not reach the grading server" (frontend error)
**Solution:** Backend must be running. In Terminal 1, start:
```bash
.venv\Scripts\uvicorn backend:app --reload --port 8000
```

### Port 3000 already in use (React error)
**Solution:** Kill the process or use a different port:
```bash
.venv\Scripts\activate
cd frontend
node_modules\.bin\react-scripts start --port 3001
```

### "npm: command not found"
**Solution:** Node.js/npm not installed or not in PATH. Download from [nodejs.org](https://nodejs.org/) and restart your terminal.

### Model file too large / slow to load
The `textcnn_model.pth` (~55 MB) loads into GPU memory on first request. This is normal and takes ~2–5 seconds.

---

### 1 — Train the model

```bash
.venv\Scripts\python asag_cnn_pipeline.py
```

Trains for 5 epochs, saves weights to `textcnn_model.pth` and writes `test_set.csv`.

### 2 — Run evaluation

```bash
# Standard random-split evaluation
.venv\Scripts\python test_model.py

# Cross-question generalisation test
.venv\Scripts\python cross_question_test.py
```

### 3 — Start the backend API

```bash
.venv\Scripts\uvicorn backend:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

### 4 — Start the frontend

```bash
cd frontend
node_modules\.bin\react-scripts start
```

Open **http://localhost:3000** in your browser.

---

## API Reference

### `POST /predict`

**Request**
```json
{
  "question": "Explain backpropagation in neural networks.",
  "answer": "It uses gradient descent to propagate errors backward through the layers."
}
```

**Response**
```json
{
  "score": 0.5,
  "confidence": 0.87,
  "feedback": "Your answer shows some understanding but is missing key details."
}
```

---

## Model Architecture

| Layer | Details |
|-------|---------|
| Embedding | 300-dim, vocab-size × 300, padding_idx=0 |
| Conv2D ×4 | Kernel sizes [2, 3, 4, 5], 128 filters each |
| Activation | ReLU |
| Pooling | Max-over-time (per filter) |
| Dropout | p = 0.5 |
| Fully Connected | 512 → 1 (regression) |
| Loss | MSELoss |
| Optimiser | Adam (lr = 1e-3) |

---

## Classification Scheme

Continuous 0–1 grades are binned into 3 classes for classification metrics:

| Class | Grade Range | Label |
|-------|-------------|-------|
| 0 | 0.00 – 0.32 | Low |
| 1 | 0.33 – 0.66 | Medium |
| 2 | 0.67 – 1.00 | High |

---

## License

This project is for educational and research purposes.
