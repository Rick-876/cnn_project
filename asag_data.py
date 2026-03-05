# --- Imports and Initialization ---
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, cohen_kappa_score

# --- Configuration ---
EMBED_DIM = 300

# --- Load ASAG2024 dataset from local parquet file ---
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets--Meyerger--ASAG2024", "snapshots", "9a7a179de4e227f5e78625bc9ded2d8761f1ff09")
train_df = pd.read_parquet(os.path.join(DATASET_DIR, "train.parquet"))
train_df = train_df[["question", "reference_answer", "provided_answer", "normalized_grade"]]
# Fill missing text values
train_df = train_df.fillna({"question": "", "reference_answer": "", "provided_answer": "", "normalized_grade": 0.0})
train_df["text"] = (
    "Question: " + train_df["question"] + " "
    "Reference: " + train_df["reference_answer"] + " "
    "Student: " + train_df["provided_answer"]
)
train_df["normalized_grade"] = train_df["normalized_grade"].fillna(0.0)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_df["text"].tolist(),
    train_df["normalized_grade"].tolist(),
    test_size=0.2,
    random_state=42
)

# --- Tokenization, Vocabulary, Encoding ---
def tokenize(text):
    return re.findall(r"\w+", text.lower())

all_texts = train_texts + test_texts
words = [word for text in all_texts for word in tokenize(text)]
vocab = {word: i+1 for i, (word, _) in enumerate(Counter(words).most_common())}  # 0 for padding
vocab_size = len(vocab) + 1
max_len = 100

# --- Build Random Embedding Matrix ---
print("Building embedding matrix (random initialization)...")
embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, EMBED_DIM))
print(f"Embedding matrix built: {embedding_matrix.shape}")

def encode_text(text):
    tokens = tokenize(text)
    ids = [vocab.get(token, 0) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

train_inputs = [encode_text(text) for text in train_texts]
test_inputs = [encode_text(text) for text in test_texts]

# --- PyTorch Dataset and DataLoader ---
class ASAGDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = ASAGDataset(train_inputs, train_labels)
test_dataset = ASAGDataset(test_inputs, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=64)

# --- TextCNN Model Definition with Pretrained Embeddings ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=128, filter_sizes=[2,3,4,5], output_dim=1, dropout=0.3, pretrained_weights=None):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_weights, dtype=torch.float32))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_x]
        cat = torch.cat(pooled_x, 1)
        cat = self.dropout(cat)
        out = self.fc(cat)
        return out.squeeze(1)

# --- Model Initialization, Loss, Optimizer (Tuned Hyperparameters) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_filters=128,
    filter_sizes=[2, 3, 4, 5],
    dropout=0.3,
    pretrained_weights=embedding_matrix
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# --- Model Save/Load Path ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "textcnn_model.pth")

# --- Training Loop (skip if saved model exists) ---
if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully. Skipping training.")
else:
    epochs = 15  # Increased epochs for better convergence
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)  # Adjust learning rate
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved (loss: {best_loss:.4f})")
    print(f"Training complete. Best model saved to {MODEL_PATH}")

# --- Evaluation and Metrics ---
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds.extend(outputs.cpu().numpy())
        trues.extend(batch_y.numpy())

mse = mean_squared_error(trues, preds)
print("Test MSE:", mse)

pred_classes = np.round(np.array(preds) * 3).astype(int)
true_classes = np.round(np.array(trues) * 3).astype(int)
qwk = cohen_kappa_score(true_classes, pred_classes, weights="quadratic")
print("Quadratic Weighted Kappa:", qwk)
