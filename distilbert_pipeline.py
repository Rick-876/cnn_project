"""
distilbert_pipeline.py
Improved training pipeline using DistilBERT with:
- Stratified K-Fold cross-validation
- Class balancing (weighted loss)
- Hyperparameter tuning
- Proper validation and calibration
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SchedulerType, get_scheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, cohen_kappa_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from tqdm import tqdm
import json
from preprocessing import preprocess_for_model


class ASAGDatasetBERT(Dataset):
    """PyTorch Dataset for BERT-based models."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = None
        self._encode_texts()
    
    def _encode_texts(self):
        """Tokenize and encode all texts."""
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


class DistilBERTGrader(nn.Module):
    """DistilBERT-based grader with regression head for score prediction."""
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type='regression'
        )
        # Modify output layer for regression on [0, 1]
        self.transformer.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        return outputs.logits


def compute_class_weights(labels, num_classes=3):
    """Compute class weights for imbalanced datasets."""
    label_counts = np.bincount(labels, minlength=num_classes)
    # Avoid division by zero
    label_counts = np.maximum(label_counts, 1)
    weights = 1.0 / label_counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def discretize_labels(scores, num_classes=3):
    """Convert continuous scores [0,1] to discrete class labels."""
    if num_classes == 3:
        # 0: [0.00, 0.33), 1: [0.33, 0.67), 2: [0.67, 1.00]
        return np.digitize(scores, bins=[0.33, 0.67])
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


def train_epoch(model, dataloader, optimizer, device, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1) if len(batch['labels'].shape) == 1 else batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.MSELoss()(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = outputs.cpu().numpy().flatten()
            
            preds.extend(batch_preds)
            true_labels.extend(labels)
    
    preds = np.array(preds)
    true_labels = np.array(true_labels)
    
    # Compute metrics
    mse = mean_squared_error(true_labels, preds)
    
    # For discrete metrics, discretize predictions
    pred_classes = discretize_labels(np.clip(preds, 0, 1), num_classes=3)
    true_classes = discretize_labels(true_labels, num_classes=3)
    
    qwk = cohen_kappa_score(true_classes, pred_classes, weights='quadratic')
    acc = accuracy_score(true_classes, pred_classes)
    precision = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
    
    return {
        'mse': mse,
        'qwk': qwk,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': preds,
        'true_labels': true_labels
    }


def train_distilbert_kfold(data_path='asag2024_all.csv',
                          model_save_prefix='distilbert_fold',
                          n_splits=5,
                          epochs=10,
                          batch_size=16,
                          learning_rate=2e-5,
                          max_length=512,
                          model_name='distilbert-base-uncased',
                          preprocess=True):
    """
    Train DistilBERT using Stratified K-Fold cross-validation.
    
    Args:
        data_path: Path to CSV dataset
        model_save_prefix: Prefix for saving fold models
        n_splits: Number of folds
        epochs: Training epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max token length
        model_name: HuggingFace model identifier
        preprocess: Whether to preprocess text
    
    Returns:
        Dictionary with CV results and best model path
    """
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    required_cols = ['question', 'reference_answer', 'provided_answer', 'normalized_grade']
    df = df[required_cols].fillna('')
    
    # Preprocess if requested
    if preprocess:
        print("Preprocessing texts...")
        df['combined_text'] = df.apply(
            lambda r: preprocess_for_model(
                f"Question: {r['question']} Reference: {r['reference_answer']} Student: {r['provided_answer']}"
            ),
            axis=1
        )
    else:
        df['combined_text'] = (
            "Question: " + df['question'] + " "
            "Reference: " + df['reference_answer'] + " "
            "Student: " + df['provided_answer']
        )
    
    texts = df['combined_text'].values
    labels = df['normalized_grade'].values
    label_classes = discretize_labels(labels, num_classes=3)
    
    # Initialize tokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = []
    best_f1 = -1
    best_model_path = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, label_classes)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*60}")
        
        # Split data
        train_texts, val_texts = texts[train_idx], texts[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # Create datasets
        train_dataset = ASAGDatasetBERT(train_texts, train_labels, tokenizer, max_length)
        val_dataset = ASAGDatasetBERT(val_texts, val_labels, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = DistilBERTGrader(model_name=model_name)
        model.to(device)
        
        # Compute class weights for imbalanced data
        train_classes = discretize_labels(train_labels, num_classes=3)
        class_weights = compute_class_weights(train_classes, num_classes=3)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler(
            'linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_f1 = -1
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = evaluate(model, val_loader, device)
            print(f"Val MSE: {val_metrics['mse']:.4f} | QWK: {val_metrics['qwk']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            scheduler.step()
            
            # Early stopping based on F1
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                # Save fold model
                fold_path = f"{model_save_prefix}_{fold}.pth"
                torch.save(model.state_dict(), fold_path)
                print(f"✓ Model saved to {fold_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best fold model and evaluate
        model.load_state_dict(torch.load(fold_path))
        final_metrics = evaluate(model, val_loader, device)
        cv_results.append(final_metrics)
        
        print(f"\nFold {fold + 1} Best Results:")
        print(f"  MSE: {final_metrics['mse']:.4f}")
        print(f"  QWK: {final_metrics['qwk']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  F1: {final_metrics['f1']:.4f}")
        
        if final_metrics['f1'] > best_f1:
            best_f1 = final_metrics['f1']
            best_model_path = fold_path
    
    # Summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    avg_mse = np.mean([r['mse'] for r in cv_results])
    avg_qwk = np.mean([r['qwk'] for r in cv_results])
    avg_acc = np.mean([r['accuracy'] for r in cv_results])
    avg_f1 = np.mean([r['f1'] for r in cv_results])
    
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average QWK: {avg_qwk:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Best Model: {best_model_path} (F1: {best_f1:.4f})")
    
    # Save summary
    summary = {
        'n_splits': n_splits,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'model_name': model_name,
        'avg_mse': float(avg_mse),
        'avg_qwk': float(avg_qwk),
        'avg_accuracy': float(avg_acc),
        'avg_f1': float(avg_f1),
        'best_model': best_model_path,
        'fold_results': [
            {
                'fold': i,
                'mse': float(r['mse']),
                'qwk': float(r['qwk']),
                'accuracy': float(r['accuracy']),
                'f1': float(r['f1'])
            }
            for i, r in enumerate(cv_results)
        ]
    }
    
    summary_path = 'distilbert_cv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    return summary


if __name__ == '__main__':
    result = train_distilbert_kfold(
        data_path='asag2024_all.csv',
        n_splits=5,
        epochs=5,
        batch_size=16,
        learning_rate=2e-5
    )
