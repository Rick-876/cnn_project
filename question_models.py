"""
question_models.py
Question-specific grading models.
Trains separate lightweight models per question for improved accuracy.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json
import pickle
from pathlib import Path
from tqdm import tqdm


class QuestionSpecificModel:
    """Lightweight model trained on a single question."""
    
    def __init__(self, question_id: str, min_samples: int = 20):
        self.question_id = question_id
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.min_samples = min_samples
        self.trained = False
    
    def extract_features(self, texts: list) -> np.ndarray:
        """Extract simple features from texts."""
        features = []
        
        for text in texts:
            tokens = text.lower().split()
            feature_vec = [
                len(tokens),  # Token count
                len(text),    # Character count
                len(set(tokens)),  # Unique tokens
                np.mean([len(t) for t in tokens]) if tokens else 0,  # Avg token length
                text.count('?'),  # Question marks
                text.count('!'),  # Exclamation marks
            ]
            features.append(feature_vec)
        
        self.feature_names = ['token_count', 'char_count', 'unique_tokens', 
                             'avg_token_len', 'questions', 'exclamations']
        return np.array(features)
    
    def fit(self, texts: list, labels: np.ndarray):
        """Train model for this question."""
        if len(texts) < self.min_samples:
            print(f"Question {self.question_id}: Insufficient samples ({len(texts)} < {self.min_samples}), skipping")
            return False
        
        try:
            X = self.extract_features(texts)
            X_scaled = self.scaler.fit_transform(X)
            
            # Use Ridge regression for simplicity
            self.model = Ridge(alpha=1.0)
            self.model.fit(X_scaled, labels)
            
            self.trained = True
            return True
        except Exception as e:
            print(f"Question {self.question_id}: Training failed - {e}")
            return False
    
    def predict(self, text: str) -> float:
        """Predict score for a text."""
        if not self.trained or self.model is None:
            return 0.5  # Default to neutral score
        
        try:
            X = self.extract_features([text])
            X_scaled = self.scaler.transform(X)
            score = float(self.model.predict(X_scaled)[0])
            return float(np.clip(score, 0, 1))
        except:
            return 0.5
    
    def get_metrics(self, texts: list, labels: np.ndarray) -> dict:
        """Compute metrics for this question."""
        if not self.trained:
            return {}
        
        try:
            X = self.extract_features(texts)
            X_scaled = self.scaler.transform(X)
            preds = self.model.predict(X_scaled)
            mse = mean_squared_error(labels, preds)
            
            return {
                'mse': float(mse),
                'samples': len(texts),
                'trained': True
            }
        except:
            return {}
    
    def save(self, path: str):
        """Save model to disk."""
        if not self.trained:
            return False
        
        state = {
            'question_id': self.question_id,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained': self.trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        return True
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.question_id = state['question_id']
            self.model = state['model']
            self.scaler = state['scaler']
            self.feature_names = state['feature_names']
            self.trained = state['trained']
            return True
        except:
            return False


class QuestionSpecificEnsemble:
    """Ensemble of question-specific models."""
    
    def __init__(self, save_dir: str = 'question_models'):
        self.models = {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.question_to_id = {}
        self.id_to_question = {}
    
    def fit(self, df: pd.DataFrame, min_samples: int = 20):
        """
        Train models for each question in the dataset.
        
        Args:
            df: DataFrame with columns ['question', 'provided_answer', 'normalized_grade']
            min_samples: Minimum samples required to train a model
        """
        # Group by question
        grouped = df.groupby('question')
        
        print(f"Training question-specific models for {len(grouped)} questions...")
        successful = 0
        
        for question_text, group_df in tqdm(grouped, desc='Training models'):
            question_id = str(len(self.question_to_id))
            self.question_to_id[question_text] = question_id
            self.id_to_question[question_id] = question_text
            
            texts = group_df['provided_answer'].values
            labels = group_df['normalized_grade'].values
            
            model = QuestionSpecificModel(question_id, min_samples=min_samples)
            if model.fit(texts, labels):
                self.models[question_id] = model
                successful += 1
        
        print(f"Successfully trained {successful}/{len(grouped)} question models")
        self._save_mappings()
    
    def predict(self, question: str, answer: str) -> float:
        """Predict score using question-specific model if available."""
        question_id = self.question_to_id.get(question)
        
        if question_id and question_id in self.models:
            return self.models[question_id].predict(answer)
        else:
            return 0.5  # Fallback
    
    def save_all(self):
        """Save all models to disk."""
        for question_id, model in self.models.items():
            path = self.save_dir / f"question_{question_id}.pkl"
            model.save(str(path))
        
        print(f"Saved {len(self.models)} question models to {self.save_dir}")
    
    def load_all(self):
        """Load all models from disk."""
        if not self.save_dir.exists():
            print(f"Save directory {self.save_dir} not found")
            return 0
        
        pkl_files = list(self.save_dir.glob("question_*.pkl"))
        loaded = 0
        
        for pkl_file in pkl_files:
            question_id = pkl_file.stem.replace("question_", "")
            model = QuestionSpecificModel(question_id)
            
            if model.load(str(pkl_file)):
                self.models[question_id] = model
                loaded += 1
        
        self._load_mappings()
        print(f"Loaded {loaded} question models from {self.save_dir}")
        return loaded
    
    def _save_mappings(self):
        """Save question ID mappings."""
        mapping = {
            'question_to_id': self.question_to_id,
            'id_to_question': self.id_to_question
        }
        with open(self.save_dir / 'mappings.json', 'w') as f:
            json.dump(mapping, f)
    
    def _load_mappings(self):
        """Load question ID mappings."""
        try:
            with open(self.save_dir / 'mappings.json', 'r') as f:
                mapping = json.load(f)
            self.question_to_id = mapping['question_to_id']
            self.id_to_question = mapping['id_to_question']
        except:
            pass
    
    def get_coverage(self) -> dict:
        """Get statistics on model coverage."""
        total_questions = len(self.question_to_id)
        trained_questions = len(self.models)
        
        return {
            'total_questions': total_questions,
            'trained_questions': trained_questions,
            'coverage_pct': (trained_questions / total_questions * 100) if total_questions > 0 else 0,
            'fallback_required_pct': 100 - (trained_questions / total_questions * 100) if total_questions > 0 else 100
        }


def train_question_models(data_path: str = 'asag2024_all.csv',
                         min_samples: int = 20,
                         save_dir: str = 'question_models'):
    """
    Train question-specific models.
    
    Args:
        data_path: Path to dataset CSV
        min_samples: Minimum samples required per question
        save_dir: Directory to save models
    """
    df = pd.read_csv(data_path)
    
    ensemble = QuestionSpecificEnsemble(save_dir=save_dir)
    ensemble.fit(df, min_samples=min_samples)
    ensemble.save_all()
    
    stats = ensemble.get_coverage()
    print(f"\nQuestion Model Coverage:")
    print(f"  Total Questions: {stats['total_questions']}")
    print(f"  Trained Models: {stats['trained_questions']}")
    print(f"  Coverage: {stats['coverage_pct']:.1f}%")
    
    return ensemble


if __name__ == '__main__':
    ensemble = train_question_models()
