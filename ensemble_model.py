"""
ensemble_model.py
Ensemble methods for combining multiple model predictions.
Improves robustness and generalization by leveraging diverse architectures.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Union
from dataclasses import dataclass
import json


@dataclass
class EnsembleWeights:
    """Container for ensemble member weights."""
    weights: Dict[str, float]
    
    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            weights = {}
        # Normalize weights to sum to 1
        total = sum(weights.values()) if weights else 1.0
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = weights
    
    def to_dict(self):
        return self.weights
    
    @staticmethod
    def from_dict(d):
        return EnsembleWeights(d)


class EnsemblePredictor:
    """Ensemble model for combining predictions from multiple sources."""
    
    def __init__(self, weights: Dict[str, float] = None, method: str = 'weighted_average'):
        """
        Initialize ensemble predictor.
        
        Args:
            weights: Dictionary mapping model names to their weights
            method: 'weighted_average', 'median', 'max_confidence', or 'voting'
        """
        self.weights = EnsembleWeights(weights or {})
        self.method = method
        self.calibration_params = None
    
    def add_model(self, name: str, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.weights.weights[name] = weight
        # Renormalize
        total = sum(self.weights.weights.values())
        if total > 0:
            self.weights.weights = {k: v / total for k, v in self.weights.weights.items()}
    
    def predict(self, predictions: Dict[str, Union[float, np.ndarray]], 
                confidences: Dict[str, float] = None) -> Dict:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: Dict mapping model names to predictions (scalar or array)
            confidences: Optional dict of confidence scores for each model
        
        Returns:
            Dict with 'score', 'confidence', and 'method' keys
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        if self.method == 'weighted_average':
            return self._weighted_average(predictions, confidences)
        elif self.method == 'median':
            return self._median(predictions, confidences)
        elif self.method == 'max_confidence':
            return self._max_confidence(predictions, confidences)
        elif self.method == 'voting':
            return self._voting(predictions, confidences)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def _weighted_average(self, predictions: Dict[str, float], 
                         confidences: Dict[str, float] = None) -> Dict:
        """Weighted average of predictions."""
        preds = []
        weights = []
        
        for model_name, pred in predictions.items():
            weight = self.weights.weights.get(model_name, 1.0)
            
            # Optionally boost weight if confidence is high
            if confidences and model_name in confidences:
                weight *= (0.5 + 0.5 * confidences[model_name])
            
            preds.append(float(pred))
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_pred = np.average(preds, weights=weights)
        ensemble_confidence = np.mean([confidences.get(m, 0.5) for m in predictions.keys()])
        
        return {
            'score': float(np.clip(weighted_pred, 0, 1)),
            'confidence': float(ensemble_confidence),
            'method': 'weighted_average',
            'component_preds': predictions,
            'component_weights': {k: float(w) for k, w in zip(predictions.keys(), weights)}
        }
    
    def _median(self, predictions: Dict[str, float],
                confidences: Dict[str, float] = None) -> Dict:
        """Median of predictions (robust to outliers)."""
        preds = [float(p) for p in predictions.values()]
        median_pred = np.median(preds)
        ensemble_confidence = np.mean([confidences.get(m, 0.5) for m in predictions.keys()])
        
        return {
            'score': float(np.clip(median_pred, 0, 1)),
            'confidence': float(ensemble_confidence),
            'method': 'median',
            'component_preds': predictions
        }
    
    def _max_confidence(self, predictions: Dict[str, float],
                       confidences: Dict[str, float] = None) -> Dict:
        """Select prediction with highest confidence."""
        if not confidences:
            # Without confidences, use equal weighting
            return self._weighted_average(predictions, confidences)
        
        best_model = max(confidences.items(), key=lambda x: x[1])
        best_model_name = best_model[0]
        best_confidence = best_model[1]
        
        return {
            'score': float(predictions[best_model_name]),
            'confidence': float(best_confidence),
            'method': 'max_confidence',
            'selected_model': best_model_name,
            'component_preds': predictions
        }
    
    def _voting(self, predictions: Dict[str, float],
                confidences: Dict[str, float] = None) -> Dict:
        """Discretize predictions to classes and use voting."""
        # Discretize to 3 classes: 0 (0-0.33), 1 (0.33-0.67), 2 (0.67-1)
        classes = []
        for pred in predictions.values():
            if pred < 0.33:
                classes.append(0)
            elif pred < 0.67:
                classes.append(1)
            else:
                classes.append(2)
        
        # Count votes
        votes = np.bincount(classes, minlength=3)
        winning_class = np.argmax(votes)
        
        # Convert winning class back to score range
        class_to_score = {0: 0.15, 1: 0.5, 2: 0.85}
        ensemble_pred = class_to_score[winning_class]
        ensemble_confidence = np.max(votes) / len(classes)
        
        return {
            'score': float(ensemble_pred),
            'confidence': float(ensemble_confidence),
            'method': 'voting',
            'winning_class': int(winning_class),
            'votes': {str(i): int(v) for i, v in enumerate(votes)},
            'component_preds': predictions
        }
    
    def to_dict(self):
        """Serialize ensemble configuration."""
        return {
            'method': self.method,
            'weights': self.weights.to_dict(),
            'calibration_params': self.calibration_params
        }
    
    @staticmethod
    def from_dict(d):
        """Deserialize ensemble configuration."""
        ensemble = EnsemblePredictor(
            weights=d.get('weights', {}),
            method=d.get('method', 'weighted_average')
        )
        ensemble.calibration_params = d.get('calibration_params')
        return ensemble


class CalibrationCurve:
    """Isotonic calibration for confidence scores."""
    
    def __init__(self):
        self.calibration_scale = 1.0
        self.calibration_shift = 0.0
    
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """Fit calibration parameters using Platt scaling."""
        # Simple Platt scaling: calibrated = 1 / (1 + exp(A * pred + B))
        # For simplicity, using linear scaling
        errors = np.abs(predictions - ground_truth)
        mean_error = np.mean(errors)
        
        # Scale confidence inversely to error magnitude
        if mean_error > 0:
            self.calibration_scale = 1.0 / (1.0 + mean_error)
        else:
            self.calibration_scale = 1.0
    
    def calibrate(self, confidence: float) -> float:
        """Apply calibration to a confidence score."""
        # Calibrated confidence = 1 - (1 - original_confidence) * scale
        # When scale = 1, confidence is unchanged
        # When scale > 1, confidence is reduced (more conservative)
        calibrated = 1.0 - (1.0 - confidence) * self.calibration_scale
        return float(np.clip(calibrated, 0, 0.99))


def blend_predictions(pred_dict: Dict[str, float], 
                     blend_weights: Dict[str, float] = None) -> float:
    """
    Simple weighted blending of predictions.
    
    Args:
        pred_dict: Dictionary mapping source names to predictions
        blend_weights: Dictionary mapping source names to blend weights
    
    Returns:
        Blended prediction score
    """
    if blend_weights is None:
        blend_weights = {k: 1.0 for k in pred_dict.keys()}
    
    total_weight = sum(blend_weights.get(k, 1.0) for k in pred_dict.keys())
    if total_weight == 0:
        return np.mean(list(pred_dict.values()))
    
    blended = sum(
        pred_dict[k] * blend_weights.get(k, 1.0)
        for k in pred_dict.keys()
    ) / total_weight
    
    return float(np.clip(blended, 0, 1))


def aggregate_metrics(metric_dicts: List[Dict]) -> Dict:
    """Aggregate metrics from multiple models."""
    if not metric_dicts:
        return {}
    
    aggregated = {}
    for key in metric_dicts[0].keys():
        if key in ['preds', 'true_labels', 'confusion_matrix']:
            continue
        
        values = [m[key] for m in metric_dicts if key in m]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
    
    return aggregated
