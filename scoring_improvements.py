"""
scoring_improvements.py
Advanced scoring system with calibration, confidence tuning, and multi-source blending.
Replaces the simple backend scoring logic with a more sophisticated approach.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


class ConfidenceCalibrator:
    """Calibrate confidence scores to match actual accuracy."""
    
    def __init__(self):
        self.temperature = 1.0
        self.offset = 0.0
        self.fitted = False
    
    def fit(self, predictions: np.ndarray, 
            true_labels: np.ndarray,
            confidences: np.ndarray = None):
        """
        Fit calibration parameters using Platt scaling.
        
        Args:
            predictions: Model predictions (scores 0-1)
            true_labels: Ground truth labels (scores 0-1)
            confidences: Confidence scores from model (optional)
        """
        if confidences is None:
            # Use prediction magnitude as proxy for confidence
            confidences = np.abs(predictions - 0.5) * 2 + 0.5
        
        # Compute errors
        errors = np.abs(predictions - true_labels)
        
        # Simple calibration: adjust temperature based on error magnitude
        # Higher error → lower temperature (less confident)
        avg_error = np.mean(errors)
        
        if avg_error > 0:
            # If average error is high, be more conservative
            self.temperature = 1.0 / (1.0 + avg_error)
        
        # Offset calibration
        # If model tends to over/underpredict, adjust offset
        prediction_bias = np.mean(predictions - true_labels)
        self.offset = -prediction_bias * 0.5
        
        self.fitted = True
    
    def calibrate(self, prediction: float, confidence: float) -> float:
        """
        Apply calibration to a confidence score.
        
        Args:
            prediction: Raw model prediction
            confidence: Raw confidence score
        
        Returns:
            Calibrated confidence score (0-1)
        """
        if not self.fitted:
            return float(np.clip(confidence, 0, 0.99))
        
        # Apply temperature scaling
        calibrated = 1.0 / (1.0 + np.exp(-1.0 / self.temperature * (confidence - 0.5)))
        
        # Apply offset
        calibrated = calibrated + self.offset
        
        # Clip to valid range
        return float(np.clip(calibrated, 0.01, 0.99))


class AdaptiveScorer:
    """Adaptive scoring that adjusts blending based on model agreement."""
    
    def __init__(self, 
                 cnn_weight: float = 0.35,
                 similarity_weight: float = 0.65):
        """
        Initialize scorer.
        
        Args:
            cnn_weight: Weight for CNN model predictions
            similarity_weight: Weight for similarity features
        """
        self.cnn_weight = cnn_weight
        self.similarity_weight = similarity_weight
        self.calibrator = ConfidenceCalibrator()
    
    def compute_score(self, 
                     cnn_pred: float,
                     cnn_confidence: float,
                     similarity_score: float,
                     similarity_confidence: float,
                     relevance_score: float = 1.0) -> Dict:
        """
        Compute final score by adaptively blending CNN and similarity predictions.
        
        Args:
            cnn_pred: CNN model prediction (0-1)
            cnn_confidence: CNN model confidence (0-1)
            similarity_score: Similarity-based score (0-1)
            similarity_confidence: Similarity metric confidence (0-1)
            relevance_score: Relevance gate score (0-1), 0 means off-topic
        
        Returns:
            Dict with 'score', 'confidence', 'feedback', and component scores
        """
        
        # Apply relevance gate
        if relevance_score < 0.04:  # Off-topic threshold
            return {
                'score': 0.0,
                'confidence': 0.97,
                'feedback': 'Off-topic response detected. Your answer should address the specific concept in the question.',
                'method': 'relevance_gate',
                'relevance_score': float(relevance_score)
            }
        
        # Compute agreement between CNN and similarity
        pred_diff = abs(cnn_pred - similarity_score)
        agreement = 1.0 - np.tanh(pred_diff)  # [0, 1], higher = more agreement
        
        # Adaptive weighting: when models agree, trust them more
        if agreement > 0.7:
            # High agreement: trust averages more
            adaptive_cnn_weight = 0.45
            adaptive_sim_weight = 0.55
        elif agreement > 0.5:
            # Moderate agreement: use default weights
            adaptive_cnn_weight = self.cnn_weight
            adaptive_sim_weight = self.similarity_weight
        else:
            # Disagreement: trust similarity more (it's more diverse)
            adaptive_cnn_weight = 0.25
            adaptive_sim_weight = 0.75
        
        # Blend predictions
        blended_pred = (
            cnn_pred * adaptive_cnn_weight +
            similarity_score * adaptive_sim_weight
        )
        blended_pred = float(np.clip(blended_pred, 0, 1))
        
        # Confidence: combination of both sources
        blended_confidence = (
            cnn_confidence * adaptive_cnn_weight +
            similarity_confidence * adaptive_sim_weight
        )
        
        # Calibrate confidence
        calibrated_confidence = self.calibrator.calibrate(blended_pred, blended_confidence)
        
        # Generate feedback
        feedback = self._generate_feedback(
            blended_pred,
            agreement,
            cnn_confidence,
            similarity_confidence
        )
        
        return {
            'score': blended_pred,
            'confidence': calibrated_confidence,
            'feedback': feedback,
            'method': 'adaptive_blend',
            'agreement': float(agreement),
            'cnn_pred': float(cnn_pred),
            'cnn_confidence': float(cnn_confidence),
            'similarity_score': float(similarity_score),
            'similarity_confidence': float(similarity_confidence),
            'adaptive_weights': {
                'cnn': float(adaptive_cnn_weight),
                'similarity': float(adaptive_sim_weight)
            }
        }
    
    def _generate_feedback(self, score: float, agreement: float,
                          cnn_conf: float, sim_conf: float) -> str:
        """Generate human-readable feedback based on score and confidence."""
        
        if score >= 0.75:
            base = "Well done! Your answer covers the main points accurately and concisely."
        elif score >= 0.50:
            base = "Good start, but your answer could be more precise. Review the reference material for additional detail."
        elif score >= 0.25:
            base = "Your answer shows some understanding but is incomplete. Try to include more key concepts."
        else:
            base = "Your answer does not adequately address the question. Review the question and reference material carefully."
        
        # Add agreement note
        if agreement < 0.5:
            addition = " There's some uncertainty in the scoring."
        else:
            addition = ""
        
        return base + addition
    
    def fit_calibration(self, predictions: np.ndarray, 
                       true_labels: np.ndarray):
        """Fit confidence calibration on validation set."""
        self.calibrator.fit(predictions, true_labels)


class TopicalRelevanceGate:
    """Check if student answer is topically relevant to the question."""
    
    def __init__(self, stopwords=None):
        """Initialize relevance gate."""
        if stopwords is None:
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                'she', 'it', 'we', 'they', 'what', 'which', 'who', 'where', 'when',
                'why', 'how', 'not', 'no'
            }
        self.stopwords = stopwords
    
    def compute_relevance(self, question: str, answer: str) -> float:
        """
        Compute relevance score using Jaccard similarity.
        
        Args:
            question: Question text
            answer: Student answer text
        
        Returns:
            Relevance score (0-1)
        """
        question_words = set(w.lower() for w in question.split() 
                            if w.isalnum() and w.lower() not in self.stopwords)
        answer_words = set(w.lower() for w in answer.split() 
                          if w.isalnum() and w.lower() not in self.stopwords)
        
        if not question_words or not answer_words:
            return 0.5
        
        intersection = len(question_words & answer_words)
        union = len(question_words | answer_words)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        return float(jaccard)


class FinalScorer:
    """Final scoring combining all components."""
    
    def __init__(self):
        self.adaptive_scorer = AdaptiveScorer()
        self.relevance_gate = TopicalRelevanceGate()
    
    def score_answer(self,
                    question: str,
                    answer: str,
                    cnn_pred: float,
                    cnn_confidence: float,
                    similarity_score: float,
                    similarity_confidence: float,
                    question_specific_pred: float = None,
                    ensemble_pred: float = None) -> Dict:
        """
        Compute final score using all available signals.
        
        Args:
            question: Question text
            answer: Student answer text
            cnn_pred: CNN prediction (0-1)
            cnn_confidence: CNN confidence (0-1)
            similarity_score: Semantic similarity score (0-1)
            similarity_confidence: Similarity confidence (0-1)
            question_specific_pred: Optional question-specific model pred
            ensemble_pred: Optional ensemble model pred
        
        Returns:
            Final scoring result dict
        """
        
        # Check topical relevance
        relevance_score = self.relevance_gate.compute_relevance(question, answer)
        
        # If ensemble or question-specific available, use them in scoring
        final_cnn_pred = cnn_pred
        
        if ensemble_pred is not None:
            # Blend with ensemble: 60% original, 40% ensemble
            final_cnn_pred = 0.6 * cnn_pred + 0.4 * ensemble_pred
        
        if question_specific_pred is not None:
            # Blend with question model: 70% previous, 30% question
            final_cnn_pred = 0.7 * final_cnn_pred + 0.3 * question_specific_pred
        
        # Compute adaptive score
        result = self.adaptive_scorer.compute_score(
            final_cnn_pred,
            cnn_confidence,
            similarity_score,
            similarity_confidence,
            relevance_score
        )
        
        result['relevance_score'] = float(relevance_score)
        
        return result


def create_scoring_system() -> FinalScorer:
    """Factory function to create a fully configured scorer."""
    return FinalScorer()
