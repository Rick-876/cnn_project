"""
reference_answers.py
Reference answer management and semantic similarity scoring.
Uses BERTScore and other metrics for comparing student answers to references.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List
import json
from pathlib import Path


class ReferenceAnswerBase:
    """Base class for managing reference answers."""
    
    def __init__(self, data_path: str = 'asag2024_all.csv'):
        """Initialize with dataset."""
        self.df = pd.read_csv(data_path)
        self.reference_map = self._build_reference_map()
    
    def _build_reference_map(self) -> Dict[str, List[str]]:
        """Build mapping from questions to reference answers."""
        ref_map = {}
        
        for _, row in self.df.iterrows():
            question = row['question']
            reference = row['reference_answer']
            
            if question not in ref_map:
                ref_map[question] = []
            
            if reference and str(reference).strip():
                ref_map[question].append(reference)
        
        # Keep only unique references per question
        for q in ref_map:
            ref_map[q] = list(set(ref_map[q]))
        
        return ref_map
    
    def get_references(self, question: str) -> List[str]:
        """Get reference answers for a question."""
        return self.reference_map.get(question, [])
    
    def get_stats(self) -> dict:
        """Get statistics on reference answers."""
        total_questions = len(self.reference_map)
        avg_refs_per_question = np.mean([len(refs) for refs in self.reference_map.values()])
        
        return {
            'total_questions': total_questions,
            'avg_references_per_question': float(avg_refs_per_question),
            'total_unique_references': sum(len(refs) for refs in self.reference_map.values())
        }


class SemanticSimilarityScorer:
    """Compute semantic similarity between student answers and references."""
    
    def __init__(self, use_bert_score: bool = True):
        """
        Initialize scorer.
        
        Args:
            use_bert_score: Whether to use BERTScore (requires bert_score package)
        """
        self.use_bert_score = use_bert_score
        self.bert_scorer = None
        
        if use_bert_score:
            try:
                from bert_score import score as bert_score
                self.bert_scorer = bert_score
                print("BERTScore initialized")
            except ImportError:
                print("Warning: bert_score not installed. Using fallback similarity.")
                self.use_bert_score = False
    
    def compute_similarity(self, student_answer: str, reference_answer: str) -> Dict:
        """
        Compute multiple similarity metrics.
        
        Args:
            student_answer: Student's answer text
            reference_answer: Reference answer text
        
        Returns:
            Dict with similarity scores
        """
        result = {}
        
        # Preprocessing
        student_tokens = set(student_answer.lower().split())
        ref_tokens = set(reference_answer.lower().split())
        
        # 1. Jaccard similarity
        if student_tokens or ref_tokens:
            intersection = len(student_tokens & ref_tokens)
            union = len(student_tokens | ref_tokens)
            result['jaccard'] = intersection / union if union > 0 else 0.0
        else:
            result['jaccard'] = 0.0
        
        # 2. Token overlap ratio
        if ref_tokens:
            result['token_overlap'] = len(student_tokens & ref_tokens) / len(ref_tokens)
        else:
            result['token_overlap'] = 0.0
        
        # 3. Length ratio
        student_len = len(student_answer)
        ref_len = len(reference_answer)
        if ref_len > 0:
            result['length_ratio'] = min(student_len / ref_len, 1.0)
        else:
            result['length_ratio'] = 0.0
        
        # 4. BERTScore if available
        if self.use_bert_score and self.bert_scorer:
            try:
                precision, recall, f1 = self.bert_scorer(
                    [student_answer],
                    [reference_answer],
                    lang='en',
                    verbose=False
                )
                result['bertscore_precision'] = float(precision[0])
                result['bertscore_recall'] = float(recall[0])
                result['bertscore_f1'] = float(f1[0])
            except Exception as e:
                print(f"BERTScore computation failed: {e}")
        
        # 5. Weighted composite score
        weights = {
            'jaccard': 0.3,
            'token_overlap': 0.2,
            'length_ratio': 0.1,
        }
        
        if 'bertscore_f1' in result:
            weights['bertscore_f1'] = 0.4
        
        composite = sum(
            result.get(k, 0) * w
            for k, w in weights.items()
        ) / sum(weights.values())
        
        result['composite'] = float(composite)
        
        return result
    
    def compare_to_references(self, student_answer: str, 
                             references: List[str]) -> Dict:
        """
        Compare student answer to multiple reference answers.
        
        Args:
            student_answer: Student's answer text
            references: List of reference answers
        
        Returns:
            Dict with best match and average scores
        """
        if not references:
            return {
                'best_similarity': 0.0,
                'avg_similarity': 0.0,
                'num_references': 0
            }
        
        similarities = []
        
        for ref in references:
            sim = self.compute_similarity(student_answer, ref)
            similarities.append(sim['composite'])
        
        return {
            'best_similarity': float(np.max(similarities)),
            'avg_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'num_references': len(references),
            'all_similarities': similarities
        }


class ReferenceAugmentation:
    """Augment references through paraphrasing and extension."""
    
    @staticmethod
    def create_keyword_summary(text: str, top_k: int = 5) -> str:
        """Extract and return key terms from text."""
        from collections import Counter
        
        tokens = text.lower().split()
        # Filter common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}
        filtered = [t.strip('.,!?;:') for t in tokens if t.lower() not in stopwords]
        
        counter = Counter(filtered)
        top_terms = [term for term, _ in counter.most_common(top_k)]
        
        return ' '.join(top_terms)
    
    @staticmethod
    def expand_reference(reference: str) -> List[str]:
        """Create variations of a reference answer."""
        variations = [reference]  # Original
        
        # Create keyword-only version
        keywords = ReferenceAugmentation.create_keyword_summary(reference)
        if keywords:
            variations.append(keywords)
        
        return variations


def build_reference_database(data_path: str = 'asag2024_all.csv',
                            output_path: str = 'reference_database.json'):
    """
    Build a reference answer database from the dataset.
    
    Args:
        data_path: Path to dataset CSV
        output_path: Path to save database
    """
    reference_base = ReferenceAnswerBase(data_path)
    
    database = {}
    for question, refs in reference_base.reference_map.items():
        database[question] = {
            'references': refs,
            'count': len(refs),
            'augmented': []
        }
        
        # Augment references
        for ref in refs:
            augmented = ReferenceAugmentation.expand_reference(ref)
            database[question]['augmented'].extend(augmented)
        
        database[question]['augmented'] = list(set(database[question]['augmented']))
    
    # Save database
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"Reference database saved to {output_path}")
    print(f"Total questions: {len(database)}")
    
    return database


def load_reference_database(path: str = 'reference_database.json') -> dict:
    """Load reference database from file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Reference database not found at {path}")
        return {}


if __name__ == '__main__':
    build_reference_database()
