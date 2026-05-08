"""
Toxicity filter for removing offensive/harmful content.
Uses pre-trained models to detect toxic language.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ToxicityFilter:
    """
    Filter out toxic and offensive content.
    
    Features:
    - Multi-language toxicity detection
    - Configurable toxicity threshold
    - Severity scoring
    """
    
    def __init__(self, 
                 toxicity_threshold: float = 0.5,
                 model_name: str = 'unbiased'):
        """
        Initialize toxicity filter.
        
        Args:
            toxicity_threshold: Toxicity score threshold (0-1)
            model_name: Toxicity model to use
        """
        self.toxicity_threshold = toxicity_threshold
        self.model_name = model_name
        
        # In production, load a real toxicity model
        # e.g., from Perspective API or detoxify
        
    def score_toxicity(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Score toxicity of text.
        
        Args:
            text: Input text
            
        Returns:
            (overall_score, detailed_scores) tuple
        """
        # Placeholder scoring logic
        # In production, use real toxicity model
        
        detailed_scores = {
            'toxic': self._score_pattern(text, 'toxic'),
            'severe_toxic': self._score_pattern(text, 'severe'),
            'obscene': self._score_pattern(text, 'obscene'),
            'threat': self._score_pattern(text, 'threat'),
            'insult': self._score_pattern(text, 'insult'),
            'identity_hate': self._score_pattern(text, 'hate')
        }
        
        overall_score = max(detailed_scores.values())
        return overall_score, detailed_scores
    
    def _score_pattern(self, text: str, category: str) -> float:
        """Simple pattern-based scoring (placeholder)."""
        # In production, use ML model
        bad_words = {
            'toxic': ['bad', 'stupid'],
            'severe': ['very bad', 'extremely'],
            'obscene': ['profanity'],
            'threat': ['burn', 'kill'],
            'insult': ['you suck', 'idiot'],
            'hate': ['hate group', 'hate speech']
        }
        
        words = bad_words.get(category, [])
        text_lower = text.lower()
        
        matches = sum(1 for word in words if word in text_lower)
        return min(matches * 0.1, 1.0)
    
    def is_toxic(self, text: str) -> bool:
        """Check if text is toxic."""
        score, _ = self.score_toxicity(text)
        return score > self.toxicity_threshold
    
    def filter_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Filter out toxic documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Non-toxic documents with toxicity scores
        """
        filtered = []
        
        for doc in documents:
            content = doc.get('content', '')
            score, details = self.score_toxicity(content)
            
            if score <= self.toxicity_threshold:
                doc_copy = doc.copy()
                doc_copy['toxicity_score'] = float(score)
                doc_copy['toxicity_details'] = details
                filtered.append(doc_copy)
            else:
                logger.debug(f"Filtered toxic content (score: {score})")
        
        return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    filter = ToxicityFilter()
    # Example usage
