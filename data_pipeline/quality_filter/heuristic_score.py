"""
Heuristic quality scoring for documents.
Assigns quality scores based on various linguistic features.
"""

import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)


class HeuristicScorer:
    """
    Score document quality using heuristic features.
    
    Features:
    - Length scoring
    - Punctuation balance
    - Vocabulary diversity
    - Sentence structure quality
    """
    
    def __init__(self, 
                 min_length: int = 50,
                 max_length: int = 100000):
        """
        Initialize heuristic scorer.
        
        Args:
            min_length: Minimum text length
            max_length: Maximum text length
        """
        self.min_length = min_length
        self.max_length = max_length
        
    def score_quality(self, text: str) -> float:
        """
        Score overall quality of text (0-1).
        
        Args:
            text: Input text
            
        Returns:
            Quality score
        """
        scores = {
            'length': self._score_length(text) * 0.2,
            'vocabulary': self._score_vocabulary(text) * 0.2,
            'punctuation': self._score_punctuation(text) * 0.15,
            'sentence_structure': self._score_sentence_structure(text) * 0.25,
            'readability': self._score_readability(text) * 0.2
        }
        
        overall = sum(scores.values())
        return min(overall, 1.0)
    
    def _score_length(self, text: str) -> float:
        """Score based on text length."""
        length = len(text)
        
        if length < self.min_length or length > self.max_length:
            return 0.0
        
        # Optimal length around 500-5000 characters
        optimal_length = 1000
        if length <= optimal_length:
            return length / optimal_length
        else:
            return max(0.5, 1.0 - (length - optimal_length) / (self.max_length - optimal_length))
    
    def _score_vocabulary(self, text: str) -> float:
        """Score vocabulary diversity."""
        words = text.lower().split()
        unique_words = len(set(words))
        
        if len(words) == 0:
            return 0.0
        
        diversity = unique_words / len(words)
        return min(diversity, 1.0)
    
    def _score_punctuation(self, text: str) -> float:
        """Score punctuation balance."""
        punct_count = sum(1 for c in text if c in '.,!?;:')
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        punct_ratio = punct_count / word_count
        optimal_ratio = 0.1
        
        if punct_ratio == 0:
            return 0.0
        return min(punct_ratio / optimal_ratio, 1.0)
    
    def _score_sentence_structure(self, text: str) -> float:
        """Score sentence structure quality."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Optimal sentence length: 10-20 words
        if avg_len < 5 or avg_len > 50:
            return 0.3
        elif 10 <= avg_len <= 20:
            return 1.0
        else:
            return 0.7
    
    def _score_readability(self, text: str) -> float:
        """Score readability (simple heuristic)."""
        # Check for common readability issues
        issues = 0
        max_issues = 5
        
        # Too many words in caps
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        if caps_ratio > 0.5:
            issues += 1
        
        # Too many special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and c != ' ') / max(1, len(text))
        if special_ratio > 0.3:
            issues += 1
        
        return 1.0 - (issues / max_issues)
    
    def filter_by_quality(self, 
                         documents: List[Dict],
                         threshold: float = 0.5) -> List[Dict]:
        """
        Filter documents by quality score.
        
        Args:
            documents: List of documents
            threshold: Minimum quality score
            
        Returns:
            Documents meeting quality threshold
        """
        filtered = []
        
        for doc in documents:
            content = doc.get('content', '')
            score = self.score_quality(content)
            
            if score >= threshold:
                doc_copy = doc.copy()
                doc_copy['quality_score'] = float(score)
                filtered.append(doc_copy)
        
        return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scorer = HeuristicScorer()
    # Example usage
