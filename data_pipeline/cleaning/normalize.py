"""
Text normalization module for data cleaning.
Handles encoding, whitespace, punctuation normalization.
"""

import logging
import re
import unicodedata
from typing import List, Dict

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalize text data.
    
    Operations:
    - Unicode normalization
    - Whitespace cleaning
    - Encoding fixes
    - Case normalization (optional)
    """
    
    def __init__(self, language: str = 'en'):
        """Initialize normalizer."""
        self.language = language
        
    def normalize(self, text: str) -> str:
        """
        Apply all normalization operations.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Unicode normalization (NFD form)
        text = unicodedata.normalize('NFD', text)
        
        # Remove control characters
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        
        # Fix common encoding issues
        text = self._fix_encoding_issues(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        return text.strip()
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding problems."""
        # Replace curly quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace ellipsis variants with standard
        text = re.sub(r'\.{2,}', '...', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove spaces around punctuation
        text = re.sub(r'\s+([,.])', r'\1', text)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation."""
        # Normalize multiple punctuation
        text = re.sub(r'([?!]){2,}', r'\1', text)
        
        return text
    
    def normalize_batch(self, documents: List[Dict]) -> List[Dict]:
        """Normalize a batch of documents."""
        normalized = []
        
        for doc in documents:
            normalized_doc = doc.copy()
            if 'content' in normalized_doc:
                normalized_doc['content'] = self.normalize(normalized_doc['content'])
            normalized.append(normalized_doc)
        
        return normalized


if __name__ == "__main__":
    normalizer = TextNormalizer()
    # Example usage
