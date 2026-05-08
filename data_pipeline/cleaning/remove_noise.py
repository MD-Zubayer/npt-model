"""
Noise removal from text data.
Removes spam, boilerplate, HTML artifacts, etc.
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


class NoiseRemover:
    """
    Remove noise and boilerplate from text.
    
    Operations:
    - HTML tag removal
    - Email/URL filtering
    - Numbers/symbols cleanup
    - Boilerplate detection
    """
    
    def __init__(self):
        """Initialize noise remover."""
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'https?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        
    def remove_noise(self, text: str, keep_urls: bool = False) -> str:
        """
        Remove noise from text.
        
        Args:
            text: Input text
            keep_urls: Whether to keep URLs
            
        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Remove URLs unless specified to keep
        if not keep_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        text = self.email_pattern.sub(' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive special characters
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', text)
        
        return text.strip()
    
    def is_boilerplate(self, text: str, threshold: float = 0.3) -> bool:
        """
        Detect if text is likely boilerplate.
        
        Args:
            text: Input text
            threshold: Boilerplate score threshold
            
        Returns:
            True if likely boilerplate
        """
        if len(text.split()) < 10:
            return True
        
        # Check for common boilerplate patterns
        boilerplate_indicators = [
            r'cookie|privacy|terms of service',
            r'copyright|all rights reserved',
            r'advertisement|sponsored|ad',
            r'javascript must be enabled',
        ]
        
        match_count = sum(
            1 for pattern in boilerplate_indicators
            if re.search(pattern, text, re.IGNORECASE)
        )
        
        return match_count / len(boilerplate_indicators) > threshold
    
    def remove_noise_batch(self, documents: List[Dict]) -> List[Dict]:
        """Remove noise from batch of documents."""
        cleaned = []
        
        for doc in documents:
            if self.is_boilerplate(doc.get('content', '')):
                logger.debug(f"Skipping boilerplate document")
                continue
            
            cleaned_doc = doc.copy()
            if 'content' in cleaned_doc:
                cleaned_doc['content'] = self.remove_noise(cleaned_doc['content'])
            
            if cleaned_doc['content'].strip():  # Only keep non-empty
                cleaned.append(cleaned_doc)
        
        return cleaned


if __name__ == "__main__":
    remover = NoiseRemover()
    # Example usage
