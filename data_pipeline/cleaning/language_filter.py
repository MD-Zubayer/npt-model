"""
Language detection and filtering for text data.
Filters documents to keep only specified languages.
"""

import logging
from typing import List, Dict, Set
try:
    from langdetect import LangDetectException, detect_langs
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False
    LangDetectException = Exception

logger = logging.getLogger(__name__)


class LanguageFilter:
    """
    Filter text by language.
    
    Features:
    - Language detection
    - Multi-language filtering
    - Confidence-based filtering
    - Mixed language handling
    """
    
    def __init__(self, 
                 target_languages: Set[str] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize language filter.
        
        Args:
            target_languages: Set of language codes to keep (e.g., {'en', 'bn'})
            confidence_threshold: Minimum confidence for detection
        """
        self.target_languages = target_languages or {'en'}
        self.confidence_threshold = confidence_threshold
        
    def detect_language(self, text: str) -> tuple:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            (language_code, confidence) tuple
        """
        if len(text.split()) < 5:
            return None, 0.0
        
        if not HAS_LANGDETECT:
            # Fallback heuristic for environments without langdetect installed.
            bn_chars = sum(1 for ch in text if "\u0980" <= ch <= "\u09FF")
            if bn_chars > 0:
                return "bn", 0.8
            return "en", 0.6

        try:
            # Get all detected languages with probabilities
            detected = detect_langs(text)
            if detected:
                best = detected[0]
                return best.lang, best.prob
        except LangDetectException as e:
            logger.debug(f"Language detection failed: {e}")
        
        return None, 0.0
    
    def is_target_language(self, text: str) -> bool:
        """Check if text is in target language."""
        lang, confidence = self.detect_language(text)
        
        if lang is None:
            return False
        
        if confidence < self.confidence_threshold:
            return False
        
        return lang in self.target_languages
    
    def filter_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Filter documents by language.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Filtered documents with detected language metadata
        """
        filtered = []
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue
            
            lang, confidence = self.detect_language(content)
            
            if self.is_target_language(content):
                doc_copy = doc.copy()
                doc_copy['language'] = lang
                doc_copy['language_confidence'] = confidence
                filtered.append(doc_copy)
            else:
                logger.debug(f"Filtered out {lang} (confidence: {confidence})")
        
        return filtered
    
    def set_target_languages(self, languages: Set[str]):
        """Update target languages."""
        self.target_languages = languages
        logger.info(f"Target languages set to: {languages}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    filter_en = LanguageFilter({'en'})
    # Example usage
