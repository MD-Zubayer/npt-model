"""
Semantic deduplication using embeddings.
Removes semantically similar documents while keeping diverse content.
"""

import logging
from typing import List, Dict, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """
    Remove semantic duplicates from document collection.
    
    Features:
    - Embedding-based similarity detection
    - Configurable similarity threshold
    - Efficient nearest neighbor search
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.95,
                 embedding_model: str = None):
        """
        Initialize semantic deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering documents duplicates
            embedding_model: Name of embedding model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.embeddings = {}
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        In practice, this would use a real embedding model.
        """
        # Placeholder: use simple hash-based embedding
        # In production, use transformers or similar
        from hashlib import md5
        hash_val = int(md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(384)  # MiniLM dimensionality
    
    def deduplicate(self, documents: List[Dict]) -> List[Dict]:
        """
        Remove semantic duplicates from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Deduplicated documents
        """
        if not documents:
            return []
        
        # Get embeddings
        embeddings = np.array([self._get_embedding(doc['content']) for doc in documents])
        
        # Compute similarity matrix
        similarities = cosine_similarity(embeddings)
        
        # Find unique documents
        unique_indices = set(range(len(documents)))
        duplicates = set()
        
        for i in range(len(documents)):
            if i in duplicates:
                continue
            
            for j in range(i + 1, len(documents)):
                if j in duplicates:
                    continue
                
                if similarities[i][j] > self.similarity_threshold:
                    # Keep the longer document (more content)
                    if len(documents[j]['content']) > len(documents[i]['content']):
                        duplicates.add(i)
                        unique_indices.discard(i)
                    else:
                        duplicates.add(j)
                        unique_indices.discard(j)
        
        result = [documents[i] for i in sorted(unique_indices)]
        logger.info(f"Removed {len(documents) - len(result)} duplicates")
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    deduplicator = SemanticDeduplicator()
    # Example usage
