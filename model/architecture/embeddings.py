"""
Embedding layers for the transformer model.
Includes token, position, and segment embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class TokenEmbedding(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, hidden_size: int):
        """
        Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, seq_length, hidden_size]
        """
        return self.embedding(input_ids) * math.sqrt(self.hidden_size)


class PositionalEmbedding(nn.Module):
    """Positional encoding using learned embeddings."""
    
    def __init__(self, max_seq_length: int, hidden_size: int):
        """
        Initialize positional embedding.
        
        Args:
            max_seq_length: Maximum sequence length
            hidden_size: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, hidden_size)
    
    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Get positional embeddings.
        
        Args:
            seq_length: Sequence length
            device: Device to create tensor on
            
        Returns:
            Positional embeddings [1, seq_length, hidden_size]
        """
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        return self.embedding(positions)


class SegmentEmbedding(nn.Module):
    """Segment/token type embedding."""
    
    def __init__(self, type_vocab_size: int, hidden_size: int):
        """
        Initialize segment embedding.
        
        Args:
            type_vocab_size: Number of segment types
            hidden_size: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
    
    def forward(self, token_type_ids: Optional[torch.Tensor] = None,
                seq_length: int = 0,
                batch_size: int = 0,
                device: torch.device = None) -> torch.Tensor:
        """
        Embed segment types.
        
        Args:
            token_type_ids: Segment IDs [batch_size, seq_length]
            seq_length: Sequence length (if token_type_ids is None)
            batch_size: Batch size (if token_type_ids is None)
            device: Device (if token_type_ids is None)
            
        Returns:
            Segment embeddings
        """
        if token_type_ids is not None:
            return self.embedding(token_type_ids)
        
        # Default: all zeros
        token_type_ids = torch.zeros(
            (batch_size, seq_length),
            dtype=torch.long,
            device=device
        )
        return self.embedding(token_type_ids)


class TransformerEmbedding(nn.Module):
    """Combined embedding layer for transformer."""
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 max_position_embeddings: int,
                 type_vocab_size: int = 2,
                 dropout_prob: float = 0.1):
        """
        Initialize transformer embeddings.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension
            max_position_embeddings: Maximum sequence length
            type_vocab_size: Number of segment types
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size)
        self.position_embedding = PositionalEmbedding(max_position_embeddings, hidden_size)
        self.segment_embedding = SegmentEmbedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            token_type_ids: Segment IDs [batch_size, seq_length]
            position_ids: Position IDs (optional)
            
        Returns:
            Combined embeddings [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length = input_ids.size()
        device = input_ids.device
        
        # Token embedding
        embeddings = self.token_embedding(input_ids)
        
        # Positional embedding
        if position_ids is None:
            pos_emb = self.position_embedding(seq_length, device)
            embeddings += pos_emb
        else:
            pos_emb = self.position_embedding.embedding(position_ids)
            embeddings += pos_emb
        
        # Segment embedding
        if token_type_ids is not None:
            seg_emb = self.segment_embedding(token_type_ids)
            embeddings += seg_emb
        
        # Layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
