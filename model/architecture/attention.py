"""
Multi-head attention mechanisms.
Includes standard attention and flash attention variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = math.sqrt(self.attention_head_size)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for attention calculation."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_size]
            key: Key tensor [batch_size, seq_len, hidden_size]
            value: Value tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            (attention_output, attention_weights) tuple
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.transpose_for_scores(self.query(query))
        k = self.transpose_for_scores(self.key(key))
        v = self.transpose_for_scores(self.value(value))
        
        # Attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape output
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.all_head_size)
        
        return context, attention_weights


class AttentionOutput(nn.Module):
    """Attention output projection and layer norm."""
    
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        """Initialize attention output layer."""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention output.
        
        Args:
            hidden_states: Attention context
            input_tensor: Residual connection
            
        Returns:
            Output tensor
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlashAttention(nn.Module):
    """
    Flash Attention: Fast and Memory-Efficient Attention.
    Placeholder implementation with standard attention semantics.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1):
        """Initialize flash attention."""
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            dropout_prob
        )
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for flash attention."""
        # In production, would use optimized CUDA kernels
        return self.attention(query, key, value, attention_mask)


class Attention(nn.Module):
    """Wrapper for standard + output projection + residual."""
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 dropout_prob: float = 0.1,
                 use_flash_attention: bool = False):
        """Initialize attention module."""
        super().__init__()
        
        if use_flash_attention:
            self.self_attention = FlashAttention(
                hidden_size,
                num_attention_heads,
                dropout_prob
            )
        else:
            self.self_attention = MultiHeadAttention(
                hidden_size,
                num_attention_heads,
                dropout_prob
            )
        
        self.output = AttentionOutput(hidden_size, dropout_prob)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for attention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        attention_output, _ = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask
        )
        
        output = self.output(attention_output, hidden_states)
        return output
