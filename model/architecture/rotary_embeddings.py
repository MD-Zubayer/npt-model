"""
Rotary Position Embeddings (RoPE).
Modern position encoding used in Llama, PaLM, and other state-of-the-art models.
"""

import torch
import torch.nn as nn
from typing import Tuple
import math


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
    
    Features:
    - Extrapolates well beyond training sequence length
    - Preserves relative position information
    - Computationally efficient
    """
    
    def __init__(self,
                 dim: int,
                 max_seq_length: int = 2048,
                 base: float = 10000.0):
        """
        Initialize rotary embeddings.
        
        Args:
            dim: Dimension of each attention head
            max_seq_length: Maximum sequence length
            base: Base for frequency (default: 10000)
        """
        super().__init__()
        
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for cos and sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = None
    
    def _update_cos_sin_tables(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos and sin tables."""
        if seq_len <= self._seq_len_cached and device == self._cos_cached.device:
            return
        
        self._seq_len_cached = max(self.max_seq_length, seq_len)
        
        # Generate position indices
        t = torch.arange(
            self._seq_len_cached,
            device=device,
            dtype=self.inv_freq.dtype
        )
        
        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Different from paper, but matches reference implementations
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self._cos_cached = emb.cos()[None, None, :, :].to(dtype)
        self._sin_cached = emb.sin()[None, None, :, :].to(dtype)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key.
        
        Args:
            q: Query tensor [batch, heads, seq_len, dim]
            k: Key tensor [batch, heads, seq_len, dim]
            
        Returns:
            (rotated_q, rotated_k) tuple
        """
        seq_len = q.size(2)
        device = q.device
        dtype = q.dtype
        
        self._update_cos_sin_tables(seq_len, device, dtype)
        
        # Get cos and sin for this sequence length
        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]
        
        # Apply rotation
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class RotaryEmbeddingOpt(RotaryEmbedding):
    """
    Optimized rotary embeddings with pre-computed cache.
    """
    
    def __init__(self,
                 dim: int,
                 max_seq_length: int = 2048,
                 base: float = 10000.0):
        """Initialize optimized rotary embeddings."""
        super().__init__(dim, max_seq_length, base)
        
        # Pre-compute for full sequence length
        device = torch.device('cpu')
        self._update_cos_sin_tables(max_seq_length, device, torch.float32)


def apply_rotary_pos_emb(q: torch.Tensor,
                         k: torch.Tensor,
                         rope: RotaryEmbedding) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key.
    
    Args:
        q: Query tensor
        k: Key tensor
        rope: RotaryEmbedding module
        
    Returns:
        (rotated_q, rotated_k) tuple
    """
    return rope(q, k)


if __name__ == "__main__":
    # Test rotary embeddings
    batch_size = 2
    num_heads = 8
    seq_len = 64
    head_dim = 64
    
    rope = RotaryEmbedding(head_dim)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    q_rot, k_rot = rope(q, k)
    
    print(f"Query shape: {q_rot.shape}")
    print(f"Key shape: {k_rot.shape}")
