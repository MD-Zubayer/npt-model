"""
Transformer model architecture.
Core transformer encoder combining attention and feed-forward layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import ModelConfig
from .embeddings import TransformerEmbedding
from .attention import Attention
from .rotary_embeddings import RotaryEmbedding, apply_rotary_pos_emb


class FeedForward(nn.Module):
    """Feed-forward network in transformer."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout_prob: float = 0.1):
        """Initialize feed-forward."""
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden_states_residual = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_residual)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single transformer layer."""
    
    def __init__(self, config: ModelConfig):
        """Initialize transformer layer."""
        super().__init__()
        
        self.attention = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout_prob,
            use_flash_attention=config.use_flash_attention
        )
        
        self.feed_forward = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_dropout_prob
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Attention
        attention_output = self.attention(hidden_states, attention_mask)
        
        # Feed-forward
        layer_output = self.feed_forward(attention_output)
        
        return layer_output


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple layers."""
    
    def __init__(self, config: ModelConfig):
        """Initialize transformer encoder."""
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Rotary embeddings if configured
        if config.embedding_type == 'rotary':
            self.rotary_emb = RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                config.max_position_embeddings
            )
        else:
            self.rotary_emb = None
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False) -> Tuple[torch.Tensor, tuple]:
        """
        Forward pass.
        
        Args:
            hidden_states: Input embeddings
            attention_mask: Attention mask
            output_hidden_states: Return hidden states from all layers
            
        Returns:
            (last_hidden_state, all_hidden_states)
        """
        all_hidden_states = [] if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            if self.gradient_checkpointing and self.training:
                hidden_states = self._checkpoint_forward(
                    layer,
                    hidden_states,
                    attention_mask
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return hidden_states, all_hidden_states
    
    @staticmethod
    def _checkpoint_forward(layer, hidden_states, attention_mask):
        """Gradient checkpointing forward pass."""
        return torch.utils.checkpoint.checkpoint(
            layer,
            hidden_states,
            attention_mask,
            use_reentrant=False
        )


class NPTModel(nn.Module):
    """
    NPT (Neural Pre-Trained) Model - Main transformer model.
    
    Architecture:
    - Token, Positional, and Segment Embeddings
    - Multi-layer Transformer Encoder
    - Optional Rotary Position Embeddings
    - Support for sequences up to max_position_embeddings tokens
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize NPT model."""
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.embeddings = TransformerEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            config.hidden_dropout_prob
        )
        
        # Transformer encoder
        self.encoder = TransformerEncoder(config)
        
        # Model dtype
        self.dtype = torch.bfloat16 if config.use_bfloat16 else torch.float32
    
    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[tuple]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            token_type_ids: Segment IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            output_hidden_states: Return hidden states
            
        Returns:
            (sequence_output, hidden_states) tuple
        """
        # Embeddings
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids
        )
        
        # Create attention mask if provided
        if attention_mask is not None:
            # Extended attention mask
            extended_attention_mask = self._get_extended_attention_mask(
                attention_mask, input_ids.size()
            )
        else:
            extended_attention_mask = None
        
        # Encoder
        sequence_output, all_hidden_states = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states
        )
        
        return sequence_output, all_hidden_states
    
    @staticmethod
    def _get_extended_attention_mask(attention_mask: torch.Tensor,
                                     input_shape: Tuple[int, int]) -> torch.Tensor:
        """Create extended attention mask."""
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask
        
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        
        return extended_attention_mask


if __name__ == "__main__":
    # Test model
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12
    )
    
    model = NPTModel(config)
    
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    outputs, hidden_states = model(input_ids)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output dtype: {outputs.dtype}")
