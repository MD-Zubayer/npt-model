"""
Model configuration module.
Defines all hyperparameters and model configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Main model configuration."""
    
    # Model architecture
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # Activation and embeddings
    hidden_act: str = 'gelu'
    embedding_type: str = 'rotary'  # 'standard' or 'rotary'
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Distributed training
    use_distributed: bool = False
    distributed_backend: str = 'deepspeed'  # 'fsdp' or 'deepspeed'
    gradient_checkpointing: bool = False
    
    # Model variants
    use_bfloat16: bool = False
    use_flash_attention: bool = True
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save(self, path: str):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimizer
    optimizer: str = 'adamw'
    lr_scheduler_type: str = 'linear'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Training loop
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Scheduling
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Saving & logging
    output_dir: str = './outputs'
    save_strategy: str = 'steps'
    save_steps: int = 1000
    save_total_limit: int = 3
    logging_steps: int = 100
    evaluation_strategy: str = 'steps'
    eval_steps: int = 500
    
    # Data
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


# Preset configurations
@dataclass
class PresetConfigs:
    """Preset model configurations."""
    
    @staticmethod
    def tiny() -> ModelConfig:
        """Tiny model for testing."""
        return ModelConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
    
    @staticmethod
    def small() -> ModelConfig:
        """Small model."""
        return ModelConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048
        )
    
    @staticmethod
    def base() -> ModelConfig:
        """Base model (default)."""
        return ModelConfig()
    
    @staticmethod
    def large() -> ModelConfig:
        """Large model."""
        return ModelConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096
        )
    
    @staticmethod
    def xlarge() -> ModelConfig:
        """XLarge model."""
        return ModelConfig(
            vocab_size=100000,
            hidden_size=1536,
            num_hidden_layers=48,
            num_attention_heads=24,
            intermediate_size=6144,
            max_position_embeddings=4096
        )
