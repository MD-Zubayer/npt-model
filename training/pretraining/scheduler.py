"""
Learning rate scheduler for training.
Implements various scheduling strategies.
"""

import math
from typing import List
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


class WarmupLinearScheduler(LambdaLR):
    """Linear warmup followed by linear decay."""
    
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        """
        Initialize warmup linear scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        super().__init__(optimizer, lr_lambda)


class WarmupCosineScheduler(LambdaLR):
    """Linear warmup followed by cosine annealing."""
    
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        """Initialize warmup cosine scheduler."""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        super().__init__(optimizer, lr_lambda)


class WarmupConstantScheduler(LambdaLR):
    """Linear warmup followed by constant learning rate."""
    
    def __init__(self, optimizer, num_warmup_steps: int):
        """Initialize warmup constant scheduler."""
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        
        super().__init__(optimizer, lr_lambda)


def get_scheduler(name: str, optimizer, **kwargs):
    """
    Get learning rate scheduler.
    
    Args:
        name: Scheduler name ('linear', 'cosine', 'constant')
        optimizer: PyTorch optimizer
        **kwargs: Scheduler arguments
        
    Returns:
        LR scheduler instance
    """
    schedulers = {
        'linear': WarmupLinearScheduler,
        'cosine': WarmupCosineScheduler,
        'constant': WarmupConstantScheduler,
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")
    
    scheduler_class = schedulers[name]
    return scheduler_class(optimizer, **kwargs)
