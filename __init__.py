"""
Main project entry point and utilities.
"""

__version__ = "0.1.0"
__author__ = "NPT Team"

from model.architecture.config import ModelConfig, TrainingConfig, PresetConfigs
from model.architecture.transformer import NPTModel
from training.pretraining.trainer import Trainer

__all__ = [
    'ModelConfig',
    'TrainingConfig',
    'PresetConfigs',
    'NPTModel',
    'Trainer'
]
