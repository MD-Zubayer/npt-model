"""
Main trainer class for model training.
Handles training loops, checkpointing, and evaluation.
"""

import logging
import os
from typing import Optional, Dict, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from datetime import datetime

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main training orchestrator.
    
    Features:
    - Mixed precision training
    - Distributed training support
    - Gradient accumulation
    - Checkpoint saving/loading
    - Metrics tracking
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 scheduler,
                 device: torch.device = None,
                 output_dir: str = './outputs',
                 mixed_precision: bool = False,
                 distributed: bool = False,
                 use_deepspeed: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Output directory
            mixed_precision: Use mixed precision
            distributed: Use distributed training
            use_deepspeed: Use DeepSpeed
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self.use_deepspeed = use_deepspeed
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        self.model = self.model.to(self.device)
    
    def train_epoch(self,
                   train_dataloader: DataLoader,
                   gradient_accumulation_steps: int = 1,
                   max_grad_norm: float = 1.0) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Training data loader
            gradient_accumulation_steps: Accumulation steps
            max_grad_norm: Max gradient norm for clipping
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            else:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                if max_grad_norm is not None and max_grad_norm > 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                
                # Optimizer step
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            eval_dataloader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def train(self,
             train_dataloader: DataLoader,
             eval_dataloader: Optional[DataLoader] = None,
             num_epochs: int = 3,
             gradient_accumulation_steps: int = 1,
             save_steps: int = 500,
             eval_steps: int = 500,
             max_grad_norm: float = 1.0):
        """
        Full training loop.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Gradient accumulation steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            max_grad_norm: Max gradient norm for clipping
        """
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(
                train_dataloader,
                gradient_accumulation_steps,
                max_grad_norm
            )
            logger.info(f"Train loss: {train_metrics['loss']:.4f}")
            
            # Evaluation
            if eval_dataloader is not None and self.global_step % eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(f"Eval loss: {eval_metrics['loss']:.4f}")
                
                # Save best model
                if eval_metrics['loss'] < self.best_loss:
                    self.best_loss = eval_metrics['loss']
                    self.save_checkpoint(self.output_dir / 'best_model.pt')
            
            # Save checkpoint
            if self.global_step % save_steps == 0:
                self.save_checkpoint(
                    self.output_dir / f'checkpoint-{self.global_step}.pt'
                )

        # Always persist final artifacts for short laptop runs.
        self.save_checkpoint(self.output_dir / "final_model.pt")
        if self.best_loss == float("inf"):
            self.save_checkpoint(self.output_dir / "best_model.pt")
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        weights_only_mode = os.getenv("NPT_SAVE_WEIGHTS_ONLY", "0") == "1"
        if weights_only_mode:
            checkpoint = self.model.state_dict()
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
            }

            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Training example would go here
