"""
Model quantization for inference optimization.
Reduces model size and speeds up inference.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Quantize models for inference.
    
    Methods:
    - INT8 Quantization
    - Dynamic Quantization
    - QAT (Quantization Aware Training)
    """
    
    @staticmethod
    def quantize_int8(model: nn.Module) -> nn.Module:
        """
        Quantize model to INT8.
        
        Args:
            model: Model to quantize
            
        Returns:
            Quantized model
        """
        logger.info("Applying INT8 quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def quantize_float16(model: nn.Module) -> nn.Module:
        """
        Convert model to float16 (half precision).
        
        Args:
            model: Model to convert
            
        Returns:
            Half-precision model
        """
        logger.info("Converting to float16...")
        return model.half()
    
    @staticmethod
    def export_to_onnx(model: nn.Module,
                      output_path: str,
                      input_shape: tuple = (1, 128)):
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            output_path: Output path for ONNX model
            input_shape: Shape of input tensor
        """
        logger.info(f"Exporting to ONNX: {output_path}")
        
        dummy_input = torch.randn(*input_shape, dtype=torch.long)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input_ids'],
            output_names=['output'],
            dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}},
            verbose=False,
            opset_version=14
        )
        
        logger.info(f"✓ ONNX model saved to {output_path}")
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
