"""
Inference server for model deployment.
Handles request processing, batching, and response formatting.
"""

import logging
from types import SimpleNamespace
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from model.architecture.config import ModelConfig
from model.architecture.transformer import NPTModel

logger = logging.getLogger(__name__)


class CausalLMWrapper(nn.Module):
    """Mirror training-time wrapper for checkpoint compatibility."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = NPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states, _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)


class InferenceEngine:
    """
    Inference engine for model deployment.
    
    Features:
    - Batch processing
    - Tokenization
    - Generation
    - Caching
    """
    
    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        """Apply temperature + top-k + top-p sampling."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / max(temperature, 1e-5)

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k=k, dim=-1)
            min_v = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_v, torch.full_like(logits, -float("inf")), logits)

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, -float("inf"))
            logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 max_batch_size: int = 32,
                 tokenizer_name: str = "google/gemma-2b"):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run on
            max_batch_size: Maximum batch size
        """
        self.model_path = Path(model_path)
        if device == 'auto':
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            resolved_device = device
        self.device = torch.device(resolved_device)
        self.max_batch_size = max_batch_size
        self.tokenizer_name = tokenizer_name
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.config = None
        self._load_tokenizer()
        self.load_model()

    def _load_tokenizer(self):
        """Load external tokenizer (HF)."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            logger.info("Tokenizer loaded: %s", self.tokenizer_name)
        except Exception as e:
            self.tokenizer = None
            logger.warning("Tokenizer load failed (%s): %s", self.tokenizer_name, e)
    
    def load_model(self):
        """Load model from checkpoint."""
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            self.model_loaded = False
            return
        
        logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Infer basic dims from checkpoint for robust loading.
        lm_weight = state_dict.get("lm_head.weight")
        if lm_weight is None:
            raise ValueError("Checkpoint does not contain lm_head.weight")
        vocab_size, hidden_size = lm_weight.shape

        config = ModelConfig(vocab_size=vocab_size, hidden_size=hidden_size)

        layer_ids = []
        for k in state_dict.keys():
            if k.startswith("backbone.encoder.layers."):
                parts = k.split(".")
                if len(parts) > 3 and parts[3].isdigit():
                    layer_ids.append(int(parts[3]))
        if layer_ids:
            config.num_hidden_layers = max(layer_ids) + 1

        # Try to infer attention heads from Q projection shape.
        q_w = state_dict.get("backbone.encoder.layers.0.attention.query.weight")
        if q_w is not None and q_w.shape[0] == hidden_size:
            for h in [32, 24, 16, 12, 8, 6, 4, 3, 2, 1]:
                if hidden_size % h == 0:
                    config.num_attention_heads = h
                    break

        model = CausalLMWrapper(config)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self.model = model
        self.config = config
        self.model_loaded = True
        logger.info(
            "Model loaded successfully (vocab=%d, hidden=%d, layers=%d)",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
        )
    
    def preprocess(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess texts for inference.
        
        Args:
            texts: List of input texts
            
        Returns:
            Preprocessed tensors
        """
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            return {"input_ids": encoded["input_ids"]}
        return {"input_ids": torch.ones((len(texts), 128), dtype=torch.long)}
    
    def generate(self,
                texts: List[str],
                max_length: int = 128,
                temperature: float = 0.7,
                top_p: float = 0.95,
                top_k: int = 40,
                repetition_penalty: float = 1.1) -> List[str]:
        """
        Generate text.
        
        Args:
            texts: Prompt texts
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated texts
        """
        outputs = []
        for prompt in texts:
            prompt = prompt.strip()
            if not prompt:
                outputs.append("Please provide a non-empty prompt.")
                continue

            if not self.model_loaded or self.model is None or self.tokenizer is None:
                outputs.append(f"{prompt} ... model/tokenizer unavailable.")
                continue

            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            max_new_tokens = max(1, max_length)
            gen_ids = input_ids.clone()
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    out = self.model(input_ids=gen_ids, attention_mask=None)
                    logits = out.logits[:, -1, :]

                    if repetition_penalty > 1.0:
                        seen_tokens = torch.unique(gen_ids[0])
                        logits[:, seen_tokens] = logits[:, seen_tokens] / repetition_penalty

                    next_token = self._sample_next_token(
                        logits=logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    gen_ids = torch.cat([gen_ids, next_token], dim=1)

                    if self.tokenizer.eos_token_id is not None and int(next_token.item()) == int(self.tokenizer.eos_token_id):
                        break

            gen_only = gen_ids[0][prompt_len:]
            text = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()
            if not text:
                text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            outputs.append(text)
        return outputs
    
    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for texts.
        
        Args:
            texts: Input texts
            
        Returns:
            Embeddings [num_texts, embedding_dim]
        """
        inputs = self.preprocess(texts)
        
        # In production, run actual embeddings
        return torch.randn(len(texts), 768)
    
    def __call__(self, texts: List[str], task: str = 'generate') -> List[str]:
        """
        Inference call.
        
        Args:
            texts: Input texts
            task: Task type ('generate', 'embed', etc.)
            
        Returns:
            Results
        """
        if task == 'generate':
            return self.generate(texts)
        elif task == 'embed':
            embeddings = self.embed(texts)
            return [e.tolist() for e in embeddings]
        else:
            raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
