"""
Main pretraining script for NPT model.
Provides a lightweight local-training entrypoint for laptop testing.
"""

import argparse
import json
import logging
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.architecture.config import ModelConfig, TrainingConfig
from model.architecture.transformer import NPTModel
from training.pretraining.scheduler import get_scheduler
from training.pretraining.trainer import Trainer

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Random token dataset for quick smoke tests."""

    def __init__(self, size: int, seq_length: int, vocab_size: int):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class TokenizedTextDataset(Dataset):
    """Dataset তৈরি করে raw/cleaned text থেকে external tokenizer দিয়ে."""

    def __init__(self, texts, tokenizer, seq_length: int, max_samples: int):
        self.samples = []
        for text in texts[:max_samples]:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0].long()
            self.samples.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


class CausalLMWrapper(nn.Module):
    """Adds a lightweight LM head + CE loss on top of NPT encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = NPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden_states, _ = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _read_training_texts(max_texts: int = 10000) -> list[str]:
    candidate_paths = [
        Path("data/deduplicated/deduplicated.jsonl"),
        Path("data/filtered/filtered.jsonl"),
        Path("data/cleaned/cleaned.jsonl"),
        Path("data/raw/collected.jsonl"),
        Path("data/raw/train_data.txt"),
    ]
    texts = []
    for p in candidate_paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    content = row.get("content", "").strip()
                    if content:
                        texts.append(content)
                except Exception:
                    texts.append(line)
                if len(texts) >= max_texts:
                    break
        if texts:
            break
    return texts


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model_config = ModelConfig.load(args.model_config) if args.model_config else ModelConfig()
    train_config = TrainingConfig()

    if args.num_epochs is not None:
        train_config.num_train_epochs = args.num_epochs
    if args.batch_size is not None:
        train_config.per_device_train_batch_size = args.batch_size
    if args.output_dir:
        train_config.output_dir = args.output_dir

    tokenizer = None
    if args.tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            logger.info("Using external tokenizer: %s", args.tokenizer_name)
            if tokenizer.vocab_size and tokenizer.vocab_size != model_config.vocab_size:
                model_config.vocab_size = int(tokenizer.vocab_size)
                logger.info("Aligned model vocab_size to tokenizer: %d", model_config.vocab_size)
        except Exception as e:
            logger.warning("Tokenizer load failed, fallback to synthetic dataset: %s", e)

    model = CausalLMWrapper(model_config)
    logger.info("Model initialized (hidden=%d, layers=%d)", model_config.hidden_size, model_config.num_hidden_layers)

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_epsilon,
    )

    num_training_steps = max(1, train_config.num_train_epochs * max(1, args.dataset_size // train_config.per_device_train_batch_size))
    scheduler = get_scheduler(
        train_config.lr_scheduler_type,
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    if tokenizer is not None:
        texts = _read_training_texts(max_texts=max(args.dataset_size * 2, 1000))
        if texts:
            train_dataset = TokenizedTextDataset(texts, tokenizer, args.seq_length, args.dataset_size)
            eval_dataset = TokenizedTextDataset(texts, tokenizer, args.seq_length, max(8, args.dataset_size // 10))
            logger.info("Loaded tokenized text dataset: train=%d eval=%d", len(train_dataset), len(eval_dataset))
        else:
            logger.warning("No text files found for tokenizer-based training, using synthetic data.")
            train_dataset = DummyDataset(size=args.dataset_size, seq_length=args.seq_length, vocab_size=model_config.vocab_size)
            eval_dataset = DummyDataset(size=max(8, args.dataset_size // 10), seq_length=args.seq_length, vocab_size=model_config.vocab_size)
    else:
        train_dataset = DummyDataset(size=args.dataset_size, seq_length=args.seq_length, vocab_size=model_config.vocab_size)
        eval_dataset = DummyDataset(size=max(8, args.dataset_size // 10), seq_length=args.seq_length, vocab_size=model_config.vocab_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.per_device_train_batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_config.per_device_eval_batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=train_config.output_dir,
        mixed_precision=False,
    )

    trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        num_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        save_steps=max(1, train_config.save_steps),
        eval_steps=max(1, train_config.eval_steps),
        max_grad_norm=train_config.max_grad_norm,
    )
    logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining script for NPT model")
    parser.add_argument("--model-config", type=str, default=None, help="Path to model config JSON")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to store checkpoints")
    parser.add_argument("--dataset-size", type=int, default=128, help="Number of synthetic training samples")
    parser.add_argument("--seq-length", type=int, default=64, help="Synthetic sequence length")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs for quick local tests")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--tokenizer-name", type=str, default="google/gemma-2b", help="HF tokenizer name")
    main(parser.parse_args())
