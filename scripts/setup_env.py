#!/usr/bin/env python3
"""
Generate a laptop-friendly small testing config based on available hardware.
"""

import json
import os
import platform
from pathlib import Path


def detect_memory_gb() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return (page_size * pages) / (1024 ** 3)
    except Exception:
        return 8.0


def detect_gpu():
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            return {"enabled": False, "name": None}
        return {"enabled": True, "name": torch.cuda.get_device_name(0)}
    except Exception:
        return {"enabled": False, "name": None}


def build_small_config(ram_gb: float, has_gpu: bool) -> dict:
    if has_gpu and ram_gb >= 16:
        hidden_size, layers, heads, batch = 512, 6, 8, 8
    elif ram_gb >= 8:
        hidden_size, layers, heads, batch = 256, 4, 4, 4
    else:
        hidden_size, layers, heads, batch = 192, 2, 3, 2

    return {
        "model_config": {
            "vocab_size": 32000,
            "hidden_size": hidden_size,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": hidden_size * 4,
            "max_position_embeddings": 512,
            "embedding_type": "rotary",
            "use_flash_attention": False,
        },
        "training_overrides": {
            "num_epochs": 1,
            "batch_size": batch,
            "dataset_size": 128,
            "seq_length": 64,
            "device": "cuda" if has_gpu else "cpu",
        },
    }


def main():
    out_dir = Path("configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "small_laptop_config.json"

    ram_gb = detect_memory_gb()
    gpu_info = detect_gpu()
    cfg = build_small_config(ram_gb, gpu_info["enabled"])
    cfg["hardware_info"] = {
        "platform": platform.platform(),
        "ram_gb": round(ram_gb, 2),
        "gpu_enabled": gpu_info["enabled"],
        "gpu_name": gpu_info["name"],
    }

    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Generated: {out_path}")
    print(json.dumps(cfg["hardware_info"], indent=2))


if __name__ == "__main__":
    main()
