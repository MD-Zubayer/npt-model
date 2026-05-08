"""Colab-friendly training entrypoint for NPT model."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run NPT training on Google Colab")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/npt-model", help="Project root inside Google Drive")
    parser.add_argument("--tokenizer-name", type=str, default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset-size", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.drive_root)
    if not root.exists():
        raise FileNotFoundError(f"Drive root not found: {root}")

    os.chdir(root)

    cmd = [
        "python3",
        "training/pretraining/train.py",
        "--num-epochs", str(args.epochs),
        "--dataset-size", str(args.dataset_size),
        "--seq-length", str(args.seq_length),
        "--batch-size", str(args.batch_size),
        "--tokenizer-name", args.tokenizer_name,
        "--output-dir", "./outputs",
    ]

    print("[colab] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
