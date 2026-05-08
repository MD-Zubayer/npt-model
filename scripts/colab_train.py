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
    parser.add_argument("--save-weights-only", action="store_true", help="Export weights-only file and delete large full checkpoints")
    parser.add_argument("--weights-only-during-training", action="store_true", help="Save checkpoints in weights-only mode during training")
    parser.add_argument("--weights-dir", type=str, default="/content/drive/MyDrive/npt-weights", help="Where to store compact weights-only checkpoint")
    parser.add_argument("--weights-name", type=str, default="best_model_weights_only.pt", help="Output weights-only filename")
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
    env = os.environ.copy()
    if args.weights_only_during_training:
        env["NPT_SAVE_WEIGHTS_ONLY"] = "1"
        print("[colab] Enabled weights-only checkpoint mode during training.")
    subprocess.run(cmd, check=True, env=env)

    if args.save_weights_only:
        weights_dir = Path(args.weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        output_weights = weights_dir / args.weights_name

        # Prefer best_model; fallback to final_model.
        src_best = root / "outputs" / "best_model.pt"
        src_final = root / "outputs" / "final_model.pt"
        src = src_best if src_best.exists() else src_final
        if not src.exists():
            raise FileNotFoundError("No checkpoint found in outputs/ to export weights-only.")

        export_cmd = [
            "python3",
            "scripts/export_weights_only.py",
            "--input",
            str(src),
            "--output",
            str(output_weights),
        ]
        print("[colab] Exporting compact weights:", " ".join(export_cmd))
        subprocess.run(export_cmd, check=True)

        # Remove large checkpoints to save Drive quota.
        for p in [src_best, src_final]:
            if p.exists():
                p.unlink()
                print(f"[colab] Removed large checkpoint: {p}")
        print(f"[colab] Quota-safe artifact ready: {output_weights}")


if __name__ == "__main__":
    main()
