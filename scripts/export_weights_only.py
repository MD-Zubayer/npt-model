#!/usr/bin/env python3
"""Export weights-only checkpoint for lighter inference artifacts."""

import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Export weights-only checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output weights file (.pt)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    torch.save(state_dict, out_path)

    in_size = in_path.stat().st_size / (1024 ** 3)
    out_size = out_path.stat().st_size / (1024 ** 3)
    print(f"Saved weights-only checkpoint: {out_path}")
    print(f"Input size:  {in_size:.2f} GB")
    print(f"Output size: {out_size:.2f} GB")


if __name__ == "__main__":
    main()
