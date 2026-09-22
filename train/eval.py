"""Evaluate a saved trajectory-prediction checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.utils import (build_model, load_checkpoint, make_dataset,
                         make_loader, read_trajectories, run_epoch,
                         select_device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "best.pt",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "generated"
    )
    parser.add_argument(
        "--split", choices=("eval", "validation", "train"), default="eval"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a torch device name"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    trajectories = read_trajectories(args.data_dir, args.split)
    dataset = make_dataset(
        trajectories,
        history_len=config["history_len"],
        prediction_len=config["prediction_len"],
        input_indices=config["input_indices"],
        target_indices=config["target_indices"],
    )
    loader = make_loader(dataset, args.batch_size, shuffle=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalization = {
        name: value.to(device) for name, value in checkpoint["normalization"].items()
    }
    metrics = run_epoch(model, loader, device, normalization)
    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"Split: {args.split} ({len(trajectories)} trajectories, {len(dataset)} windows)"
    )
    print(f"Normalized MSE: {metrics['normalized_mse']:.6f}")
    print(f"Position/state MAE (original units): {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
