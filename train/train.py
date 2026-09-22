"""Train an MLP or transformer on generated trajectory windows.

Examples (from the project root)::

    python -m train.train --model transformer
    python train/train.py --model mlp --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from train.utils import (
        build_model,
        compute_normalization,
        load_metadata,
        make_dataset,
        make_loader,
        read_trajectories,
        run_epoch,
        seed_everything,
        select_device,
    )
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from train.utils import (
        build_model,
        compute_normalization,
        load_metadata,
        make_dataset,
        make_loader,
        read_trajectories,
        run_epoch,
        seed_everything,
        select_device,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("transformer", "mlp"), default="transformer"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "generated"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "best.pt",
    )
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--prediction-len", type=int, default=10)
    parser.add_argument("--input-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--target-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a torch device name"
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden width")
    parser.add_argument(
        "--d-model", type=int, default=64, help="Transformer embedding width"
    )
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.history_len < 1 or args.prediction_len < 1:
        raise ValueError("history-len and prediction-len must be positive")
    if args.model == "transformer":
        if args.nhead < 1 or args.d_model % args.nhead:
            raise ValueError("d-model must be divisible by a positive nhead")
        if args.d_model % 2:
            raise ValueError("d-model must be even for the positional encoding")

    seed_everything(args.seed)
    device = select_device(args.device)
    train_trajectories = read_trajectories(args.data_dir, "train")
    validation_trajectories = read_trajectories(args.data_dir, "validation")
    dataset_args = {
        "history_len": args.history_len,
        "prediction_len": args.prediction_len,
        "input_indices": args.input_indices,
        "target_indices": args.target_indices,
    }
    train_dataset = make_dataset(train_trajectories, **dataset_args)
    validation_dataset = make_dataset(validation_trajectories, **dataset_args)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    validation_loader = make_loader(validation_dataset, args.batch_size, shuffle=False)

    config = {
        "model": args.model,
        "history_len": args.history_len,
        "prediction_len": args.prediction_len,
        "input_indices": args.input_indices,
        "target_indices": args.target_indices,
        "hidden_dim": args.hidden_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    normalization = compute_normalization(
        train_trajectories, args.input_indices, args.target_indices
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    data_metadata = load_metadata(args.data_dir)
    print(f"Device: {device}")
    print(
        f"Training windows: {len(train_dataset)}; validation windows: {len(validation_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, normalization, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, normalization)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_mse={train_metrics['normalized_mse']:.6f} "
            f"validation_mse={validation_metrics['normalized_mse']:.6f} "
            f"validation_mae={validation_metrics['mae']:.6f}"
        )
        if validation_metrics["normalized_mse"] < best_validation_loss:
            best_validation_loss = validation_metrics["normalized_mse"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "normalization": normalization,
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "seed": args.seed,
                "dataset_metadata": data_metadata,
            }
            torch.save(checkpoint, args.checkpoint)

    print(
        f"Best checkpoint saved to {args.checkpoint} (validation MSE {best_validation_loss:.6f})"
    )


if __name__ == "__main__":
    main()
