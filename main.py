"""Entry point for the trajectory prediction train/eval pipeline.

Examples from the project root::

    python -m main --model transformer
    python -m main --model mlp --history-len 20 --prediction-len 10
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from train.eval import eval as evaluate
from train.train import train
from train.utils import (
    NormalizedDataset,
    build_model,
    compute_normalization,
    make_dataset,
    make_loader,
    read_trajectories,
    seed_everything,
    select_device,
)

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PipelineConfig:
    model: str = "transformer"
    data_dir: Path = PROJECT_ROOT / "data" / "generated"
    checkpoint: Path = PROJECT_ROOT / "models" / "checkpoints" / "best.pt"
    history_len: int = 20
    prediction_len: int = 10
    input_indices: tuple[int, ...] = (0, 1, 2, 3)
    target_indices: tuple[int, ...] = (0, 1, 2, 3)
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "auto"
    hidden_dim: int = 128
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1


VisualizationCallback = Callable[
    [torch.nn.Module, list, PipelineConfig, dict[str, torch.Tensor], torch.device],
    None,
]


def run_pipeline(
    config: PipelineConfig,
    visualization_callback: VisualizationCallback | None = None,
) -> dict[str, float]:
    """Prepare data, train, select the best model, and evaluate the eval split.

    A future visualization function can be passed as ``visualization_callback``;
    it receives the best model and the raw eval trajectories after evaluation.
    """
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.history_len < 1 or config.prediction_len < 1:
        raise ValueError("history_len and prediction_len must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if config.model == "transformer":
        if config.nhead < 1 or config.d_model % config.nhead:
            raise ValueError("d_model must be divisible by a positive nhead")
        if config.d_model % 2:
            raise ValueError("d_model must be even for the positional encoding")

    seed_everything(config.seed)
    device = select_device(config.device)
    input_indices = list(config.input_indices)
    target_indices = list(config.target_indices)
    train_trajectories = read_trajectories(config.data_dir, "train")
    validation_trajectories = read_trajectories(config.data_dir, "validation")
    eval_trajectories = read_trajectories(config.data_dir, "eval")

    normalization = compute_normalization(
        train_trajectories, input_indices, target_indices
    )
    dataset_options = {
        "history_len": config.history_len,
        "prediction_len": config.prediction_len,
        "input_indices": input_indices,
        "target_indices": target_indices,
    }
    train_dataset = NormalizedDataset(
        make_dataset(train_trajectories, **dataset_options), normalization
    )
    validation_dataset = NormalizedDataset(
        make_dataset(validation_trajectories, **dataset_options), normalization
    )
    eval_dataset = NormalizedDataset(
        make_dataset(eval_trajectories, **dataset_options), normalization
    )
    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
    validation_loader = make_loader(
        validation_dataset, config.batch_size, shuffle=False
    )
    eval_loader = make_loader(eval_dataset, config.batch_size, shuffle=False)

    model_config = asdict(config)
    model_config["input_indices"] = input_indices
    model_config["target_indices"] = target_indices
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    print(f"Device: {device}")
    print(
        f"Windows: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, eval={len(eval_dataset)}"
    )
    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        validation_loss = evaluate(model, validation_loader, device)
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"train_mse={train_loss:.6f} validation_mse={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "normalization": normalization,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "seed": config.seed,
                },
                config.checkpoint,
            )

    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_loss = evaluate(model, eval_loader, device)
    metrics = {
        "best_epoch": float(best_epoch),
        "best_validation_mse": best_validation_loss,
        "eval_mse": eval_loss,
    }
    print(
        f"Best checkpoint: {config.checkpoint} (epoch {best_epoch}, "
        f"validation MSE {best_validation_loss:.6f})"
    )
    print(f"Eval MSE: {eval_loss:.6f}")

    if visualization_callback is not None:
        visualization_callback(model, eval_trajectories, config, normalization, device)
    return metrics


def _parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("transformer", "mlp"), default="transformer"
    )
    parser.add_argument("--data-dir", type=Path, default=PipelineConfig.data_dir)
    parser.add_argument("--checkpoint", type=Path, default=PipelineConfig.checkpoint)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--prediction-len", type=int, default=10)
    parser.add_argument("--input-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--target-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    args.input_indices = tuple(args.input_indices)
    args.target_indices = tuple(args.target_indices)
    return PipelineConfig(**vars(args))


def main(config: PipelineConfig | None = None) -> dict[str, float]:
    """Run the complete current pipeline; callable with a config or via CLI."""
    return run_pipeline(config if config is not None else _parse_args())


if __name__ == "__main__":
    main()
