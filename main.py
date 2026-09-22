"""Entry point for the trajectory prediction train/eval pipeline.

Examples from the project root::

    python -m main --model transformer
    python -m main --model mlp --validation-metrics mse mae rmse
"""

from __future__ import annotations

import argparse
import json
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
from train.validation import build_validation_metrics, validate

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PipelineConfig:
    model: str = "transformer"
    data_dir: Path = PROJECT_ROOT / "data" / "generated"
    checkpoint: Path = PROJECT_ROOT / "models" / "checkpoints" / "best.pt"
    results_dir: Path = PROJECT_ROOT / "results"
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
    validation_metrics: tuple[str, ...] = (
        "mse",
        "mae",
        "final_displacement_error",
    )
    monitor_metric: str = "mse"
    monitor_mode: str = "min"


def run_pipeline(config: PipelineConfig) -> dict[str, float]:
    """Prepare data, train, validate, evaluate, and persist results.

    Visualization is run separately with ``python -m train.visualize`` using the
    saved checkpoint and trajectory split.
    """
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.history_len < 1 or config.prediction_len < 1:
        raise ValueError("history_len and prediction_len must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not config.validation_metrics:
        raise ValueError("Choose at least one validation metric")
    if config.monitor_mode not in ("min", "max"):
        raise ValueError("monitor_mode must be 'min' or 'max'")
    if config.monitor_metric not in config.validation_metrics:
        raise ValueError("monitor_metric must be included in validation_metrics")
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
    validation_metric_functions = build_validation_metrics(
        list(config.validation_metrics),
        normalization["target_mean"],
        normalization["target_std"],
        target_indices,
    )
    dataset_options = {
        "history_len": config.history_len,
        "prediction_len": config.prediction_len,
        "input_indices": input_indices,
        "target_indices": target_indices,
    }

    def make_normalized_split(trajectories: list) -> NormalizedDataset:
        return NormalizedDataset(
            make_dataset(trajectories, **dataset_options), normalization
        )

    train_dataset = make_normalized_split(train_trajectories)
    validation_dataset = make_normalized_split(validation_trajectories)
    eval_dataset = make_normalized_split(eval_trajectories)
    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
    validation_loader = make_loader(
        validation_dataset, config.batch_size, shuffle=False
    )
    eval_loader = make_loader(eval_dataset, config.batch_size, shuffle=False)

    model_config = asdict(config)
    model_config["input_indices"] = input_indices
    model_config["target_indices"] = target_indices
    model_config["validation_metrics"] = list(config.validation_metrics)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    best_monitor_value = float("inf") if config.monitor_mode == "min" else -float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    print(f"Device: {device}")
    print(
        f"Windows: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, eval={len(eval_dataset)}"
    )
    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        validation_scores = validate(
            model, validation_loader, device, validation_metric_functions
        )
        monitor_value = validation_scores[config.monitor_metric]
        history_row = {
            "epoch": float(epoch),
            "train_normalized_mse": train_loss,
            **{
                f"validation_{name}": value for name, value in validation_scores.items()
            },
        }
        history.append(history_row)
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"train_normalized_mse={train_loss:.6f} "
            + " ".join(
                f"validation_{name}={value:.6f}"
                for name, value in validation_scores.items()
            )
        )

        improved = (
            monitor_value < best_monitor_value
            if config.monitor_mode == "min"
            else monitor_value > best_monitor_value
        )
        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "normalization": normalization,
                    "epoch": epoch,
                    "validation_metrics": validation_scores,
                    "monitor_metric": config.monitor_metric,
                    "monitor_mode": config.monitor_mode,
                    "seed": config.seed,
                },
                config.checkpoint,
            )

    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_loss = evaluate(model, eval_loader, device)
    eval_scores = validate(model, eval_loader, device, validation_metric_functions)

    serializable_config = asdict(config)
    for key in ("data_dir", "checkpoint", "results_dir"):
        serializable_config[key] = str(serializable_config[key])
    for key in ("input_indices", "target_indices", "validation_metrics"):
        serializable_config[key] = list(serializable_config[key])
    final_results = {
        "best_epoch": best_epoch,
        "monitor_metric": config.monitor_metric,
        "monitor_mode": config.monitor_mode,
        "best_monitor_value": best_monitor_value,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "eval_normalized_mse": eval_loss,
        "eval_metrics": eval_scores,
        "checkpoint": str(config.checkpoint),
        "data_dir": str(config.data_dir),
    }
    history_path = config.results_dir / "training_history.json"
    results_path = config.results_dir / "final_results.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    results_path.write_text(
        json.dumps({"config": serializable_config, "results": final_results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Best checkpoint: {config.checkpoint} (epoch {best_epoch}, "
        f"{config.monitor_metric} {best_monitor_value:.6f})"
    )
    print(f"Eval normalized MSE: {eval_loss:.6f}")
    print(f"Saved history to {history_path}")
    print(f"Saved final results to {results_path}")

    return {
        "best_epoch": float(best_epoch),
        "best_monitor_value": best_monitor_value,
        "eval_normalized_mse": eval_loss,
        **{f"eval_{key}": value for key, value in eval_scores.items()},
    }


def _parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("transformer", "mlp"), default="transformer"
    )
    parser.add_argument("--data-dir", type=Path, default=PipelineConfig.data_dir)
    parser.add_argument("--checkpoint", type=Path, default=PipelineConfig.checkpoint)
    parser.add_argument("--results-dir", type=Path, default=PipelineConfig.results_dir)
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
    parser.add_argument(
        "--validation-metrics",
        nargs="+",
        choices=("mse", "mae", "rmse", "final_displacement_error"),
        default=["mse", "mae", "final_displacement_error"],
    )
    parser.add_argument("--monitor-metric", default="mse")
    parser.add_argument("--monitor-mode", choices=("min", "max"), default="min")
    args = parser.parse_args()
    args.input_indices = tuple(args.input_indices)
    args.target_indices = tuple(args.target_indices)
    args.validation_metrics = tuple(args.validation_metrics)
    return PipelineConfig(**vars(args))


def main(config: PipelineConfig | None = None) -> dict[str, float]:
    """Run the full pipeline, callable with a config or from the command line."""
    return run_pipeline(config if config is not None else _parse_args())


if __name__ == "__main__":
    main()
