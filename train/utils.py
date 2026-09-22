"""Shared data, model, and metric helpers for training and evaluation."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    # Prefer normal imports when the package is available on sys.path.
    from data.generate_dataset import load_trajectory_split
    from data.trajectory_dataset import TrajectoryDataset
    from models.mlp import MLP
    from models.transformer import Transformer
except ModuleNotFoundError:
    # Support running scripts from the repository root (``python -m train.utils``
    # or direct execution) by inserting the project root into sys.path.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from data.generate_dataset import load_trajectory_split
    from data.trajectory_dataset import TrajectoryDataset
    from models.mlp import MLP
    from models.transformer import Transformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return device


def read_trajectories(data_dir: Path, split: str) -> list[np.ndarray]:
    path = data_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split} dataset at {path}. Run `python -m data.generate_dataset` first."
        )
    return load_trajectory_split(path)


def make_dataset(
    trajectories: list[np.ndarray],
    history_len: int,
    prediction_len: int,
    input_indices: list[int],
    target_indices: list[int],
) -> TrajectoryDataset:
    dataset = TrajectoryDataset(
        trajectories=trajectories,
        history_len=history_len,
        prediction_len=prediction_len,
        input_indices=input_indices,
        target_indices=target_indices,
    )
    if len(dataset) == 0:
        raise ValueError(
            "No prediction windows were created; check trajectory length, "
            "history_len, and prediction_len."
        )
    return dataset


def compute_normalization(
    trajectories: list[np.ndarray], input_indices: list[int], target_indices: list[int]
) -> dict[str, torch.Tensor]:
    """Compute feature statistics using training trajectories only."""
    states = np.concatenate(trajectories, axis=0)
    input_values = torch.as_tensor(states[:, input_indices], dtype=torch.float32)
    target_values = torch.as_tensor(states[:, target_indices], dtype=torch.float32)
    input_std = input_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_std = target_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "input_mean": input_values.mean(dim=0),
        "input_std": input_std,
        "target_mean": target_values.mean(dim=0),
        "target_std": target_std,
    }


def build_model(config: dict[str, Any]) -> nn.Module:
    model_name = config["model"].lower()
    if model_name == "mlp":
        return MLP(
            input_dim=len(config["input_indices"]),
            history_len=config["history_len"],
            prediction_len=config["prediction_len"],
            output_dim=len(config["target_indices"]),
            hidden_dim=config["hidden_dim"],
        )
    if model_name == "transformer":
        return Transformer(
            input_dim=len(config["input_indices"]),
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            prediction_len=config["prediction_len"],
            output_dim=len(config["target_indices"]),
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
        )
    raise ValueError("model must be either 'mlp' or 'transformer'")


def make_loader(
    dataset: TrajectoryDataset, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs_error = 0.0
    total_elements = 0
    criterion = nn.MSELoss(reduction="sum")

    for history, future in loader:
        history = history.to(device)
        future = future.to(device)
        input_mean = normalization["input_mean"].to(device)
        input_std = normalization["input_std"].to(device)
        target_mean = normalization["target_mean"].to(device)
        target_std = normalization["target_std"].to(device)

        normalized_history = (history - input_mean) / input_std
        normalized_future = (future - target_mean) / target_std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(normalized_history)
            loss_sum = criterion(prediction, normalized_future)
            if training:
                (loss_sum / normalized_future.numel()).backward()
                optimizer.step()

        elements = normalized_future.numel()
        total_loss += float(loss_sum.detach().cpu())
        original_prediction = prediction.detach() * target_std + target_mean
        total_abs_error += float((original_prediction - future).abs().sum().cpu())
        total_elements += elements

    return {
        "normalized_mse": total_loss / total_elements,
        "mae": total_abs_error / total_elements,
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def load_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}
