"""Data preparation helpers shared by the pipeline entry point."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.generate_dataset import load_trajectory_split
from data.trajectory_dataset import TrajectoryDataset
from models import build_model as model_registry_build_model


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
    """Compute feature statistics from training trajectories only."""
    states = np.concatenate(trajectories, axis=0)
    input_values = torch.as_tensor(states[:, input_indices], dtype=torch.float32)
    target_values = torch.as_tensor(states[:, target_indices], dtype=torch.float32)
    return {
        "input_mean": input_values.mean(dim=0),
        "input_std": input_values.std(dim=0, unbiased=False).clamp_min(1e-6),
        "target_mean": target_values.mean(dim=0),
        "target_std": target_values.std(dim=0, unbiased=False).clamp_min(1e-6),
    }


class NormalizedDataset(Dataset):
    """Normalize the history and target tensors returned by TrajectoryDataset."""

    def __init__(
        self,
        dataset: TrajectoryDataset,
        normalization: dict[str, torch.Tensor],
    ) -> None:
        self.dataset = dataset
        self.input_mean = normalization["input_mean"]
        self.input_std = normalization["input_std"]
        self.target_mean = normalization["target_mean"]
        self.target_std = normalization["target_std"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        history, target = self.dataset[index]
        history = (history - self.input_mean) / self.input_std
        target = (target - self.target_mean) / self.target_std
        return history, target


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Construct a selected model; kept in orchestration, outside train/eval."""
    name = config["model"].lower()
    kwargs: dict[str, Any] = {
        "input_dim": len(config["input_indices"]),
        "prediction_len": config["prediction_len"],
        "output_dim": len(config["target_indices"]),
    }
    if name == "mlp":
        kwargs.update(
            history_len=config["history_len"], hidden_dim=config["hidden_dim"]
        )
    elif name == "transformer":
        kwargs.update(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model '{name}'")
    return model_registry_build_model(name, **kwargs)
