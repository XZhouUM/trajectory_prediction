"""Model-agnostic training loop for one epoch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train ``model`` for one epoch and return mean batch MSE.

    The dataloader is responsible for providing model-ready ``(history, target)``
    tensors. No model type or trajectory-specific details are assumed here.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for history, target in dataloader:
        history = history.to(device)
        target = target.to(device)
        prediction = model(history)
        loss = F.mse_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training dataloader is empty")
    return total_loss / num_batches
