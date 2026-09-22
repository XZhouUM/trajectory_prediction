"""Model-agnostic evaluation loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def eval(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate ``model`` and return mean batch MSE."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for history, target in dataloader:
            history = history.to(device)
            target = target.to(device)
            prediction = model(history)
            total_loss += F.mse_loss(prediction, target).item()
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Evaluation dataloader is empty")
    return total_loss / num_batches


# A descriptive alias for callers that prefer not to use Python's built-in name.
evaluate = eval
