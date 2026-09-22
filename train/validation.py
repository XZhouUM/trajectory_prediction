"""Model-agnostic validation loop and built-in trajectory metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

Metric = Callable[[Tensor, Tensor], Tensor | float]


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics: Mapping[str, Metric],
) -> dict[str, float]:
    """Run the model over a validation loader and aggregate named metrics.

    Each metric receives prediction and target batches and should return a scalar
    batch mean. The results are weighted by batch size, so a short final batch is
    handled correctly.
    """
    if not metrics:
        raise ValueError("At least one validation metric must be provided")
    model.eval()
    totals = {name: 0.0 for name in metrics}
    num_samples = 0

    with torch.no_grad():
        for history, target in dataloader:
            prediction = model(history.to(device))
            target = target.to(device)
            batch_size = target.shape[0]
            for name, metric in metrics.items():
                value = metric(prediction, target)
                if isinstance(value, Tensor):
                    value = value.detach().item()
                totals[name] += float(value) * batch_size
            num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Validation dataloader is empty")
    return {name: total / num_samples for name, total in totals.items()}


def build_validation_metrics(
    names: list[str],
    target_mean: Tensor,
    target_std: Tensor,
    target_indices: list[int],
) -> dict[str, Metric]:
    """Create metrics computed in original feature units from normalized batches.

    Available names: ``mse``, ``mae``, ``rmse``, and ``final_displacement_error``.
    FDE is Euclidean x-y error at the final prediction frame and requires both
    state columns 0 (x) and 1 (y) among the target features.
    """
    target_mean = target_mean.detach().clone()
    target_std = target_std.detach().clone()
    metrics: dict[str, Metric] = {}

    def denormalize(values: Tensor) -> Tensor:
        return values * target_std.to(values.device) + target_mean.to(values.device)

    for name in names:
        if name == "mse":
            metrics[name] = lambda pred, target: (
                (denormalize(pred) - denormalize(target)).square().flatten(1).mean(1).mean()
            )
        elif name == "mae":
            metrics[name] = lambda pred, target: (
                (denormalize(pred) - denormalize(target)).abs().flatten(1).mean(1).mean()
            )
        elif name == "rmse":
            metrics[name] = lambda pred, target: (
                (denormalize(pred) - denormalize(target))
                .square()
                .flatten(1)
                .mean(1)
                .sqrt()
                .mean()
            )
        elif name == "final_displacement_error":
            if 0 not in target_indices or 1 not in target_indices:
                raise ValueError(
                    "final_displacement_error requires x and y (state indices 0 and 1) "
                    "in --target-indices"
                )
            x_index = target_indices.index(0)
            y_index = target_indices.index(1)

            def fde(prediction: Tensor, target: Tensor) -> Tensor:
                prediction_real = denormalize(prediction)
                target_real = denormalize(target)
                delta = prediction_real[:, -1, [x_index, y_index]] - target_real[
                    :, -1, [x_index, y_index]
                ]
                return torch.linalg.vector_norm(delta, dim=-1).mean()

            metrics[name] = fde
        else:
            raise ValueError(
                f"Unknown validation metric {name!r}. Available metrics: "
                "mse, mae, rmse, final_displacement_error"
            )
    return metrics
