from abc import ABC, abstractmethod

from torch import Tensor, nn


class TrajectoryPredictionModel(nn.Module, ABC):
    """Base interface for trajectory prediction models."""

    @abstractmethod
    def forward(self, history: Tensor) -> Tensor:
        """
        Args:
            history: [B, T_in, D]

        Returns:
            prediction: [B, T_out, D_out]
        """
        raise NotImplementedError