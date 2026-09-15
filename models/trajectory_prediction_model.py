from abc import ABC, abstractmethod

from torch import Tensor, nn


class TrajectoryPredictionModel(nn.Module, ABC):
    """Base interface for trajectory prediction models."""

    @abstractmethod
    def forward(self, history: Tensor) -> Tensor:
        """
        Args:
            history: [B, T_in, D], where
                B: batch size
                T_in: number of input time steps
                D: number of input features

        Returns:
            prediction: [B, T_out, D_out], where
                T_out: number of output time steps
                D_out: number of output features
        """
        raise NotImplementedError