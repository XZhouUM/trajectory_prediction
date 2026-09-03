from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MotionModel(ABC):
    """Abstract base class for simple vehicle motion models.

    The vehicle state is stored as a 1D NumPy vector, for example:
    - [x, y, vx, vy]
    - [x, y, heading, speed]

    The control input is also a 1D NumPy vector, typically:
    - [acceleration, steering_angle]
    """

    def __init__(self, expected_state_dim: int | None, dt: float = 0.1):
        self.name = self.__class__.__name__
        self.dt = float(dt)
        # The default dimension of the state vector is 4, but it can be overridden by subclasses.
        self.expected_dim = expected_state_dim if expected_state_dim is not None else 4

        # Initialize the state vector to an empty array. This means the motion model can be defined without an initial state,
        # and the user can set it later using the setter of state property.
        self._state = np.array([], dtype=float)

    @property
    def state(self) -> np.ndarray:
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        state_vector = np.asarray(value, dtype=float)
        if state_vector.ndim != 1:
            raise ValueError(
                f"State must be a 1D vector, got shape {state_vector.shape}."
            )
        if state_vector.shape[0] != self.expected_dim:
            raise ValueError(
                f"Expected state dimension {self.expected_dim}, got {state_vector.shape[0]}."
            )
        self._state = state_vector

    @staticmethod
    @abstractmethod
    def transition(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Return the next state vector from the current state and control input."""
        raise NotImplementedError

    def step(self, control: np.ndarray) -> np.ndarray:
        """Advance the model by one time step using the given control input."""
        control_vector = np.asarray(control, dtype=float)
        if control_vector.ndim != 1:
            raise ValueError(
                f"Control must be a 1D vector, got shape {control_vector.shape}."
            )

        self.state = self.transition(self.state, control_vector)
        return self.state
