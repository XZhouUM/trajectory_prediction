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

    def __init__(
        self,
        expected_state_dim: int | None,
        expected_control_dim: int | None,
        dt: float = 0.1,
    ):
        self.name = self.__class__.__name__
        self._dt = float(dt)
        # The default dimension of the state vector and control vector is set to 4 and 2, respectively, if not provided.
        self.expected_dim = expected_state_dim if expected_state_dim is not None else 4
        self.expected_control_dim = (
            expected_control_dim if expected_control_dim is not None else 2
        )

        # Initialize the state vector to an empty array. This means the motion model can be defined without an initial state, and the user can set it later using the setter of state property.
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

    @abstractmethod
    def transition(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Return the next state vector from the current state and control input."""
        raise NotImplementedError

    def step(self, control: np.ndarray) -> np.ndarray:
        """Advance the model by one time step using the given control input."""
        # Validate the state and control before proceeding with the transition through the state equation.
        if self._state.size == np.array([]).size:
            raise ValueError(
                "System state is not initialized. Please set the state before calling step()."
            )

        control_vector = np.asarray(control, dtype=float)
        if control_vector.ndim != 1:
            raise ValueError(
                f"Control must be a 1D vector, got shape {control_vector.shape}."
            )
        if control_vector.shape[0] != self.expected_control_dim:
            raise ValueError(
                f"Expected control dimension {self.expected_control_dim}, got {control_vector.shape[0]}."
            )

        self.state = self.transition(self._state, control_vector)
        return self.state
