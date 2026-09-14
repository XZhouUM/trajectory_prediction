from __future__ import annotations

import os
import sys
from collections.abc import Mapping

import numpy as np

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from sim.model_registry import MODEL_REGISTRY

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    plt = None


class TrajectorySimulator:
    """Generic trajectory simulator for motion models.

    The simulator accepts model metadata to initialize a model instance. It can be used
    for any motion model that follows the MotionModel API.
    """

    def __init__(
        self,
        model_name: str = "KinematicBicycleModel",
        dt: float = 0.1,
        extra_params: dict | None = None,
    ):
        # keep model name for display purposes
        self.model_name = model_name
        constructor_kwargs = {
            "dt": dt,
        }

        if extra_params:
            constructor_kwargs.update(extra_params)

        try:
            model_class = MODEL_REGISTRY[model_name]
        except KeyError:
            raise ValueError(
                f"Unknown motion model: {model_name}. "
                f"Available models: {list(MODEL_REGISTRY)}"
            )

        self.model = model_class(**constructor_kwargs)

    def simulate(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """Simulate a trajectory from an initial state and control sequence."""
        if self.model is None:
            raise ValueError(
                "A motion model must be constructed in the simulator initializer using model_name."
            )

        if initial_state.ndim != 1 or initial_state.shape[0] != self.model.expected_dim:
            raise ValueError(
                "initial_state must be a 1D vector of length "
                f"{self.model.expected_dim}, got shape {initial_state.shape}."
            )

        if controls.ndim != 2 or controls.shape[1] != self.model.expected_control_dim:
            raise ValueError(
                "controls must be a 2D array of shape (N, "
                f"{self.model.expected_control_dim}), got shape {controls.shape}."
            )

        # Keep track of the state trajectory.
        states = np.empty((controls.shape[0] + 1, initial_state.shape[0]), dtype=float)
        states[0] = initial_state

        state = initial_state.copy()
        for i, control in enumerate(controls):
            self.model.state = state
            state = self.model.transition(state, control)
            states[i + 1] = state

        return states

    def plot_trajectory(
        self,
        states: np.ndarray,
        title: str | None = None,
        ax=None,
        x_index: int = 0,
        y_index: int = 1,
        color: str = "tab:blue",
        label: str | None = None,
        show: bool = False,
    ):
        """Plot x-y trajectory for a state history."""
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is required for trajectory visualization. Install it with `pip install matplotlib`."
            )

        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] < 2:
            raise ValueError(
                "State history must have shape (N, >=2) with at least x and y columns."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        x = states[:, x_index]
        y = states[:, y_index]
        ax.plot(x, y, color=color, linewidth=2.0, label=label or "trajectory")
        ax.scatter([x[0]], [y[0]], color="black", s=25, zorder=3)
        ax.scatter([x[-1]], [y[-1]], color="tab:red", s=30, zorder=3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title or f"{self.model_name} trajectory")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.axis("equal")

        if show:
            plt.show()

        return fig

    def plot_trajectories(
        self,
        trajectories: Mapping[str, np.ndarray],
        title: str | None = None,
        show: bool = False,
        save_path: str | None = None,
    ):
        """Plot multiple trajectories on the same axes."""
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is required for trajectory visualization. Install it with `pip install matplotlib`."
            )

        fig, ax = plt.subplots(figsize=(7, 7))
        for name, states in trajectories.items():
            self.plot_trajectory(states, ax=ax, title=None, label=name, color=None)

        if title is None:
            title = f"{self.model_name} trajectory comparison"
        ax.set_title(title)
        ax.legend(loc="best")

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return fig


def main() -> None:
    """Generate and visualize a few representative trajectories."""
    simulator = TrajectorySimulator(
        model_name="KinematicBicycleModel",
        dt=0.1,
    )

    # Define a few representative initial states and control sequences.
    initial_states = {
        "straight": np.array([0.0, 0.0, 0.0, 5.0]),  # x, y, theta, v
        "left_turn": np.array([0.0, 0.0, 0.0, 5.0]),
        "right_turn": np.array([0.0, 0.0, 0.0, 5.0]),
    }
    controls = {
        "straight": np.tile(np.array([0.0, 0.0]), (50, 1)),  # steering, acceleration
        "left_turn": np.tile(np.array([0.2, 0.0]), (50, 1)),
        "right_turn": np.tile(np.array([-0.2, 0.0]), (50, 1)),
    }

    trajectories = {
        name: simulator.simulate(initial_state=state, controls=control)
        for name, state, control in zip(
            initial_states.keys(), initial_states.values(), controls.values()
        )
    }

    for name, states in trajectories.items():
        print(f"{name}: first 5 states")
        print(states[:5])
        print()

    output_path = os.path.join(os.path.dirname(__file__), "trajectory_examples.png")
    simulator.plot_trajectories(
        trajectories, title="Typical vehicle trajectories", save_path=output_path
    )
    print(f"Saved trajectory visualization to {output_path}")


if __name__ == "__main__":
    main()
