"""Generate simulated vehicle trajectories and save train/eval/validation splits.

Each output file is a compressed NumPy archive containing a ``trajectories``
array with shape ``(N, frames, 4)``. State columns are ``[x, y, speed, heading]``.
The arrays can be loaded with :func:`load_trajectory_split` and passed directly
to :class:`data.trajectory_dataset.TrajectoryDataset`.

Run from the repository root with::

    python -m data.generate_dataset --num-trajectories 1000

or run this file directly. Output defaults to ``data/generated``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:  # Support both ``python -m data.generate_dataset`` and direct execution.
    from sim.trajectory_simulator import TrajectorySimulator
except ModuleNotFoundError:  # pragma: no cover - direct execution from data/
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sim.trajectory_simulator import TrajectorySimulator


SPLIT_NAMES = ("train", "eval", "validation")
STATE_COLUMNS = ("x", "y", "speed", "heading")
CONTROL_COLUMNS = ("acceleration", "steering_angle")


def _smooth_control_profile(
    rng: np.random.Generator,
    num_steps: int,
    maneuver: str,
) -> np.ndarray:
    """Create plausible, smoothly varying acceleration and steering inputs."""
    # Random knot spacing gives each trajectory a distinct control profile.
    num_knots = int(rng.integers(4, 9))
    knot_frames = np.linspace(0, num_steps - 1, num_knots)
    controls = np.zeros((num_steps, 2), dtype=np.float64)

    acceleration_scale = {
        "steady": 0.18,
        "accelerate": 0.65,
        "brake": 0.65,
        "turn_left": 0.25,
        "turn_right": 0.25,
        "lane_change": 0.22,
    }[maneuver]
    accel_knots = rng.normal(0.0, acceleration_scale, size=num_knots)
    if maneuver == "accelerate":
        accel_knots += rng.uniform(0.35, 0.75)
    elif maneuver == "brake":
        accel_knots -= rng.uniform(0.4, 0.8)

    steering_knots = rng.normal(0.0, 0.025, size=num_knots)
    if maneuver in ("turn_left", "turn_right"):
        direction = 1.0 if maneuver == "turn_left" else -1.0
        steering_knots += direction * rng.uniform(0.07, 0.18)
    elif maneuver == "lane_change":
        # Smooth S-turn with randomized phase and magnitude.
        phase = rng.uniform(-0.35, 0.35)
        steering_knots += rng.uniform(0.07, 0.14) * np.sin(
            np.linspace(-np.pi + phase, np.pi + phase, num_knots)
        )

    frame_positions = np.arange(num_steps)
    controls[:, 0] = np.interp(frame_positions, knot_frames, accel_knots)
    controls[:, 1] = np.interp(frame_positions, knot_frames, steering_knots)
    # Keep controls within ordinary passenger vehicle ranges.
    controls[:, 0] = np.clip(controls[:, 0], -2.5, 2.0)
    controls[:, 1] = np.clip(controls[:, 1], -0.35, 0.35)
    return controls


def simulate_trajectories(
    num_trajectories: int = 1000,
    num_steps: int = 80,
    dt: float = 0.1,
    seed: int = 42,
    model_name: str = "KinematicBicycleModel",
) -> tuple[np.ndarray, list[str]]:
    """Simulate a varied collection of vehicle trajectories.

    ``num_steps`` is the number of control steps; each result has
    ``num_steps + 1`` states because the initial state is included.
    """
    if num_trajectories < 10:
        raise ValueError("num_trajectories must be at least 10 for a 7:2:1 split")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if dt <= 0:
        raise ValueError("dt must be positive")

    rng = np.random.default_rng(seed)
    simulator = TrajectorySimulator(model_name=model_name, dt=dt)
    maneuver_names = (
        "steady",
        "accelerate",
        "brake",
        "turn_left",
        "turn_right",
        "lane_change",
    )
    maneuver_ids = np.arange(num_trajectories) % len(maneuver_names)
    rng.shuffle(maneuver_ids)

    trajectories = np.empty((num_trajectories, num_steps + 1, 4), dtype=np.float32)
    labels: list[str] = []
    for index, maneuver_id in enumerate(maneuver_ids):
        maneuver = maneuver_names[int(maneuver_id)]
        # Small variation in starting pose and speed prevents memorizing one setup.
        initial_state = np.array(
            [
                rng.uniform(-5.0, 5.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(4.0, 16.0),
                rng.uniform(-0.12, 0.12),
            ],
            dtype=np.float64,
        )
        controls = _smooth_control_profile(rng, num_steps, maneuver)
        states = simulator.simulate(initial_state=initial_state, controls=controls)
        # The bicycle model permits negative speed under prolonged braking; keep
        # sampled scenarios physically interpretable by rejecting and resampling
        # their braking profile with a gentler cap.
        if np.min(states[:, 2]) < 0.5:
            controls[:, 0] = np.maximum(controls[:, 0], -0.25)
            states = simulator.simulate(initial_state=initial_state, controls=controls)
        trajectories[index] = states.astype(np.float32)
        labels.append(maneuver)

    return trajectories, labels


def split_trajectories(
    trajectories: np.ndarray,
    labels: Sequence[str],
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Randomly split trajectories in approximately 7:2:1 proportions."""
    if len(trajectories) != len(labels):
        raise ValueError("labels and trajectories must have the same length")
    count = len(trajectories)
    train_count = int(count * 0.7)
    eval_count = int(count * 0.2)
    validation_count = count - train_count - eval_count
    if min(train_count, eval_count, validation_count) == 0:
        raise ValueError("at least 10 trajectories are required to populate all splits")

    indices = np.random.default_rng(seed).permutation(count)
    boundaries = (train_count, train_count + eval_count)
    index_groups = (
        indices[: boundaries[0]],
        indices[boundaries[0] : boundaries[1]],
        indices[boundaries[1] :],
    )
    label_array = np.asarray(labels, dtype="U16")
    return {
        name: (trajectories[group], label_array[group])
        for name, group in zip(SPLIT_NAMES, index_groups)
    }


def save_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path,
    *,
    dt: float,
    seed: int,
    model_name: str,
) -> dict[str, Path]:
    """Write one compressed ``.npz`` archive per split plus dataset metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        trajectories, labels = splits[split_name]
        path = output_path / f"{split_name}.npz"
        np.savez_compressed(
            path,
            trajectories=trajectories.astype(np.float32, copy=False),
            maneuver_labels=labels,
        )
        saved[split_name] = path

    total = sum(len(trajectories) for trajectories, _ in splits.values())
    metadata = {
        "format_version": 1,
        "model_name": model_name,
        "dt_seconds": float(dt),
        "state_columns": list(STATE_COLUMNS),
        "control_columns": list(CONTROL_COLUMNS),
        "trajectory_count": total,
        "split_counts": {name: len(splits[name][0]) for name in SPLIT_NAMES},
        "split_ratio": "7:2:1",
        "seed": int(seed),
        "trajectory_file_format": "NumPy .npz; trajectories has shape (N, frames, 4)",
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    saved["metadata"] = metadata_path
    return saved


def load_trajectory_split(path: str | Path) -> list[np.ndarray]:
    """Load a split into the list-of-arrays format used by TrajectoryDataset."""
    with np.load(path, allow_pickle=False) as archive:
        trajectories = archive["trajectories"]
    if trajectories.ndim != 3 or trajectories.shape[-1] != 4:
        raise ValueError(
            f"Expected trajectories with shape (N, frames, 4), got {trajectories.shape}"
        )
    return [trajectory for trajectory in trajectories]


def generate_dataset(
    output_dir: str | Path | None = None,
    num_trajectories: int = 1000,
    num_steps: int = 80,
    dt: float = 0.1,
    seed: int = 42,
    model_name: str = "KinematicBicycleModel",
) -> dict[str, Path]:
    """Generate, split, and save a complete trajectory dataset."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "generated"
    trajectories, labels = simulate_trajectories(
        num_trajectories=num_trajectories,
        num_steps=num_steps,
        dt=dt,
        seed=seed,
        model_name=model_name,
    )
    splits = split_trajectories(trajectories, labels, seed=seed + 1)
    return save_splits(splits, output_dir, dt=dt, seed=seed, model_name=model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="KinematicBicycleModel")
    args = parser.parse_args()

    outputs = generate_dataset(
        output_dir=args.output_dir,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        dt=args.dt,
        seed=args.seed,
        model_name=args.model_name,
    )
    for split_name, path in outputs.items():
        print(f"Saved {split_name}: {path}")


if __name__ == "__main__":
    main()
