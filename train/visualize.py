"""Plot one saved model prediction against a trajectory from a saved data split.

Run repeatedly with different trajectory indices or start frames, independently
of the training and evaluation pipeline::

    python -m train.visualize --checkpoint models/checkpoints/best.pt
    python -m train.visualize --trajectory-index 12 --start-frame 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train.utils import build_model, load_checkpoint, read_trajectories, select_device

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def visualize_prediction(
    checkpoint_path: Path,
    data_dir: Path,
    split: str = "eval",
    trajectory_index: int = 0,
    start_frame: int = 0,
    output_path: Path | None = None,
    device_name: str = "auto",
    show: bool = False,
) -> Path:
    """Create an x-y overlay of observed history, true future, and prediction."""
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    input_indices = list(config["input_indices"])
    target_indices = list(config["target_indices"])
    if 0 not in target_indices or 1 not in target_indices:
        raise ValueError(
            "The checkpoint must predict x and y (state columns 0 and 1) to plot a path"
        )
    if trajectory_index < 0 or start_frame < 0:
        raise ValueError("trajectory-index and start-frame must be non-negative")

    trajectories = read_trajectories(data_dir, split)
    if trajectory_index >= len(trajectories):
        raise IndexError(
            f"trajectory-index {trajectory_index} is out of range for {split} "
            f"({len(trajectories)} trajectories)"
        )
    trajectory = trajectories[trajectory_index]
    history_len = int(config["history_len"])
    prediction_len = int(config["prediction_len"])
    stop_history = start_frame + history_len
    stop_future = stop_history + prediction_len
    if stop_future > len(trajectory):
        raise ValueError(
            f"Window ending at frame {stop_future} exceeds trajectory length "
            f"{len(trajectory)}"
        )

    history_raw = trajectory[start_frame:stop_history, input_indices]
    history = torch.as_tensor(history_raw, dtype=torch.float32, device=device)
    normalization = checkpoint["normalization"]
    input_mean = normalization["input_mean"].to(device)
    input_std = normalization["input_std"].to(device)
    target_mean = normalization["target_mean"].to(device)
    target_std = normalization["target_std"].to(device)
    history_normalized = ((history - input_mean) / input_std).unsqueeze(0)

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction_normalized = model(history_normalized)[0]
    prediction = (prediction_normalized * target_std + target_mean).cpu().numpy()

    x_target_col = target_indices.index(0)
    y_target_col = target_indices.index(1)
    history_xy = trajectory[start_frame:stop_history, :2]
    future_xy = trajectory[stop_history:stop_future, :2]
    predicted_xy = prediction[:, [x_target_col, y_target_col]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        history_xy[:, 0],
        history_xy[:, 1],
        color="tab:blue",
        marker="o",
        markersize=3,
        label="Observed history",
    )
    ax.plot(
        future_xy[:, 0],
        future_xy[:, 1],
        color="tab:green",
        marker="o",
        markersize=3,
        label="Ground-truth future",
    )
    ax.plot(
        predicted_xy[:, 0],
        predicted_xy[:, 1],
        color="tab:red",
        marker="o",
        markersize=3,
        linestyle="--",
        label="Predicted future",
    )
    ax.scatter(*history_xy[0], color="black", marker="s", s=35, label="Window start")
    ax.scatter(*future_xy[-1], color="tab:green", marker="x", s=50)
    ax.scatter(*predicted_xy[-1], color="tab:red", marker="x", s=50)
    ax.set_title(
        f"{config['model']} | {split} trajectory {trajectory_index} | "
        f"frames {start_frame}:{stop_future}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "results"
            / "visualizations"
            / f"{split}_trajectory_{trajectory_index}_frame_{start_frame}.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "best.pt",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "generated"
    )
    parser.add_argument(
        "--split", choices=("eval", "validation", "train"), default="eval"
    )
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--show", action="store_true", help="also display the plot interactively"
    )
    args = parser.parse_args()
    output = visualize_prediction(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        trajectory_index=args.trajectory_index,
        start_frame=args.start_frame,
        output_path=args.output,
        device_name=args.device,
        show=args.show,
    )
    print(f"Saved trajectory plot to {output}")


if __name__ == "__main__":
    main()
