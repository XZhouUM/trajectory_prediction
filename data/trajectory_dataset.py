import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Convert continuous trajectories into prediction windows."""

    def __init__(
        self,
        trajectories: list[np.ndarray],
        history_len: int,
        prediction_len: int,
        input_indices: list[int],
        target_indices: list[int],
    ):
        self.samples = []

        for trajectory in trajectories:
            num_frames = len(trajectory)

            for start in range(num_frames - history_len - prediction_len + 1):
                history = trajectory[
                    start : start + history_len,
                    input_indices,
                ]

                future = trajectory[
                    start + history_len : start + history_len + prediction_len,
                    target_indices,
                ]

                self.samples.append(
                    (
                        history.astype(np.float32),
                        future.astype(np.float32),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        history, future = self.samples[index]

        return (
            torch.from_numpy(history),
            torch.from_numpy(future),
        )
