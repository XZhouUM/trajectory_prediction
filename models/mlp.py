from torch import Tensor, nn

from .trajectory_prediction_model import TrajectoryPredictionModel


class MLP(TrajectoryPredictionModel):
    def __init__(
        self,
        input_dim: int,
        history_len: int,
        prediction_len: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.prediction_len = prediction_len
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim * history_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                prediction_len * output_dim,
            ),
        )

    def forward(self, history: Tensor) -> Tensor:
        batch_size = history.shape[0]

        x = history.flatten(start_dim=1)

        prediction = self.network(x)

        return prediction.reshape(
            batch_size,
            self.prediction_len,
            self.output_dim,
        )