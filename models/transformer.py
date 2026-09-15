import torch
from torch import Tensor, nn

from .trajectory_prediction_model import TrajectoryPredictionModel


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
    ):
        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.shape[1]]


class Transformer(TrajectoryPredictionModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        prediction_len: int,
        output_dim: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.prediction_len = prediction_len
        self.output_dim = output_dim

        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=512,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(
                d_model,
                prediction_len * output_dim,
            ),
        )

    def forward(self, history: Tensor) -> Tensor:
        batch_size = history.shape[0]

        x = self.input_projection(history)

        x = self.positional_encoding(x)

        x = self.encoder(x)

        # Representation of the latest observed state.
        h = x[:, -1]

        prediction = self.head(h)

        return prediction.reshape(
            batch_size,
            self.prediction_len,
            self.output_dim,
        )