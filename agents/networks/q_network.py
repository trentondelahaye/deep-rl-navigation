import torch
import torch.nn as nn

from torch.nn.functional import leaky_relu
from typing import Iterable

LayerSize = int


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        fc_layers: Iterable[LayerSize] = (64, 64),
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        layer_sizes = [state_size] + list(fc_layers) + [action_size]
        self.fc_layers = [
            nn.Linear(input_size, output_size)
            for input_size, output_size in zip(layer_sizes, layer_sizes[1:])
        ]

    @property
    def last_layer(self) -> nn.Linear:
        return self.fc_layers[-1]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.fc_layers[:-1]:
            x = leaky_relu(layer(x))
        return self.last_layer(x)
