from dataclasses import dataclass

from torch import nn


@dataclass
class TrainConfig:
    n_epochs: int
    lr: float
    bs: int
    loss: nn.Module
