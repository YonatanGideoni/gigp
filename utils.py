from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainConfig:
    n_epochs: int
    lr: float
    bs: int
    loss: nn.Module


def train_loop(dataloader, model: nn, loss_fn: callable, optimizer: Optimizer):
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn: callable, verbose: bool = False) -> float:
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            assert pred.shape == y.shape, 'Error - prediction/label shape mismatch!'
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    if verbose:
        print(f"Avg test loss: {test_loss:>8f} \n")

    return test_loss


def pixels2coords(imgs: torch.Tensor):
    bs, c, h, w = imgs.shape

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()

    return coords
