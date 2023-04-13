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


def test_loop(dataloader, model, loss_fn: callable, classification: bool = False, verbose: bool = False) -> float:
    num_batches = len(dataloader)
    test_loss = 0
    correct_labels = 0
    tot_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            assert classification or pred.shape == y.shape, 'Error - prediction/label shape mismatch!'

            if classification:
                tot_samples += y.shape[0]
                correct_labels += (pred.max(1).indices == y).sum().item()
            else:
                test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    if verbose:
        if classification:
            print(f"Avg accuracy: {correct_labels / tot_samples:.3f} \n")
        else:
            print(f"Avg test loss: {test_loss:>8f} \n")

    return test_loss


def pixels2coords(h: int, w: int):
    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()

    return coords
