from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainConfig:
    run_name: str
    n_epochs: int
    lr: float
    bs: int
    loss: nn.Module


def train_loop(dataloader, model: nn, loss_fn: callable, optimizer: Optimizer):
    tot_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

    return tot_loss / (batch + 1)


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
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    accuracy = correct_labels / max(tot_samples, 1)

    if verbose:
        if classification:
            print(f"Avg accuracy: {correct_labels / tot_samples:.3f} \n")
        else:
            print(f"Avg test loss: {test_loss:>8f} \n")

    return test_loss, accuracy


def pixels2coords(h: int, w: int):
    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()

    return coords


# TODO - clean up+docstring
def init_sum_mlp(mlp, n_inps_sum: int = None, std: float = 0.):
    layers = list(mlp.children())
    # second cond is necessary so layers like batchnorm aren't cached
    layers = [l for l in layers if hasattr(l, 'weight') and len(l.weight.shape) == 2]

    layer = layers[0]
    new_weights = std * torch.randn_like(layer.weight)
    if n_inps_sum is None:
        new_weights[0, :] = 1
        new_weights[1, :] = -1
    else:
        new_weights[0, :n_inps_sum] = 1
        new_weights[1, :n_inps_sum] = -1

    layer.weight.requires_grad_(False)
    layer.weight.copy_(new_weights)
    nn.init.constant_(layer.bias, 0)
    layer.weight.requires_grad_(True)

    for layer in layers[1:-1]:
        new_weights = std * torch.randn_like(layer.weight)
        new_weights[0, 0] = 1
        new_weights[0, 1] = -1
        new_weights[1, 0] = -1
        new_weights[1, 1] = 1

        layer.weight.requires_grad_(False)
        layer.weight.copy_(new_weights)
        nn.init.constant_(layer.bias, 0)
        layer.weight.requires_grad_(True)

    layer = layers[-1]
    new_weights = std * torch.randn_like(layer.weight)
    new_weights[0, 0] = 1
    new_weights[0, 1] = -1
    layer.weight.requires_grad_(False)
    layer.weight.copy_(new_weights)
    nn.init.constant_(layer.bias, 0)
    layer.weight.requires_grad_(True)


def init_id_mlp(mlp: nn.Module, n_inps: int, std: float = 0.):
    layers = list(mlp.children())
    layers = [l for l in layers if hasattr(l, 'weight') and len(l.weight.shape) == 2]

    for n_layer, layer in enumerate(layers):
        new_weights = std * torch.randn_like(layer.weight)

        new_weights[:n_inps, :n_inps] += torch.eye(n_inps)
        if n_layer == 0:
            new_weights[n_inps:2 * n_inps, :n_inps] -= torch.eye(n_inps)
        elif n_layer == len(layers) - 1:
            new_weights[:n_inps, n_inps:2 * n_inps] -= torch.eye(n_inps)
        else:
            new_weights[n_inps:2 * n_inps, n_inps:2 * n_inps] += torch.eye(n_inps)

        layer.weight.requires_grad_(False)
        layer.weight.copy_(new_weights)
        nn.init.constant_(layer.bias, 0)
        layer.weight.requires_grad_(True)
