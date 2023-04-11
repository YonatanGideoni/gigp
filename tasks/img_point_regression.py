import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from consts import DEVICE
from utils import TrainConfig


def create_dataset(img_len: int, n_imgs: int, config: TrainConfig) -> DataLoader:
    imgs = np.zeros((n_imgs, img_len, img_len), dtype=float)
    coords_x, coords_y = np.random.randint(0, img_len, n_imgs), np.random.randint(0, img_len, n_imgs)

    imgs[:, coords_y, coords_x] = 1

    labels = np.sqrt(coords_x ** 2 + coords_y ** 2)

    X = torch.Tensor(imgs).to(DEVICE)
    X = torch.flatten(X, start_dim=1, end_dim=-1)
    y = torch.tensor(labels)

    dataset = TensorDataset(X, y)

    data_loader = DataLoader(dataset=dataset, batch_size=config.bs)

    return data_loader


def main():
    config = TrainConfig(n_epochs=100, lr=1e-3, bs=10, loss=nn.MSELoss())
    create_dataset(20, 10 ** 3, config)


if __name__ == '__main__':
    main()
