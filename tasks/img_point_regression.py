import numpy as np
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import TensorDataset, DataLoader

from architectures.gigp import ImgGIGP
from consts import DEVICE
from groups import SO2
from utils import TrainConfig, train_loop, test_loop, pixels2coords


def create_dataset(img_len: int, n_imgs: int, config: TrainConfig) -> DataLoader:
    imgs = np.zeros((n_imgs, img_len, img_len), dtype=float)
    coords_x, coords_y = np.random.randint(0, img_len, n_imgs), np.random.randint(0, img_len, n_imgs)

    imgs[np.arange(n_imgs), coords_y, coords_x] = 1

    labels = np.sqrt((coords_x - img_len / 2) ** 2 + (coords_y - img_len / 2) ** 2)

    X = torch.Tensor(imgs).to(DEVICE).unsqueeze(1)
    y = torch.tensor(labels).float().unsqueeze(1)

    dataset = TensorDataset(X, y)

    data_loader = DataLoader(dataset=dataset, batch_size=config.bs)

    return data_loader


def main(train_conf: TrainConfig, img_size: int = 20,
         train_size: int = 10 ** 3, test_size: int = 10 ** 2):
    train_data = create_dataset(img_size, train_size, train_conf)
    test_data = create_dataset(img_size, test_size, train_conf)

    example_imgs = next(iter(train_data))[0]
    coords = pixels2coords(example_imgs)
    network = ImgGIGP(group=SO2(), coords=coords, in_dim=1)

    optimiser = Adam(network.parameters(), lr=train_conf.lr)
    for epoch in range(config.n_epochs):
        train_loop(train_data, network, config.loss, optimiser)
        test_loop(test_data, network, config.loss, verbose=True)


if __name__ == '__main__':
    config = TrainConfig(n_epochs=100, lr=1e-2, bs=10, loss=nn.MSELoss())
    main(config)
