import os.path

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from architectures.gigp import ImgGIGP
from datasets.rotmnist import RotatedMNIST
from groups import SO2
from utils import TrainConfig, train_loop, test_loop, pixels2coords


def get_dataset(conf: TrainConfig, train: bool = True, root=os.path.join('..', 'datasets', 'data')) -> DataLoader:
    # train=train or test set, no built-in validation set
    ds = RotatedMNIST(root=root, partition="train" if train else "validation", augment="None")

    return DataLoader(ds, batch_size=conf.bs, shuffle=True)


def naive_gigp(conf: TrainConfig, n_digits: int = 10):
    # try and predict digits using only a GIGP layer
    train_data = get_dataset(conf)
    test_data = get_dataset(conf, train=False)

    example_imgs = next(iter(train_data))[0]
    coords = pixels2coords(example_imgs)
    network = ImgGIGP(group=SO2(), coords=coords, in_dim=1, out_dim=n_digits)

    optimiser = Adam(network.parameters(), lr=conf.lr)
    for epoch in range(config.n_epochs):
        train_loop(train_data, network, config.loss, optimiser)
        test_loop(test_data, network, config.loss, verbose=True, classification=True)

    print('Final accuracy on train and test sets')
    print('Train:')
    test_loop(train_data, network, config.loss, verbose=True, classification=True)
    print('Test:')
    test_loop(test_data, network, config.loss, verbose=True, classification=True)


if __name__ == '__main__':
    config = TrainConfig(n_epochs=100, lr=1e-4, bs=64, loss=nn.CrossEntropyLoss())
    naive_gigp(config)
