import os.path
from dataclasses import asdict

import wandb
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from architectures.baselines import NormalCNN
from architectures.gigp import ImgGIGP
from consts import N_DIGITS, PROJ_NAME
from datasets.rotmnist import RotatedMNIST
from groups import SO2
from utils import TrainConfig, train_loop, test_loop, pixels2coords


def get_dataset(conf: TrainConfig, train: bool = True, root=os.path.join('../..', 'datasets', 'data')) -> DataLoader:
    # train=train or test set, no built-in validation set
    ds = RotatedMNIST(root=root, partition="train" if train else "validation", augment="None")

    return DataLoader(ds, batch_size=conf.bs, shuffle=True)


def train_model(model: nn.Module, conf: TrainConfig):
    wandb.login()

    run = wandb.init(
        project=PROJ_NAME,
        enity=ENT_NAME,
        name=conf.run_name,
        config=asdict(conf))

    train_data = get_dataset(conf)
    test_data = get_dataset(conf, train=False)

    optimiser = Adam(model.parameters(), lr=conf.lr)
    for epoch in range(conf.n_epochs):
        train_loss = train_loop(train_data, model, conf.loss, optimiser)
        test_loss, test_acc = test_loop(test_data, model, conf.loss, verbose=True, classification=True)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss, 'test_acc': test_acc})

    wand.finish()

    print('Final accuracy on train and test sets')
    print('Train:')
    test_loop(train_data, model, conf.loss, verbose=True, classification=True)
    print('Test:')
    test_loop(test_data, model, conf.loss, verbose=True, classification=True)


def naive_gigp(conf: TrainConfig):
    # try and predict digits using only a GIGP layer
    coords = pixels2coords(h=28, w=28)
    model = ImgGIGP(group=SO2(), coords=coords, in_dim=1, out_dim=N_DIGITS)

    train_model(model, conf)


def normal_cnn(conf: TrainConfig, gigp: bool = False):
    coords = pixels2coords(h=12, w=12)
    model = NormalCNN(group=SO2(), coords=coords, use_gigp=gigp)

    train_model(model, conf)


def main():
    # config = TrainConfig(n_epochs=100, lr=1e-4, bs=64, loss=nn.CrossEntropyLoss())
    # naive_gigp(config)

    config = TrainConfig(n_epochs=30, lr=1e-4, bs=64, loss=nn.CrossEntropyLoss())
    normal_cnn(config, gigp=True)


if __name__ == '__main__':
    main()
