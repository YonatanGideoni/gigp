import os.path

from torch import nn
from torch.utils.data import DataLoader

from datasets.rotmnist import RotatedMNIST
from utils import TrainConfig


# train=train or test set, no built-in validation set
def get_dataset(conf: TrainConfig, train: bool = True, root=os.path.join('..', 'datasets', 'data')) -> DataLoader:
    ds = RotatedMNIST(root=root, partition="train" if train else "validation", augment="None")

    return DataLoader(ds, batch_size=conf.bs, shuffle=True)


if __name__ == '__main__':
    config = TrainConfig(n_epochs=100, lr=1e-2, bs=10, loss=nn.MSELoss())
    get_dataset(config)
