import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

from architectures.gigp import ImgGIGP
from consts import N_DIGITS
from groups import Group


# TODO - use/create a baseline that isn't total garbage

# taken from https://github.com/pytorch/examples/tree/main/mnist
class NormalCNN(nn.Module):
    def __init__(self, use_gigp: bool = False, group: Group = None, coords: np.ndarray = None):
        super(NormalCNN, self).__init__()
        self.use_gigp = use_gigp

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

        if use_gigp:
            self.gigp = ImgGIGP(group=group, coords=coords, in_dim=64, out_dim=N_DIGITS)
        else:
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, N_DIGITS)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        if self.use_gigp:
            x = self.gigp(x)
        else:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)

        return x
