import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

from architectures.gigp import ImgGIGP
from consts import N_DIGITS
from groups import Group

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


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


# pytorch implementation of https://arxiv.org/pdf/1602.07576.pdf,
# taken from https://github.com/adambielski/pytorch-gconv-experiments/blob/master/mnist/mnist.py
# requires groupy folder, imported from https://github.com/adambielski/GrouPy
class GConvNet(nn.Module):
    def __init__(self):
        super(GConvNet, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(4 * 4 * 20 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
