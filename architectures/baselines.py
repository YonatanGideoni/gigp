import torch.nn.functional as F
import torch
from torch import nn


# TODO - use/create a baseline that isn't total garbage

# taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class BaselineConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(400, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)
