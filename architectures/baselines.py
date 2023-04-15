import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

from architectures.LieConv.lie_conv.lieConv import BottleBlock, Swish, GlobalPool, LieConv
from architectures.LieConv.lie_conv.masked_batchnorm import MaskBatchNormNd
from architectures.LieConv.lie_conv.utils import Named, Pass, Expression
from architectures.gigp import ImgGIGP
from consts import N_DIGITS
from groups import Group, SO2


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


# copies ImgLieResnet with GIGP optionally appended to it instead of global maxpooling
class LieResNet(nn.Module, metaclass=Named):
    """ Generic LieConv architecture from Fig 5. Relevant Arguments:
        [Fill] specifies the fraction of the input which is included in local neighborhood.
                (can be array to specify a different value for each layer)
        [nbhd] number of samples to use for Monte Carlo estimation (p)
        [chin] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
        [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
        [num_layers] number of BottleNeck Block layers in the network
        [k] channel width for the network. Can be int (same for all) or array to specify individually.
        [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
        [Group] Chosen group to be equivariant to.
        [bn] whether or not to use batch normalization. Recommended in all cases except dynamical systems.
        """

    def __init__(self, chin, ds_frac=1, num_outputs=1, k=1536, nbhd=np.inf,
                 act="swish", bn=True, num_layers=6, mean=True, per_point=True, pool=True,
                 liftsamples=1, fill=1 / 4, group=SO2(.05), knn=False, cache=False, gigp: bool = False, **kwargs):
        super().__init__()
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(ki, ko, mc_samples=nbhd, ds_frac=ds_frac, bn=bn, act=act, mean=mean,
                                            group=group, fill=fill, cache=cache, knn=knn, **kwargs)
        # TODO - implement GIGP
        pooling = (GlobalPool(mean=mean) if pool else Expression(lambda x: x[1])) if not gigp else 0
        self.net = nn.Sequential(
            Pass(nn.Linear(chin, k[0]), dim=1),  # embedding layer
            *[BottleBlock(k[i], k[i + 1], conv, bn=bn, act=act, fill=fill[i]) for i in range(num_layers)],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish() if act == 'swish' else nn.ReLU(), dim=1),
            Pass(nn.Linear(k[-1], num_outputs), dim=1),
            pooling,
        )
        self.liftsamples = liftsamples
        self.per_point = per_point
        self.group = group

    def forward(self, x):
        lifted_x = self.group.lift(x, self.liftsamples)
        return self.net(lifted_x)


class GIGPImgLieResnet(LieResNet):
    """ Lie Conv architecture specialized to images. Uses scaling rule to determine channel
         and downsampling scaling. Same arguments as LieResNet"""

    def __init__(self, chin=1, total_ds=1 / 64, num_layers=6, group=T(2), fill=1 / 32, k=256,
                 knn=False, nbhd=12, num_targets=10, increase_channels=True, **kwargs):
        ds_frac = (total_ds) ** (1 / num_layers)
        fill = [fill / ds_frac ** i for i in range(num_layers)]
        if increase_channels:  # whether or not to scale the channels as image is downsampled
            k = [int(k / ds_frac ** (i / 2)) for i in range(num_layers + 1)]
        super().__init__(chin=chin, ds_frac=ds_frac, num_layers=num_layers, nbhd=nbhd, mean=True,
                         group=group, fill=fill, k=k, num_outputs=num_targets, cache=True, knn=knn, **kwargs)
        self.lifted_coords = None

    def forward(self, x, coord_transform=None):
        """ assumes x is a regular image: (bs,c,h,w)"""
        bs, c, h, w = x.shape
        # Construct coordinate grid
        i = torch.linspace(-h / 2., h / 2., h)
        j = torch.linspace(-w / 2., w / 2., w)
        coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()
        # Perform center crop
        center_mask = coords.norm(dim=-1) < 15.  # crop out corners (filled only with zeros)
        coords = coords[center_mask].view(-1, 2).unsqueeze(0).repeat(bs, 1, 1).to(x.device)
        if coord_transform is not None: coords = coord_transform(coords)
        values = x.permute(0, 2, 3, 1)[:, center_mask, :].reshape(bs, -1, c)
        mask = torch.ones(bs, values.shape[1], device=x.device) > 0  # all true
        z = (coords, values, mask)
        # Perform lifting of the coordinates and cache results
        with torch.no_grad():
            if self.lifted_coords is None:
                self.lifted_coords, lifted_vals, lifted_mask = self.group.lift(z, self.liftsamples)
            else:
                lifted_vals, lifted_mask = self.group.expand_like(values, mask, self.lifted_coords)
        return self.net((self.lifted_coords, lifted_vals, lifted_mask))
