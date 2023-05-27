import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from architectures.LieConv.lie_conv.lieConv import (
    BottleBlock,
    Swish,
    GlobalPool,
    LieConv,
)
from architectures.LieConv.lie_conv.lieGroups import SO2, SE3, SO3, LieGroup
from architectures.LieConv.lie_conv.masked_batchnorm import MaskBatchNormNd
from architectures.LieConv.lie_conv.datasets import SO3aug, SE3aug
from architectures.LieConv.lie_conv.utils import Named, Pass, Expression
from architectures.gigp import ImgGIGP
from consts import N_DIGITS, DEVICE
from groups import Group
from utils import init_sum_mlp


# TODO - use/create a baseline that isn't total garbage

# taken from https://github.com/pytorch/examples/tree/main/mnist
class NormalCNN(nn.Module):
    def __init__(
            self,
            use_gigp: bool = False,
            group: Group = None,
            coords: np.ndarray = None,
    ):
        super(NormalCNN, self).__init__()
        self.use_gigp = use_gigp

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)

        if use_gigp:
            self.gigp = ImgGIGP(
                group=group, coords=coords, in_dim=64, out_dim=N_DIGITS
            )
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


# TODO - create unit tests (eg. gives a consistent result when the mlp is set to the identity)
class LieConvGIGP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            orbs_agg_dist: float = None,  # all orbits within that distance are aggregated as one
            n_agg_elems: int = 1,
            hidden_dim: int = 16,
            out_dim: int = 1,
            agg: str = "sum",
            use_orbits_data: bool = False,  # ??
            n_orbs: int = 50,  # Number of orbits
            group: LieGroup = SO2(0),
            init_glob_pool: bool = True,
            init_glob_pool_mean: bool = True,
            init_std: float = 1e-6,
    ):
        super().__init__()

        assert n_agg_elems is not None or orbs_agg_dist is not None

        # all trainable layers need to have GIGP in their name so we'll know not to freeze them if freezing
        # the base LC layers. This is useful for quickly testing things when we train just the GIGP layers appended
        # to a pretrained model
        self.gigp_orb_mlp = nn.Sequential(
            nn.Linear(in_dim + use_orbits_data, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.dist_func = group.distance

        self.use_orbs_data = use_orbits_data
        self.orbs_agg_dist = orbs_agg_dist
        self.n_agg_elems = n_agg_elems

        self.orbs = None
        self.n_orbs = n_orbs

        if init_glob_pool:
            assert agg == 'weighted_sum', 'Error - implemented only for this aggregation right now!'

        self.init_glob_pool_mean = init_glob_pool and init_glob_pool_mean
        if agg == "sum":
            self.orbs_aggregator = lambda x: x.sum(dim=1)
        elif agg == "mean":
            self.orbs_aggregator = lambda x: x.mean(dim=1)
        elif agg == "weighted_sum":
            self.init_lin_layer = True
            self.gigp_lin_layer = nn.Linear(n_orbs, 1, device=DEVICE, bias=False)
            self.orbs_aggregator = lambda x: self.gigp_lin_layer(
                x.permute(0, 2, 1)
            )[:, :, 0]

            if init_glob_pool:
                assert init_glob_pool_mean, \
                    "Error - haven't yet implemented anything for when the initialisation isn't mean pooling!"
                assert out_dim == 1, 'Error - currently only works when the outdim is 1!'

                nn.init.constant_(self.gigp_lin_layer.weight, 1)
                init_sum_mlp(self.gigp_orb_mlp, n_inps_sum=in_dim, std=init_std)
        else:
            raise NotImplementedError(
                f"Haven't implemented {agg} aggregation yet for LieConvGIGP!"
            )

    # TODO - make sure you're summing the correct orbits
    def forward(self, x):
        """x [xyz (bs,n,d), vals (bs,n,c), mask (bs,n)]"""
        if len(x) == 2:
            return x[1].mean(1)
        coords, vals, mask = x
        masked_vals = torch.where(mask.unsqueeze(-1), vals, 0.0)

        bs = coords.shape[0]

        if self.orbs is None:
            # if never seen orbits
            # TODO: is below correct? Should it not be :::1?
            min_orb = torch.min(coords[:, :, 1, 1])
            max_orb = torch.max(coords[:, :, 1, 1])
            # have fixed number of aggregated orbits
            # match each orbit to the closest one
            self.orbs = torch.linspace(min_orb, max_orb, self.n_orbs).to(DEVICE)
            # TODO - check that agg_orbs_dist isn't too small and give a warning if it's fishy

        orbs_dists = abs(coords[:, :, 1, 1].unsqueeze(-1) - self.orbs.expand(bs, coords.shape[1], self.n_orbs))
        # orbs_mask shape: [bs, coords.shape[1], n_orbs]
        if self.orbs_agg_dist is not None:
            orbs_mask = orbs_dists <= self.orbs_agg_dist
        else:
            _, min_inds = torch.topk(orbs_dists, self.n_agg_elems, largest=False, dim=-1)
            orbs_mask = torch.zeros_like(orbs_dists, dtype=torch.bool)
            orbs_mask.scatter_(dim=-1, index=min_inds, value=True)

        exp_vals = masked_vals.unsqueeze(-2)
        masked_orbs = torch.where(orbs_mask.unsqueeze(-1), exp_vals, 0.0)

        if self.use_orbs_data:
            masked_orbs = torch.cat(
                [
                    masked_orbs,
                    self.orbs.expand(bs, masked_orbs.shape[1], -1)[
                    :, :, :, None
                    ],
                ],
                dim=-1,
            )

        orbs_repr = masked_orbs.sum(dim=1)

        empty_orbs_mask = orbs_repr.sum(dim=-1) == 0
        transf_orbs = self.gigp_orb_mlp(orbs_repr)
        masked_transf_orbs = torch.where(
            ~empty_orbs_mask.unsqueeze(-1), transf_orbs, 0.0
        )

        if self.init_glob_pool_mean:
            rel_mask = masked_orbs[:, :, :, :-1] if self.use_orbs_data else masked_orbs
            n_elems = (rel_mask != 0).sum(dim=[1, 2, 3])
            return self.orbs_aggregator(masked_transf_orbs) / n_elems.unsqueeze(1)

        return self.orbs_aggregator(masked_transf_orbs)


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

    def __init__(
            self,
            chin,
            ds_frac=1,
            num_outputs=1,
            k=1536,
            nbhd=np.inf,
            act="swish",
            bn=True,
            num_layers=6,
            mean=True,
            per_point=True,
            pool=True,
            liftsamples=1,
            fill=1 / 4,
            group=SO2(0.05),
            knn=False,
            cache=False,
            gigp: bool = False,
            use_orbits_data: bool = False,
            orbs_agg_dist: float = None,
            gigp_agg: str = "sum",
            n_orbs_agg: int = None,
            **kwargs,
    ):
        super().__init__()
        if isinstance(fill, (float, int)):
            fill = [fill] * num_layers
        if isinstance(k, int):
            k = [k] * (num_layers + 1)
        conv = lambda ki, ko, fill: LieConv(
            ki,
            ko,
            mc_samples=nbhd,
            ds_frac=ds_frac,
            bn=bn,
            act=act,
            mean=mean,
            group=group,
            fill=fill,
            cache=cache,
            knn=knn,
            **kwargs,
        )
        pooling = (
            (GlobalPool(mean=mean) if pool else Expression(lambda x: x[1]))
            if not gigp
            else LieConvGIGP(
                in_dim=num_outputs,
                out_dim=num_outputs,
                use_orbits_data=use_orbits_data,
                orbs_agg_dist=orbs_agg_dist,
                n_agg_elems=n_orbs_agg,
                agg=gigp_agg,
                group=group,
            )
        )
        self.net = nn.Sequential(
            Pass(nn.Linear(chin, k[0]), dim=1),  # embedding layer
            *[
                BottleBlock(k[i], k[i + 1], conv, bn=bn, act=act, fill=fill[i])
                for i in range(num_layers)
            ],
            MaskBatchNormNd(k[-1]) if bn else nn.Sequential(),
            Pass(Swish() if act == "swish" else nn.ReLU(), dim=1),
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

    def __init__(
            self,
            chin=1,
            total_ds=1 / 64,
            num_layers=6,
            group=SO2(),
            fill=1 / 32,
            k=256,
            knn=False,
            nbhd=12,
            num_targets=10,
            increase_channels=True,
            **kwargs,
    ):
        ds_frac = (total_ds) ** (1 / num_layers)
        fill = [fill / ds_frac ** i for i in range(num_layers)]
        if (
                increase_channels
        ):  # whether or not to scale the channels as image is downsampled
            k = [int(k / ds_frac ** (i / 2)) for i in range(num_layers + 1)]
        super().__init__(
            chin=chin,
            ds_frac=ds_frac,
            num_layers=num_layers,
            nbhd=nbhd,
            mean=True,
            group=group,
            fill=fill,
            k=k,
            num_outputs=num_targets,
            cache=True,
            knn=knn,
            **kwargs,
        )
        self.lifted_coords = None

    def forward(self, x, coord_transform=None):
        """ assumes x is a regular image: (bs,c,h,w)"""
        bs, c, h, w = x.shape
        # Construct coordinate grid
        i = torch.linspace(-h / 2.0, h / 2.0, h)
        j = torch.linspace(-w / 2.0, w / 2.0, w)
        coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()
        # Perform center crop
        center_mask = (
                coords.norm(dim=-1) < 15.0
        )  # crop out corners (filled only with zeros)
        coords = (
            coords[center_mask]
            .view(-1, 2)
            .unsqueeze(0)
            .repeat(bs, 1, 1)
            .to(x.device)
        )
        if coord_transform is not None:
            coords = coord_transform(coords)
        values = x.permute(0, 2, 3, 1)[:, center_mask, :].reshape(bs, -1, c)
        mask = torch.ones(bs, values.shape[1], device=x.device) > 0  # all true
        z = (coords, values, mask)
        # Perform lifting of the coordinates and cache results
        with torch.no_grad():
            if self.lifted_coords is None:
                self.lifted_coords, lifted_vals, lifted_mask = self.group.lift(
                    z, self.liftsamples
                )
            else:
                lifted_vals, lifted_mask = self.group.expand_like(
                    values, mask, self.lifted_coords
                )
        return self.net((self.lifted_coords, lifted_vals, lifted_mask))


class GIGPMolecLieResNet(LieResNet):
    def __init__(self, num_species, charge_scale, aug=False, group=SE3, **kwargs):
        super().__init__(chin=3 * num_species, num_outputs=1, group=group, ds_frac=1, **kwargs)
        self.charge_scale = charge_scale
        self.aug = aug
        self.random_rotate = SE3aug()  # RandomRotation()

    def featurize(self, mb):
        charges = mb['charges'] / self.charge_scale
        c_vec = torch.stack([torch.ones_like(charges), charges, charges ** 2], dim=-1)  #
        one_hot_charges = (mb['one_hot'][:, :, :, None] * c_vec[:, :, None, :]).float().reshape(*charges.shape, -1)
        atomic_coords = mb['positions'].float()
        atom_mask = mb['charges'] > 0

        new_coords = atomic_coords - (atomic_coords.sum(dim=1) / atom_mask.sum(dim=1).unsqueeze(1)).unsqueeze(1)

        return new_coords, one_hot_charges, atom_mask

    def forward(self, mb):
        with torch.no_grad():
            x = self.featurize(mb)
            x = self.random_rotate(x) if self.aug else x
        return super().forward(x).squeeze(-1)
