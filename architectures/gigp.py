import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SumAggregation

from groups import Group


class GIGP(nn.Module):
    def __init__(self, group: Group, coords: np.ndarray, in_dim: int, orbs_agg_dist: float = 0,
                 hidden_dim: int = 16, out_dim: int = 1):
        super(GIGP, self).__init__()

        self.orb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, out_dim))
        coords_orbs = group.coords2orbit(coords)
        unique_orbs = torch.unique(coords_orbs)
        # index i,j,k=distance of coord[i,j]'s orbit from orbit k
        dist_from_orb = abs(coords_orbs.unsqueeze(-1) - unique_orbs)
        orb_agg_mask = dist_from_orb <= orbs_agg_dist

        # gets index of the orbit each point in the domain is mapped to
        self.agg_orbs_inds = orb_agg_mask.flatten(end_dim=1).nonzero()[:, 1]
        self.agg = SumAggregation()
        self.orbits = unique_orbs

    def forward(self, x):
        agg_orbs = self.agg(x, index=self.agg_orbs_inds, dim=-1)
        transf_orbs = self.orb_mlp(agg_orbs.permute(0, 2, 1))

        return transf_orbs.sum(dim=-2)


class ImgGIGP(GIGP):
    def forward(self, x):
        x = x.flatten(start_dim=-2)

        return super().forward(x)
