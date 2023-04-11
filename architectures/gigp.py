from torch import nn

from groups import Group


class GIGP(nn.Module):
    def __init__(self, group: Group, domain2coords: callable, orbs_agg_dist: float, in_dim: int, hidden_dim: int = 16,
                 out_dim: int = 1):
        super(GIGP, self).__init__()

        self.group = group
        self.domain2coords = domain2coords
        self.orbs_agg_dist = orbs_agg_dist

        self.orb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, out_dim))

    # TODO - specify shapes of everything
    def forward(self, x):
        coords = self.domain2coords(x)
        orbs = self.group.coords_to_orbit(coords)
        orbs_dist = self.group.orbits_dist(orbs)

        # TODO:
        #  1. get list of all orbits
        #  2. aggregate all points that are within orbs_agg_dist from a given orbit
        #  3. run through MLP
        #  4. sum the results

        raise NotImplementedError
