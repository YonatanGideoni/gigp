import numpy as np
import torch

from architectures.gigp import ImgGIGP
from groups import SO2
from tasks.img_point_regression import pixels2coords


def test_invariance():
    img = torch.tensor(np.random.rand(1, 1, 20, 20)).float()

    coords = pixels2coords(img)

    network = ImgGIGP(group=SO2(), coords=coords, in_dim=1)

    norm_out = network(img)
    rot_img = img.permute(0, 1, 3, 2).flip(3)
    for _ in range(3):
        rot_out = network(rot_img)

        assert torch.isclose(rot_out, norm_out).all(), "Error - GIGP isn't invariant!"

        rot_img = rot_img.permute(0, 1, 3, 2).flip(3)


if __name__ == '__main__':
    test_invariance()
