import torch

from architectures.LieConv.lie_conv.lieGroups import SO2
from architectures.baselines import LieConvGIGP
from consts import DEVICE


def get_lifted_data(img):
    group = SO2(.2)

    bs, c, h, w = img.shape
    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float()
    # Perform center crop
    center_mask = coords.norm(dim=-1) < 15.  # crop out corners (filled only with zeros)
    coords = coords[center_mask].view(-1, 2).unsqueeze(0).repeat(bs, 1, 1).to(DEVICE)

    values = img.permute(0, 2, 3, 1)[:, center_mask, :].reshape(bs, -1, c)
    mask = torch.ones(bs, values.shape[1], device=DEVICE) > 0  # all true

    z = (coords, values, mask)
    with torch.no_grad():
        return group.lift(z, 1)


def test_inv():
    img = torch.rand((1, 3, 28, 28))

    gigp = LieConvGIGP(in_dim=3, out_dim=1)

    norm_data = get_lifted_data(img)
    norm_res = gigp(norm_data)
    for _ in range(3):
        rot_img = img.permute(0, 1, 3, 2).flip(3)
        rot_data = get_lifted_data(rot_img)
        rot_res = gigp(rot_data)

        assert (norm_res == rot_res).all(), 'Error - GIGP is not rotation invariant!'


if __name__ == '__main__':
    test_inv()
