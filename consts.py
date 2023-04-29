import torch

PROJ_NAME = 'gigp'
ENT_NAME = 'ky-time'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DIGITS: int = 10
N_RMNIST_ORBS: int = 74
