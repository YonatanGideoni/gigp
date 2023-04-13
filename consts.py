import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DIGITS: int = 10
