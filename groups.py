from abc import ABC, abstractmethod

import torch


# TODO - write input/output shapes
class Group(ABC):
    @abstractmethod
    def coords2orbit(self, coords):
        """Input - coordinates [?], output - orbit each coord belongs to [?] """
        pass


class SO2(Group):
    def coords2orbit(self, coords):
        return torch.sqrt((coords ** 2).sum(dim=-1))
