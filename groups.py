from abc import ABC, abstractmethod


# TODO - write input/output shapes
class Group(ABC):
    @abstractmethod
    def orbits_dist(self, orbits):
        """Input - orbits [?], output - distances between pairs of orbits [?] """
        pass

    @abstractmethod
    def coords_to_orbit(self, coords):
        """Input - coordinates [?], output - orbit each coord belongs to [?] """
        pass
