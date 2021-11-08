""" Test module """
from dataclasses import dataclass, field
from typing import List, Dict
from inundation import Layer, Dem


@dataclass
class Collection:
    """ Base Collection Class """
    collection: List[Layer] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        """ Get layer names """
        return [layer.name for layer in self.collection]

    @property
    def paths(self) -> List[str]:
        """ Get layer paths """
        return [layer.path for layer in self.collection]

    def to_dict(self) -> Dict[str, str]:
        """ Get a dictionary with names as keys and paths as values """
        return {layer.name: layer.path for layer in self.collection}


class DemCollection(Collection):
    """ Collection Class for Dem Layers """
    collection: List[Dem] = field(default_factory=list)

    def fill_and_breach(self):
        # TODO apply dask parallelization
        pass
