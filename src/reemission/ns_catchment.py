""" """
from typing import List
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .catchment import Catchment
from .reservoir import Reservoir


@dataclass
class NSCatchmentCreator:
    """Representation of post-impoundment (catchment - reservoir) catchment
    properties. It is assumed that all original (whole) catchment properties
    remain unchanged except for landuse fractions and area."""
    catchment: Catchment
    reservoir: Reservoir

    @property
    def ns_catchment_area(self) -> float:
        return self.catchment.area - self.reservoir.area


    def _ns_landuse_fractions(self) -> List[float]:
        """Calculates landuse fractions in the catchment - reservoir"""
        def _r_to_c_area_fractions(r_landcover_fractions: np.array) -> np.array:
            """Reshapes the reservoir landcover fraction vector to the size
            of the catchment landcover fraction vector"""
            num_landcovers_in_catchment = len(self.catchment.area_fractions)
            num_landcovers_in_reservoir = len(self.reservoir.area_fractions)
            den = int(num_landcovers_in_reservoir/num_landcovers_in_catchment)
            return r_landcover_fractions.reshape(
                den,num_landcovers_in_catchment).sum(axis=0)

        c_landcover_fractions = np.array(self.catchment.area_fractions)
        r_landcover_fractions = _r_to_c_area_fractions(
            np.array(self.reservoir.area_fractions))
        c_landcover_areas = c_landcover_fractions * self.catchment.area
        r_landcover_areas = r_landcover_fractions * self.reservoir.area
            
        return (c_landcover_areas - r_landcover_areas) / self.ns_catchment_area


    def get_catchment(self) -> Catchment:
        """Subtracts reservoir from catchment return catchment object with
        the altered properties representing post-impoundment conditions, i.e.
        original catchment minus reservoir"""
        catchment = deepcopy(self.catchment)
        catchment.area_fractions = self._ns_landuse_fractions()
        catchment.area = self.ns_catchment_area
        # TODO: How to deal with population? Currently assumed that the population
        # stays the same because either there was no population within the reservoir contour
        # (no relocation) or some people get relocated to new places within the catchment.
        return catchment
    

if __name__ == "__main__":
    """ """
        
