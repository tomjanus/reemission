"""
This module defines the NSCatchmentCreator class, which is used to represent 
post-impoundment catchment properties. The primary functionality provided by 
this class is to modify the catchment area and landuse fractions to exclude 
the area occupied by a reservoir.

Classes:
    **NSCatchmentCreator:** A class that modifies catchment properties by 
    subtracting the reservoir area and adjusting landuse fractions accordingly.

Usage Example:

.. code-block:: Python

    from .catchment import Catchment
    from .reservoir import Reservoir
    
    catchment_area_fractions = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.01092, 0.11996, 0.867257, 0.0]
    reservoir_area_fractions = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.15, 0.4, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0]

    catchment = Catchment(
        area=100, 
        area_fractions=catchment_area_fractions)
    reservoir = Reservoir(
        area=10, 
        area_fractions=reservoir_area_fractions)

    ns_creator = NSCatchmentCreator(catchment=catchment, reservoir=reservoir)
    modified_catchment = ns_creator.get_catchment()
"""
from typing import List
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from reemission.catchment import Catchment
from reemission.reservoir import Reservoir


@dataclass
class NSCatchmentCreator:
    """Representation of post-impoundment (catchment - reservoir) catchment
    properties. It is assumed that all original (whole) catchment properties
    remain unchanged except for landuse fractions and area.
    
    Attributes:
        catchment (Catchment): The original catchment object.
        reservoir (Reservoir): The reservoir object within the catchment.
    """
    catchment: Catchment
    reservoir: Reservoir

    @property
    def ns_catchment_area(self) -> float:
        """Calculates the area of the catchment excluding the reservoir.

        Returns:
            float: The area of the catchment minus the area of the reservoir.
        """
        return self.catchment.area - self.reservoir.area


    def _ns_landuse_fractions(self) -> List[float]:
        """Calculates landuse fractions in the catchment excluding the reservoir.

        Returns:
            List[float]: The landuse fractions for the catchment minus the reservoir.
        """
        def _r_to_c_area_fractions(r_landcover_fractions: np.array) -> np.array:
            """Reshapes the reservoir landcover fraction vector to the size
            of the catchment landcover fraction vector.

            Args:
                r_landcover_fractions (np.array): The reservoir landcover fractions.

            Returns:
                np.array: Reshaped landcover fractions.
            """
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
        """Subtracts the reservoir from the catchment and returns a new catchment object 
        with the altered properties representing post-impoundment conditions, i.e., 
        original catchment minus reservoir.

        Returns:
            Catchment: The modified catchment object.
        """
        catchment = deepcopy(self.catchment)
        catchment.area_fractions = self._ns_landuse_fractions()
        catchment.area = self.ns_catchment_area
        # TODO: How to deal with population? Currently assumed that the population
        # stays the same because either there was no population within the reservoir contour
        # (no relocation) or some people get relocated to new places within the catchment.
        return catchment
    

if __name__ == "__main__":
    """ """
        
