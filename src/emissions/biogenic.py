""" Module containing a dataclass with categorical descriptors playing part
    in the determination of the trophic status of the the reservoir """
from dataclasses import dataclass
from typing import Type
from .constants import (Biome, Climate, SoilType, TreatmentFactor,
                        LanduseIntensity)


@dataclass
class BiogenicFactors:
    """ Container class for parameters characterising catchment's properties
        having influence on the reservoir's trophic status """
    biome: Type[Biome]
    climate: Type[Climate]
    soil_type: Type[SoilType]
    treatment_factor: Type[TreatmentFactor]
    landuse_intensity: Type[LanduseIntensity]
