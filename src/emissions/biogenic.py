""" Module containing a dataclass with categorical descriptors playing part
    in the determination of the trophic status of the the reservoir """
from dataclasses import dataclass
from .constants import (Biome, Climate, SoilType, TreatmentFactor,
                        LanduseIntensity)


@dataclass
class BiogenicFactors:
    """ Container class for parameters characterising catchment's properties
        having influence on the reservoir's trophic status """
    biome: Biome
    climate: Climate
    soil_type: SoilType
    treatment_factor: TreatmentFactor
    landuse_intensity: LanduseIntensity
