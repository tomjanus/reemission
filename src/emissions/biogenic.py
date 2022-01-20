""" Module containing a dataclass with categorical descriptors playing part
    in the determination of the trophic status of the the reservoir """
from dataclasses import dataclass
from typing import Type, Dict
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

    @classmethod
    def fromdict(cls, data_dict: Dict):
        """ Initialize class from dictionary """
        return cls(biome=Biome[data_dict['biome']],
                   climate=Climate[data_dict['climate']],
                   soil_type=SoilType[data_dict['soil_type']],
                   treatment_factor=
                   TreatmentFactor[data_dict['treatment_factor']],
                   landuse_intensity=
                   LanduseIntensity[data_dict['landuse_intensity']])
