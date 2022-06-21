"""Categorical descriptors for determination of the trophic status of the the
reservoir."""
from dataclasses import dataclass
from typing import Dict
from reemission.constants import (
    Biome, Climate, SoilType, TreatmentFactor, LanduseIntensity)


@dataclass
class BiogenicFactors:
    """Catchment's properties impacting the reservoir's trophic status."""

    biome: Biome
    climate: Climate
    soil_type: SoilType
    treatment_factor: TreatmentFactor
    landuse_intensity: LanduseIntensity

    @classmethod
    def fromdict(cls, data_dict: Dict):
        """Initialize class from dictionary."""
        return cls(
            biome=Biome[data_dict['biome']],
            climate=Climate[data_dict['climate']],
            soil_type=SoilType[data_dict['soil_type']],
            treatment_factor=TreatmentFactor[data_dict['treatment_factor']],
            landuse_intensity=LanduseIntensity[data_dict['landuse_intensity']],
        )

    def todict(self) -> Dict:
        """Convert the class to its dictionary representation"""
        biogenic_factors = {}
        biogenic_factors['biome'] = self.biome.name
        biogenic_factors['climate'] = self.climate.name
        biogenic_factors['soil_type'] = self.soil_type.name
        biogenic_factors['treatment_factor'] = self.treatment_factor.name
        biogenic_factors['landuse_intensity'] = self.landuse_intensity.name
        return biogenic_factors

    def __repr__(self):
        return f'Biogenic factors: {self.todict()}'
