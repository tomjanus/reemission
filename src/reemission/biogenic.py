"""Categorical descriptors for determination of the trophic status of the the
reservoir."""
from dataclasses import dataclass
from typing import Dict, Literal
from reemission.constants import (
    Biome, Climate, SoilType, TreatmentFactor, LanduseIntensity)
from reemission.exceptions import ConversionMethodUnknownException


ToDictMethod = Literal["name", "value"]


@dataclass
class BiogenicFactors:
    """Catchment's properties impacting the reservoir's trophic status."""

    biome: Biome
    climate: Climate
    soil_type: SoilType
    treatment_factor: TreatmentFactor
    landuse_intensity: LanduseIntensity

    @classmethod
    def fromdict(cls, data_dict: Dict, method: ToDictMethod = "name"):
        """Initialize class from dictionary."""
        if method == "name":
            return cls(
                biome=Biome.from_key(data_dict['biome']),
                climate=Climate.from_key(data_dict['climate']),
                soil_type=SoilType.from_key(data_dict['soil_type']),
                treatment_factor=TreatmentFactor.from_key(
                    data_dict['treatment_factor']),
                landuse_intensity=LanduseIntensity.from_key(
                    data_dict['landuse_intensity']))
        elif method == "value":
            return cls(
                biome=Biome.from_value(data_dict['biome']),
                climate=Climate.from_value(data_dict['climate']),
                soil_type=SoilType.from_value(data_dict['soil_type']),
                treatment_factor=TreatmentFactor.from_value(
                    data_dict['treatment_factor']),
                landuse_intensity=LanduseIntensity.from_value(
                    data_dict['landuse_intensity']))
        else:
            raise ConversionMethodUnknownException(
                conversion_method=method,
                available_methods=ToDictMethod.__args__)

    def todict(self, method: ToDictMethod = "name") -> Dict:
        """Convert the class to its dictionary representation"""
        biogenic_factors = {}
        if method == "name":
            biogenic_factors['biome'] = self.biome.name
            biogenic_factors['climate'] = self.climate.name
            biogenic_factors['soil_type'] = self.soil_type.name
            biogenic_factors['treatment_factor'] = self.treatment_factor.name
            biogenic_factors['landuse_intensity'] = self.landuse_intensity.name
        elif method == "value":
            biogenic_factors['biome'] = self.biome.value
            biogenic_factors['climate'] = self.climate.value
            biogenic_factors['soil_type'] = self.soil_type.value
            biogenic_factors['treatment_factor'] = self.treatment_factor.value
            biogenic_factors['landuse_intensity'] = self.landuse_intensity.value
        else:
            raise ConversionMethodUnknownException(
                conversion_method=method,
                available_methods=ToDictMethod.__args__)
        return biogenic_factors

    def __repr__(self):
        return f'Biogenic factors: {self.todict()}'
