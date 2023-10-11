"""Categorical descriptors for determination of the trophic status of the the
reservoir."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Any, List
from reemission.constants import (
    Biome, Climate, SoilType, TreatmentFactor, LanduseIntensity)
from reemission.exceptions import ConversionMethodUnknownException


ToDictMethod = Literal["name", "value"]


@dataclass
class BiogenicFactors:
    """Catchment's properties impacting the reservoir's trophic status."""
    biome: Biome
    climate: Climate
    # TODO: Move these optional fields to a config file
    soil_type: SoilType = field(default=SoilType.MINERAL)
    treatment_factor: TreatmentFactor = field(default=TreatmentFactor.NONE)
    landuse_intensity: LanduseIntensity = field(default=LanduseIntensity.LOW)

    @classmethod
    def fromdict(cls, data_dict: Dict, method: ToDictMethod = "name") -> BiogenicFactors:
        """Initialize class from dictionary."""
        fields_and_enums = [
            ('biome', Biome), ('climate', Climate), ('soil_type', SoilType), 
            ('treatment_factor', TreatmentFactor), 
            ('landuse_intensity', LanduseIntensity)]
        
        def _instantiate_from_keys(cls) -> BiogenicFactors:
            """ """
            input_dict = {}
            for key, enum_class in fields_and_enums:
                try:
                    value = data_dict[key]
                    input_dict.update({key: enum_class.from_key(value)})
                except KeyError:
                    pass
            return cls(**input_dict)

        def _instantiate_from_values(cls) -> BiogenicFactors:
            """ """
            input_dict = {}
            for key, enum_class in fields_and_enums:
                try:
                    value = data_dict[key]
                    input_dict.update({key: enum_class.from_value(value)})
                except KeyError:
                    pass
            return cls(**input_dict)

        if method == "name":
            return _instantiate_from_keys(cls)
        elif method == "value":
            return _instantiate_from_values(cls)
        else:
            raise ConversionMethodUnknownException(
                conversion_method=method,
                available_methods=ToDictMethod.__args__)
        
    def get_attributes(self) -> List[str]:
        return [
            attr for attr in dir(self) if not callable(getattr(self, attr)) and 
            not attr.startswith("__")]

    def todict(self, method: ToDictMethod = "name") -> Dict:
        """Convert the class to its dictionary representation"""
        biogenic_factors = {}
        if method == "name":
            for attribute_name in self.get_attributes():
                biogenic_factors.update(
                    {attribute_name: getattr(attribute_name).name})
        elif method == "value":
            for attribute_name in self.get_attributes():
                biogenic_factors.update(
                    {attribute_name: getattr(attribute_name).value})
        else:
            raise ConversionMethodUnknownException(
                conversion_method=method,
                available_methods=ToDictMethod.__args__)
        return biogenic_factors

    def __repr__(self):
        return f'Biogenic factors: {self.todict()}'
