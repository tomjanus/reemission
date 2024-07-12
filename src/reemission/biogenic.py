"""
This module provides a data structure to represent the biogenic factors that impact the trophic status of a reservoir.

Biogenic factors are composed of: biome, climate type, soil type, treatment factor, i.e. the type of wastewater treatment
in the catchment, and land use intensity.

Classes:
    BiogenicFactors: A class to represent and manage the properties of a catchment area that influence the trophic status of a reservoir.

Usage Example:

.. code-block:: Python

    from reemission.constants import (
        Biome, Climate, SoilType, TreatmentFactor, 
        LanduseIntensity)
    from reemission.descriptors import BiogenicFactors

    factors = BiogenicFactors(
        biome=Biome.DESERTS,
        climate=Climate.TROPICAL,
        soil_type=SoilType.ORGANIC,
        treatment_factor=TreatmentFactor.TERTIARY,
        landuse_intensity=LanduseIntensity.HIGH
    )
    print(factors)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Any, List
from reemission.constants import (
    Biome, Climate, SoilType, TreatmentFactor, LanduseIntensity)
from reemission.exceptions import ConversionMethodUnknownException


ToDictMethod = Literal["name", "value"]


@dataclass
class BiogenicFactors:
    """
    Catchment's properties impacting the reservoir's trophic status.

    Attributes:
        biome (Biome): The biome type of the catchment area.
        climate (Climate): The climate type of the catchment area.
        soil_type (SoilType): The soil type of the catchment area. Defaults to SoilType.MINERAL.
        treatment_factor (TreatmentFactor): The wastewater treatment factor. Defaults to TreatmentFactor.NONE.
        landuse_intensity (LanduseIntensity): The land use intensity. Defaults to LanduseIntensity.LOW.
    
    Note:

        Move optional attributes to a config file
    """
    biome: Biome
    climate: Climate
    soil_type: SoilType = field(default=SoilType.MINERAL)
    treatment_factor: TreatmentFactor = field(default=TreatmentFactor.NONE)
    landuse_intensity: LanduseIntensity = field(default=LanduseIntensity.LOW)

    @classmethod
    def fromdict(cls, data_dict: Dict, method: ToDictMethod = "name") -> BiogenicFactors:
        """
        Initialize class from a dictionary.

        Args:
            data_dict (Dict): Dictionary containing the data to initialize the class.
            method (ToDictMethod): Method to convert dictionary values. Either "name" or "value". Defaults to "name".

        Returns:
            BiogenicFactors: An instance of BiogenicFactors.

        Raises:
            ConversionMethodUnknownException: If the method provided is not recognized.
        """
        fields_and_enums = [
            ('biome', Biome), ('climate', Climate), ('soil_type', SoilType), 
            ('treatment_factor', TreatmentFactor), 
            ('landuse_intensity', LanduseIntensity)]
        
        def _instantiate_from_keys(cls) -> BiogenicFactors:
            """
            Instantiate the class using enum names from the dictionary.

            Returns:
                BiogenicFactors: An instance of BiogenicFactors.
            """
            input_dict = {}
            for key, enum_class in fields_and_enums:
                try:
                    value = data_dict[key]
                    input_dict.update({key: enum_class.from_key(value)})
                except KeyError:
                    pass
            return cls(**input_dict)

        def _instantiate_from_values(cls) -> BiogenicFactors:
            """
            Instantiate the class using enum values from the dictionary.

            Returns:
                BiogenicFactors: An instance of BiogenicFactors.
            """
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
        """
        Get a list of non-callable attributes of the class that do not start with '__'.

        Returns:
            List[str]: A list of attribute names.
        """
        return [
            attr for attr in dir(self) if not callable(getattr(self, attr)) and 
            not attr.startswith("__")]

    def todict(self, method: ToDictMethod = "name") -> Dict:
        """
        Convert the class to its dictionary representation.

        Args:
            method (ToDictMethod): Method to convert attributes. Either "name" or "value". Defaults to "name".

        Returns:
            Dict: A dictionary representation of the class.

        Raises:
            ConversionMethodUnknownException: If the method provided is not recognized.
        """
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

    def __repr__(self) -> str:
        """
        Return a string representation of the BiogenicFactors instance.

        Returns:
            str: String representation of the instance.
        """
        return f'Biogenic factors: {self.todict()}'
