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
import importlib
from typing import Dict, Literal, ClassVar, Type, Any, Set, Protocol
from pydantic import BaseModel, ConfigDict, create_model, Field
from rich import print as rprint
from rich.repr import RichReprResult
from reemission.constants import (
    Biome, Climate, SoilType, TreatmentFactor, LanduseIntensity)
from reemission.exceptions import ConversionMethodUnknownException


ToDictMethod = Literal["name", "value"]


class DictLike(Protocol):
    """Protocol for objects that can be converted to a dictionary."""
    def to_dict(self) -> Dict[str, Any]: ...


class BaseBiogenicFactors(BaseModel):
    """Base model for all biogenic factors with field definitions."""
    
    # Define class variables to store metadata
    _all_fields: ClassVar[Dict[str, Type]] = {
        "biome": Biome,
        "climate": Climate,
        "soil_type": SoilType,
        "treatment_factor": TreatmentFactor,
        "landuse_intensity": LanduseIntensity,
    }
    
    _default_values: ClassVar[Dict[str, Any]] = {
        "soil_type": SoilType.MINERAL,
        "treatment_factor": TreatmentFactor.NONE,
        "landuse_intensity": LanduseIntensity.LOW,
    }

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )
    
    def __repr__(self):
        # Only show actual instance fields
        fields = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({fields})"

    def __str__(self):
        # Only show actual instance fields
        fields = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({fields})"
    
    def __rich_repr__(self) -> RichReprResult:
        # Only show actual fields, no model_config
        for k, v in self.__dict__.items():
            if k != "model_config" and not k.startswith("_"):
                yield k, v
       
    @classmethod
    def register_field(cls, name: str, field_type: Type, default_value: Any = None) -> None:
        """
        Register a new field type for use in BiogenicFactors models.
        
        Args:
            name: Field name
            field_type: Field type (usually an Enum class)
            default_value: Optional default value for the field
        """
        # Add to available fields
        cls._all_fields[name] = field_type
        
        # Add default value if provided
        if default_value is not None:
            cls._default_values[name] = default_value
            
    @classmethod
    def update_fields_from_dict(cls, fields_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Update field definitions from a dictionary.
        
        Args:
            fields_dict: Dictionary where keys are field names and values are dictionaries 
                        with 'type' and optional 'default' keys.
                        
        Example:
            {
                "new_field": {
                    "type": "reemission.constants.NewEnum", 
                    "default": "DEFAULT_VALUE"
                }
            }
        """
        for field_name, field_config in fields_dict.items():
            # Get type from string if needed
            field_type = field_config["type"]
            if isinstance(field_type, str):
                # Format: "module.submodule.ClassName"
                module_path, class_name = field_type.rsplit(".", 1)
                module = importlib.import_module(module_path)
                field_type = getattr(module, class_name)
            
            # Register the field
            default_value = field_config.get("default")
            if default_value is not None and isinstance(default_value, str):
                # Convert string default to enum value
                if hasattr(field_type, "from_key"):
                    default_value = field_type.from_key(default_value)
                else:
                    # Try to find the enum value by name
                    default_value = getattr(field_type, default_value)
                    
            cls.register_field(field_name, field_type, default_value)
    
    @classmethod
    def create_model(cls, fields: Set[str] = None) -> Type[BaseBiogenicFactors]:
        """
        Create a custom BiogenicFactors model with specified fields.
        
        Args:
            fields: Set of field names to include. If None, includes all fields.
            
        Returns:
            A new Pydantic model class with the specified fields
        
        Example:
            # Create a model with only biome and climate
            MinimalFactors = BiogenicFactors.create_model({"biome", "climate"})
            factors = MinimalFactors(biome=Biome.DESERTS, climate=Climate.TROPICAL)
        """
        if fields is None:
            fields = set(cls._all_fields.keys())
        
        # Validate that all requested fields exist
        invalid_fields = fields - set(cls._all_fields.keys())
        if invalid_fields:
            raise ValueError(f"Unknown field(s): {', '.join(invalid_fields)}")
        
        # Build field definitions
        field_definitions = {}
        for field_name in fields:
            field_type = cls._all_fields[field_name]
            
            # Add default value if available
            if field_name in cls._default_values:
                field_definitions[field_name] = (field_type, Field(default=cls._default_values[field_name]))
            else:
                field_definitions[field_name] = (field_type, ...)  # ... means required
        
        # Create and return the model
        NewModel = create_model(
            'BiogenicFactors', 
            __base__=BaseBiogenicFactors,
            **field_definitions
        )
        
        # override repr/str to avoid model_config leaking
        def __repr__(self):
            return f"{self.__class__.__name__}({self.todict()})"

        def __str__(self):
            return f"{self.__class__.__name__}({self.todict()})"

        NewModel.__repr__ = __repr__
        NewModel.__str__ = __str__
        
        return NewModel
    
    @classmethod
    def get_available_fields(cls) -> Dict[str, Type]:
        """Return all available fields and their types."""
        return cls._all_fields.copy()
    
    def todict(self, method: ToDictMethod = "name") -> Dict:
        """
        Convert the model to a dictionary.
        
        Args:
            method: Either "name" to use enum names or "value" to use enum values
            
        Returns:
            Dictionary of biogenic factors
        """
        if method not in ["name", "value"]:
            raise ConversionMethodUnknownException(
                conversion_method=method,
                available_methods=ToDictMethod.__args__)
            
        # Get model as dict, with compatibility for both Pydantic v1 and v2
        try:
            # Pydantic v2
            model_dict = self.model_dump(exclude={"model_config"})
        except AttributeError:
            # Pydantic v1 fallback
            model_dict = self.dict(exclude={"model_config"})
        
        result = {}
        for field_name, field_value in model_dict.items():
            # Skip model_config and other internal fields
            if field_name == "model_config" or field_name.startswith("_"):
                continue
            if field_value is not None:
                # Check if the field value is an enum (has name and value attributes)
                if hasattr(field_value, 'name') and hasattr(field_value, 'value'):
                    if method == "name":
                        result[field_name] = field_value.name
                    else:  # method == "value"
                        result[field_name] = field_value.value
                else:
                    # If not an enum, just use the value as is
                    result[field_name] = field_value
        return result
    
    @classmethod
    def fromdict(cls, data: Dict, method: ToDictMethod = "name") -> BaseBiogenicFactors:
        """
        Create a model instance from a dictionary.
        
        Args:
            data: Dictionary of field values
            method: Either "name" to interpret values as enum names or "value" for enum values
            
        Returns:
            New BiogenicFactors instance
        """
        if method not in ["name", "value"]:
            raise ConversionMethodUnknownException(
                conversion_method=method,
                available_methods=ToDictMethod.__args__)
        
        # Convert dictionary values to enums based on method
        converted_data = {}
        
        # Get fields for this specific model class with Pydantic version compatibility
        try:
            # Pydantic v2
            model_fields = cls.model_fields
            field_names = model_fields.keys()
        except AttributeError:
            # Pydantic v1 fallback
            model_fields = cls.__fields__
            field_names = model_fields.keys()
        
        for field_name, raw_value in data.items():
            if field_name not in field_names:
                continue
                
            # Skip fields that aren't defined in our _all_fields
            if field_name not in cls._all_fields:
                continue
                
            # Get expected enum type
            enum_type = cls._all_fields[field_name]
            
            # Convert based on method
            if method == "name":
                converted_data[field_name] = enum_type.from_key(raw_value)
            else:  # method == "value"
                converted_data[field_name] = enum_type.from_value(raw_value)
        
        return cls(**converted_data)


# For backward compatibility, create the full BiogenicFactors model
BiogenicFactors = BaseBiogenicFactors.create_model()


if __name__ == "__main__":
    """ """
    # Create with all fields (same as the original)
    factors = BiogenicFactors(
        biome=Biome.DESERTS,
        climate=Climate.TROPICAL,
        soil_type=SoilType.ORGANIC
    )

    # Create a custom model with only 3 fields
    MinimalFactors = BaseBiogenicFactors.create_model(
        {"biome", "soil_type", "treatment_factor"}
    )

    # Create an instance
    factors = MinimalFactors(
        biome=Biome.DESERTS,
        soil_type=SoilType.ORGANIC,
        treatment_factor=TreatmentFactor.TERTIARY
    )

    # This would raise an error because climate isn't included in this model
    # factors = MinimalFactors(biome=Biome.DESERTS, climate=Climate.TROPICAL)
    
    # Get all available fields and their types
    field_info = BaseBiogenicFactors.get_available_fields()
    rprint("Available fields and types:")
    rprint(field_info)
    # {'biome': <enum 'Biome'>, 'climate': <enum 'Climate'>, ...}
    
    # Using the name method (default)
    factors_dict = factors.todict()
    new_factors = MinimalFactors.fromdict(factors_dict)

    # Using the value method
    factors_dict_values = factors.todict(method="value")
    new_factors_from_values = MinimalFactors.fromdict(factors_dict_values, method="value")
    rprint("Factors from dictionary:")
    rprint(new_factors_from_values)
    
    # Create a new custom enum
    from enum import Enum

    class VegetationDensity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        
        @classmethod
        def from_key(cls, key):
            return cls[key.upper()]
        
        @classmethod
        def from_value(cls, value):
            for member in cls:
                if member.value == value:
                    return member
            raise ValueError(f"No member with value {value}")

    # Register the new field
    BaseBiogenicFactors.register_field(
        "vegetation_density", 
        VegetationDensity, 
        default_value=VegetationDensity.MEDIUM
    )

    # Create a model with the new field
    ExtendedFactors = BaseBiogenicFactors.create_model(
        {"biome", "climate", "vegetation_density"}
    )

    # Use it
    factors = ExtendedFactors(
        biome=Biome.DESERTS,
        climate=Climate.TROPICAL,
        vegetation_density=VegetationDensity.HIGH
    )
    
    rprint(factors)
    
    
    class ModelSpecification:
        def __init__(self, name: str, fields: Dict[str, Dict[str, Any]]):
            self.name = name
            self.fields = fields
            
        def register_with_biogenic_factors(self):
            BaseBiogenicFactors.update_fields_from_dict(self.fields)
            return BaseBiogenicFactors.create_model(set(self.fields.keys()))

    # Define a model specification
    my_spec = ModelSpecification(
        name="CustomAnalysis",
        fields={
            "biome": {
                "type": Biome,
                "default": "DESERTS"
            },
            "vegetation_density": {  # Use the class we already defined earlier
                "type": VegetationDensity,
                "default": "MEDIUM"  # This will be converted to VegetationDensity.MEDIUM
            }
        }
    )

    # Apply it
    CustomModel = my_spec.register_with_biogenic_factors()
    rprint(CustomModel())
    print(CustomModel())