"""Input model for RE-Emission."""
from __future__ import annotations
import logging
import math
import copy
from typing import List, Optional, Literal, Dict, Any, Type, get_type_hints
from datetime import date
import pandas as pd
from reemission.constants import Biome, Climate, SoilType, TreatmentFactor, \
    LanduseIntensity
from reemission.auxiliary import rollout_nested_list

# Import appropriate version-specific components
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.functional_validators import model_validator

# Custom exception hook
def validation_exception_handler(exception_type, exception, traceback):
    """Remove trace and log the exception message"""
    logging.error("%s: %s", exception_type.__name__, exception)

# Model classes
class DamDataModel(BaseModel):
    """Dam information"""
    name: str = Field(description="Dam name")
    id: str = Field(description="Dam ID")
    type: ReservoirType = Field(description="Reservoir type", default="unknown")
    longitude: float = Field(description="Longitude")
    latitude: float = Field(description="Latitude")
    monthly_temps: List[float] = Field(
        description="Monthly average air temperatures, degC", 
        default_factory=list)
    
    model_config = ConfigDict(extra="allow")
    
    # Post init to set default type if not provided
    @model_validator(mode="after")
    def set_defaults(self):
        """Set default values after validation."""
        self.type = 'unknown'
        return self
    
    @classmethod
    def from_row(cls, row: pd.Series) -> DamDataModel:
        return cls(**row.to_dict())


class BuildStatusModel(BaseModel):
    """Build status of a reservoir"""
    status: Literal['existing', 'future'] = Field(
        description="Build status of a reservoir - existing/future")
    construction_date: Optional[int] = Field(
        None, description="Date of construction of the reservoir", ge=1800,
        le=2200)
    
    model_config = ConfigDict(extra="allow")
    
    @classmethod
    def from_row(cls, row: pd.Series) -> BuildStatusModel:
        return cls(**row.to_dict())

    @field_validator('status', mode='before')
    @classmethod
    def parse_status_literals(cls, value):
        """Convert status literals into lower case before validation"""
        return value.lower()

    @field_validator('construction_date')
    @classmethod
    def check_construction_date(cls, value, info):
        """Check construction date against status and current date"""
        values = info.data
        if value and 'status' in values:
            construction_date = date(value, 1, 1)
            if values['status'] == "future" and construction_date < date.today():
                raise ValueError(
                    "Future reservoir has construction date in the past.")
            if values['status'] == "existing" and construction_date > date.today():
                raise ValueError(
                    "Existing reservoir has construction date in the future.")
        return value


class BiogenicFactorsModel(BaseModel):
    """Model for Re-Emission biogenic factor parameters"""
    biome: Biome = Field(
        description="Biome of the area in which reservoir is located")
    climate: Climate = Field(description="Climate classification")
    soil_type: SoilType = Field(
        description="Type of soil flooded by the reservoir: mineral/organic")
    treatment_factor: TreatmentFactor = Field(
        description="Degree of wastewater treament in the catchment")
    landuse_intensity: LanduseIntensity = Field(
        description="Degree of agricultural land use")

    model_config = ConfigDict(extra="allow", model_dump_enum_values=True)
    
    @classmethod
    def from_row(cls, row: pd.Series) -> BiogenicFactorsModel:
        return cls(**row.to_dict())

    

class CatchmentModel(BaseModel):
    """Model for Re-Emission catchment parameters"""
    runoff: float = Field(..., gt=0, description="Annual runoff, mm/year")
    area: float = Field(..., gt=0, description="Catchment area, km2")
    riv_length: float = Field(
        ..., ge=0, description="Inundated river length, km")
    population: float = Field(
        ..., ge=0, description="Population in the catchment, capita")
    area_fractions: List[float] = Field(
        description="Area fractions of landuse types, -", 
        default_factory=list)
    slope: float = Field(..., ge=0, description="Mean catchment slope, %")
    precip: float = Field(
        ..., ge=0, description="Mean annual precipitation, mm/year")
    etransp: float = Field(
        ..., ge=0, description="Mean annual evapotranspiration, mm/year")
    soil_wetness: float = Field(
        ..., ge=0, description="Soil wetness, mm over profile")
    mean_olsen: float = Field(
        ..., ge=0, description="Soil Olsen P content, kgP/ha")

    model_config = ConfigDict(extra="allow")
    
    @classmethod
    def from_row(cls, row: pd.Series) -> CatchmentModel:
        return cls(**row.to_dict())

    @field_validator('riv_length', mode='before')
    @classmethod
    def convert_novals_to_zero(cls, value):
        if math.isnan(value):
            return 0.0
        return value

    @field_validator('area_fractions')
    @classmethod
    def check_area_fractions(cls, value):
        """Check that area fractions add up to approx. 1"""
        try:
            assert 1 - EPS <= sum(value) <= 1 + EPS
        except AssertionError as err:
            raise ValueError(
                f"Sum {sum(value)} of area fractions not equal 1") from err
        return value


class ReservoirModel(BaseModel):
    """Model for Re-Emission reservoir parameters"""
    volume: float = Field(..., gt=0, description="Reservoir volume, m3")
    area: float = Field(..., gt=0, description="Reservoir area, km2")
    max_depth: float = Field(
        ..., gt=0, description="Maximum reservoir depth, m")
    mean_depth: float = Field(
        ..., gt=0, description="Mean reservoir depth, m")
    area_fractions: List[float] = Field(
        description="Inundated area fractions of landuse types, -", 
        default_factory=list)
    soil_carbon: float = Field(
        ..., ge=0, description="Soil carbon in inundated area, kgC/m2")
    mean_radiance: float = Field(
        ..., ge=0, description="Mean monthly horizontal radiance, kWh/m2/d")
    mean_radiance_may_sept: float = Field(
        ..., ge=0, 
        description="Mean monthly horizontal radiance May-Sept, kWh/m2/d")
    mean_radiance_nov_mar: float = Field(
        ..., ge=0, 
        description="Mean monthly horizontal radiance Nov-Mar, kWh/m2/d")
    mean_monthly_windspeed: float = Field(
        ..., ge=0, description="Mean monthly wind speed, m/s")
    water_intake_depth: Optional[float] = Field(
        default=None, description="Water intake depth below surface, m")

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    
    @classmethod
    def from_row(cls, row: pd.Series) -> ReservoirModel:
        return cls(**row.to_dict())

    @field_validator('area_fractions')
    @classmethod
    def check_area_fractions(cls, value):
        """Check that area fractions add up to approx. 1"""
        value = rollout_nested_list(copy.deepcopy(value))
        try:
            assert 1 - EPS <= sum(value) <= 1 + EPS
        except AssertionError as err:
            raise ValueError(
                f"Sum {sum(value)} of area fractions not equal 1") from err
        return value

    @field_validator('mean_depth')
    @classmethod
    def validate_mean_depth(cls, value, info):
        """Check that mean_dept < max_depth"""
        values = info.data
        max_depth = values.get('max_depth')
        if max_depth is not None and value > max_depth:
            raise ValueError(
                f'Mean depth {value} larger than max depth {max_depth}')
        return value

    @field_validator("water_intake_depth")
    @classmethod
    def validate_water_intake_depth(cls, value, info):
        """Check that (if water intake depth) then depth <= max_depth"""
        values = info.data
        max_depth = values.get('max_depth')
        if value and max_depth:
            if value > max_depth:
                raise ValueError("Water intake below reservoir bottom")
        return value


class InputModel(BaseModel):
    """Model for re-emission inputs"""  
    dam_data: DamDataModel
    build_status: BuildStatusModel
    catchment: CatchmentModel
    reservoir: ReservoirModel
    biogenic_factors: BiogenicFactorsModel


EPS = 0.01
ReservoirType = Literal['hydroelectric', 'multipurpose', 'potable', 'irrigation', 'flood control', 'unknown']


if __name__ == "__main__":
    """ """