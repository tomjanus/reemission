"""GeoCARET input model integration for RE-Emission."""
from __future__ import annotations
from typing import ClassVar, Dict, Tuple, List, Union, Sequence, Any, Optional, cast
import sys
import pandas as pd

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from reemission.data_models.input_model import BuildStatusModel, \
    BiogenicFactorsModel, CatchmentModel, ReservoirModel, DamDataModel
from reemission.auxiliary import rollout_nested_list
from reemission.utils import read_config, get_package_file, strip_double_quotes


# Read config defaults
geocaret_config: Dict = read_config(get_package_file('config/geocaret.toml'))
# Access the selected runoff, rainfall and evapotranspiration fields from config
runoff_field = strip_double_quotes(geocaret_config['calculations']['runoff_field'])
precipitation_field = strip_double_quotes(geocaret_config['calculations']['precipitation_field'])
et_field = strip_double_quotes(geocaret_config['calculations']['et_field'])


def validator_decorator(field_name: str, pre: bool = False):
    """Create a validator decorator."""
    mode = "before" if pre else "after"
    return field_validator(field_name, mode=mode)


def root_validator_decorator(pre: bool = False):
    """Create a root validator decorator."""
    mode = "before" if pre else "after"
    return model_validator(mode=mode)


class DamDataModelGeoCaret(DamDataModel):
    """Dam, data model adapted to data format from GeoCARET"""
    
    # Define field aliases based on version
    name: str = Field(alias='name')
    id: str = Field(alias='id')
    type: str = Field(alias='type')
    longitude: float = Field(alias='dam_lon')
    latitude: float = Field(alias='dam_lat')
    
    @field_validator('id', mode='before')
    @classmethod
    def convert_id_to_string(cls, value):
        """Convert numeric IDs to strings"""
        return str(value)
    
    # Root validator for monthly temperatures
    @model_validator(mode="before")
    @classmethod
    def transform_data(cls, values):
        """Obtain a vector of monthly air temperatures for RE-EMISSION"""
        values = cast(Dict[str, Any], values)
        values["monthly_temps"] = [
            values['r_mean_temp_'+str(ix)] for ix in range(1, 13)]
        return values
    
    # Version-specific configuration
    model_config = ConfigDict(
        populate_by_name=False,  # was: validate_by_name=False
        extra="ignore"           # was: allow_extra_values=False
    )


class BuildStatusModelGeoCaret(BuildStatusModel):
    """Build status model adapted to data format from GeoCARET"""

    # Define field aliases based on version
    status: str = Field(alias='r_status')
    construction_date: Optional[int] = Field(default=None, alias='r_construction_date')

    # Version-specific configuration
    model_config = ConfigDict(
        populate_by_name=False,
        extra='ignore'
    )
        
    @classmethod
    def from_row(
            cls, row: pd.Series, r_status, 
            r_construction_date) -> BuildStatusModelGeoCaret:
        """Create model from dataframe row with additional parameters."""
        row = row.copy()
        # Supply missing information
        row['r_status'] = r_status
        row['r_construction_date'] = r_construction_date
        return cls(**row.to_dict())


class BiogenicFactorsModelGeoCaret(BiogenicFactorsModel):
    """Model for Re-Emission biogenic factor parameters adapted to read and
    parse model output from GeoCARET"""

    # Define field aliases based on version
    biome: str = Field(alias='c_biome')
    climate: str = Field(alias='c_climate_zone')
    soil_type: str = Field(alias='c_soil_type')
    treatment_factor: str = Field(alias='c_treatment_factor')
    landuse_intensity: str = Field(alias='c_landuse_intensity')
    
    # Add custom data parsers/translators for reading GeoCARET output data
    biome_map: ClassVar[Dict[str, str]] = {
        "Deserts & Xeric Shrublands ": "deserts",
        "Mediterranean Forests Woodlands & Scrub": "mediterreanan forests",
        "Montane Grasslands & Shrublands": "montane grasslands",
        "Temperate Broadleaf & Mixed Forests":
            "temperate broadleaf and mixed",
        "Temperate Conifer Forests": "temperate coniferous",
        "Temperate Grasslands Savannas & Shrublands":
            "temperate grasslands",
        "Tropical & Subtropical Dry Broadleaf Forests":
            "tropical dry broadleaf",
        "Tropical & Subtropical Grasslands Savannas & Shrublands ":
            "tropical grasslands",
        "Tropical & Subtropical Moist Broadleaf Forests":
            "tropical moist broadleaf",
        "Tundra": "tundra"}

    c_cat_koppen_map: ClassVar[Dict[Tuple[int, int], str]] = {
        (1, 3): "tropical",
        (4, 7): "subtropical",
        (8, 16): "temperate",
        (17, 30): "boreal"}

    @classmethod
    def c_cat_from_koppen(cls, koppen_id: Union[int, float, str]) -> str:
        """Conversion between Köppen-Geiger specific identifiers in GeoCARET output
        and broad classes used in RE-Emission."""
        koppen_id = int(koppen_id)
        for key, value in cls.c_cat_koppen_map.items():
            if koppen_id in range(key[0], key[1]+1):
                return value
        return "unknown"

    @classmethod
    def translate_biome_names(cls, biome_name_geocaret: str) -> str:
        """Translate biome names in GeoCARET output to names following the 
        convention implemented in RE-Emission"""
        return cls.biome_map[biome_name_geocaret]

    @classmethod
    def geocaret_soil_type_to_reemission(cls, geocaret_soil_type: str) -> str:
        """Turn GeoCaret soil types (capitalized) into Re-Emission soil types (small
        letter)"""
        return geocaret_soil_type.lower()

    # Version-specific configuration
    model_config = ConfigDict(
        model_dump_enum_values=True,
        populate_by_name=False,
        extra='ignore'
    )

    # Input value translators
    @field_validator('biome', mode='before')
    @classmethod
    def translate_biome(cls, v):
        return cls.translate_biome_names(v)
        
    @field_validator('climate', mode='before')
    @classmethod
    def translate_climate(cls, v):
        return cls.c_cat_from_koppen(v)
        
    @field_validator('soil_type', mode='before')
    @classmethod
    def translate_soil_type(cls, v):
        return cls.geocaret_soil_type_to_reemission(v)


def map_c_landuse(
        input_fractions: Dict[str, float]) -> List[float]:
    catchment_landuse_map: Dict[str, int] = {
        "BARE": 6,
        "SNOW_ICE": 8,
        "URBAN": 5,
        "WATER": 7,
        "WETLANDS": 4,
        "CROPS": 1,
        "SHRUBS": 2,
        "FOREST": 3,
        "NODATA": 0}
    return [input_fractions['c_landcover_'+str(ix)] for ix in 
            catchment_landuse_map.values()]


def map_r_landuse(r_landuses: List[float], aggregate: bool = False) -> Sequence:
    """
    Maps between 27 categories in reservoir landuse output data and
    9 categories used by the re-emission tool.
    """
    index_order: Dict[str, List[int]] = {
        "mineral": [6,  8,  5,  7,  4,  1,  2,  3,  0],
        "organic": [15, 17, 14, 16, 13, 10, 11, 12, 9],
        "nodata":  [24, 26, 23, 25, 22, 19, 20, 21, 18]}
    output_lists: List[List[float]] = []
    for _, indices in index_order.items():
        soil_cat_indices = [r_landuses[index] for index in indices]
        output_lists.append(soil_cat_indices)
    # The below returns a 9x1 vector of aggregated values
    if aggregate:
        output = [sum(x) for x in zip(*output_lists)]
    # The below returns 27x1 vector
    else:
        output = rollout_nested_list(output_lists)
    return output
    

class CatchmentModelGeoCaret(CatchmentModel):
    """Model for Re-Emission catchment parameters adapted to read and
    parse model output from GeoCARET"""
    
    # Define field aliases based on version
    runoff: float = Field(alias=runoff_field)
    area: float = Field(alias='c_area_km2')
    riv_length: float = Field(alias="ms_length")
    population: float = Field(alias='n_population')
    area_fractions: List[float] = Field(alias='c_area_fractions')
    slope: float = Field(alias='c_mean_slope_pc')
    precip: float = Field(alias=precipitation_field)
    etransp: float = Field(alias=et_field)
    soil_wetness: float = Field(alias='c_masm_mm')
    mean_olsen: float = Field(alias='c_mean_olsen')
    
    # Root validator for area fractions
    @model_validator(mode="before")
    @classmethod
    def transform_data(cls, values):
        """Obtain a vector of c area fractions for RE-EMISSION"""
        values = cast(Dict[str, Any], values)
        values["c_area_fractions"] = map_c_landuse(values)
        return values

    # Version-specific configuration
    model_config = ConfigDict(
        populate_by_name=False,
        extra='ignore'
    )


class ReservoirModelGeoCaret(ReservoirModel):
    """Model for Re-Emission reservoir parameters adapted to read and
    parse model output from GeoCARET"""

    # Define field aliases based on version
    volume: float = Field(alias='r_volume_m3')
    area: float = Field(alias='r_area_km2')
    max_depth: float = Field(alias="r_maximum_depth_m")
    mean_depth: float = Field(alias='r_mean_depth_m')
    area_fractions: List[float] = Field(alias="r_area_fractions")
    soil_carbon: float = Field(alias="r_msocs_kgperm2")
    mean_radiance: float = Field(alias="r_mghr_all_kwhperm2perday")
    mean_radiance_may_sept: float = Field(alias="r_mghr_may_sept_kwhperm2perday")
    mean_radiance_nov_mar: float = Field(alias="r_mghr_nov_mar_kwhperm2perday")
    mean_monthly_windspeed: float = Field(alias="r_mean_annual_windspeed")
    water_intake_depth: Optional[float] = None

    # Root validator for area fractions
    @model_validator(mode="before")
    @classmethod
    def transform_data(cls, values):
        """Process reservoir landuse data"""
        values = cast(Dict[str, Any], values)
        values['r_area_fractions'] = map_r_landuse(
            r_landuses=[values['r_landcover_bysoil_'+str(ix)] for
                        ix in range(0, 27)])
        return values

    def get_water_area_frac_in_res(self) -> float:
        """Finds how much water was initially within the reservoir contour
        prior to impoundment"""
        return sum([self.area_fractions[i] for i in (7, 16, 25)])

    # Version-specific configuration
    model_config = ConfigDict(
        populate_by_name=False,
        extra='ignore'
    )


if __name__ == "__main__":
    """ """
