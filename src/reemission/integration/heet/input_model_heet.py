""" """
from __future__ import annotations
from typing import ClassVar, Dict, Tuple, List, Union, Sequence
from pydantic import validator, root_validator
import pandas as pd
from reemission.data_models.input_model import BuildStatusModel, \
    BiogenicFactorsModel, CatchmentModel, ReservoirModel, DamDataModel
from reemission.auxiliary import rollout_nested_list
from reemission.utils import read_config, get_package_file, strip_double_quotes


# Read config defaults
heet_config: Dict = read_config(get_package_file('config/heet.toml'))
# Access the selected runoff, rainfall and evapotranspiration fields from config
runoff_field = strip_double_quotes(heet_config['calculations']['runoff_field'])
precipitation_field = strip_double_quotes(heet_config['calculations']['precipitation_field'])
et_field = strip_double_quotes(heet_config['calculations']['et_field'])


class DamDataModelHeet(DamDataModel):
    """Dam, data model adapted to data format from HEET"""
    
    @root_validator(pre=True)
    @classmethod
    def root_validator(cls, values):
        """Obtain a vector of monthly air temperatures for RE-EMISSION
        """
        values["monthly_temps"] = [
            values['r_mean_temp_'+str(ix)] for ix in range(1, 13)]
        return values
    
    class Config:
        """Add field aliases specific to the output data format received from
        HEET."""
        allow_population_by_field_name = False
        allow_extra_values = False
        fields = {
            'name': {'alias': 'name'},
            'id': {'alias': 'id'},
            'type': {'alias': 'type'},
            'longitude': {'alias': 'dam_lon'},
            'latitude': {'alias': 'dam_lat'},
            'monthly_temps': {'alias': 'monthly_temps'}}


class BuildStatusModelHeet(BuildStatusModel):
    """Build status model adapted to data format from HEET"""

    class Config:
        """Add field aliases specific to the output data format received from
        HEET."""
        allow_population_by_field_name = False
        allow_extra_values = False
        fields = {
            'status': {'alias': 'r_status'},
            'construction_date': {'alias': 'r_construction_date'}}
        
    @classmethod
    def from_row(
            cls, row: pd.Series, r_status, 
            r_construction_date) -> BuildStatusModelHeet:
        """ """
        row = row.copy()
        # Supply missing information
        row['r_status'] = r_status
        row['r_construction_date'] = r_construction_date
        return cls(**row.to_dict())


class BiogenicFactorsModelHeet(BiogenicFactorsModel):
    """Model for Re-Emission biogenic factor parameters adapted to read and
    parse model output from HEET"""

    # Add custom data parsers/translators for reading HEET output data
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
        """Conversion between Köppen-Geiger specific identifiers in HEET output
        and broad classes used in RE-Emission."""
        koppen_id = int(koppen_id)
        for key, value in cls.c_cat_koppen_map.items():
            if koppen_id in range(key[0], key[1]+1):
                return value
        return "unknown"

    @classmethod
    def translate_biome_names(cls, biome_name_heet: str) -> str:
        """Translate biome names in HEET output to names following the 
        convention implemented in RE-Emission"""
        return cls.biome_map[biome_name_heet]

    @classmethod
    def heet_soil_type_to_reemission(cls, heet_soil_type: str) -> str:
        """Turn Heet soil types (capitalized) into Re-Emission soil types (small
        letter)"""
        return heet_soil_type.lower()

    class Config:
        use_enum_values = True
        allow_population_by_field_name = False
        allow_extra_values = False
        fields = {
            'biome': {'alias': 'c_biome'},
            'climate': {'alias': 'c_climate_zone'},
            'soil_type': {'alias': 'c_soil_type'},
            'treatment_factor': {'alias': 'c_treatment_factor'},
            'landuse_intensity': {'alias': 'c_landuse_intensity'}}

    # Input value translators
    _translate_biome = validator('biome', pre=True)(translate_biome_names)
    _translate_climate = validator('climate', pre=True)(c_cat_from_koppen)
    _translate_soil_type = validator('soil_type', pre=True)(
        heet_soil_type_to_reemission)


def map_c_landuse(
        input_fractions: Dict[str, float]) -> List[float]:
    catchment_landuse_map: Dict[str, int] = {
        "BARE": 6,
        "SNOW_ICE": 8,
        "URBAN": 5,
        "WATER": 7,
        "WETLANDS": 4,
        "CROPS": 1,
        "SHRUBS": 2,  # Initially shrubs were 3 and forests were 2. There seems to have been a mistake in how areas were categorized in HEET
        "FOREST": 3,
        "NODATA": 0}
    return [input_fractions['c_landcover_'+str(ix)] for ix in 
            catchment_landuse_map.values()]


def map_r_landuse(r_landuses: List[float], aggregate: bool = False) -> Sequence:
    """
    Maps between 27 categories in reservoir landuse output data and
    9 categories used by the re-emission tool
    NOTE: DATA IS DIVIDED INTO LANDUSE TYPE PER SOIL (MINERAL, ORGANIC, and
    NO-DATA)

    Processes those three soil categories independently and creates 3 9x1
    lists.

    If aggregate is True, adds the three lists together and output a single
    list in which each item of index "i" for "i = 0:8" is the sum of items
    of index "i" in all three lists.
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
    

class CatchmentModelHeet(CatchmentModel):
    """Model for Re-Emission catchment parameters adapted to read and
    parse model output from HEET"""
    runoff_field: str = 'c_mar_mm_alt2'

    @root_validator(pre=True)
    @classmethod
    def root_validator(cls, values):
        """Obtain a vector of c area fractions for RE-EMISSION from fractions
        in c_landcover_[i] fields in raw tabular HEET output data.
        Remaps area fraction indices to match order of landuses in constants.Landuse"""
        values["c_area_fractions"] = map_c_landuse(values)
        return values

    class Config:
        import inspect
        allow_population_by_field_name = False
        allow_extra_values = True
        fields = {
            "runoff": {'alias': runoff_field},
            "area": {'alias': 'c_area_km2'},
            "riv_length": {'alias': "ms_length"},
            "population": {'alias': 'n_population'},
            "area_fractions": {'alias': 'c_area_fractions'},
            "slope": {'alias': 'c_mean_slope_pc'},
            "precip": {'alias': precipitation_field},
            "etransp": {'alias': et_field},
            "soil_wetness": {'alias': 'c_masm_mm'},
            "mean_olsen": {'alias': 'c_mean_olsen'}}


class ReservoirModelHeet(ReservoirModel):
    """Model for Re-Emission reservoir parameters adapted to read and
    parse model output from HEET"""

    @root_validator(pre=True)
    @classmethod
    def root_validator(cls, values):
        """Obtain a vector of r area fractions for RE-EMISSION from fractions
        in r_landcover_[i] fields in raw tabular HEET output data"""
        values['r_area_fractions'] = map_r_landuse(
                r_landuses=[values['r_landcover_bysoil_'+str(ix)] for
                            ix in range(0, 27)])
        return values

    def get_water_area_frac_in_res(self) -> float:
        """Finds how much water was initially within the reservoir contour
        prior to impoundment"""
        return sum([self.area_fractions[i] for i in (7, 16, 25)])

    class Config:
        allow_population_by_field_name = False
        allow_extra_values = True
        fields = {
            "volume": {'alias': 'r_volume_m3'},
            "area": {'alias': 'r_area_km2'},
            "max_depth": {'alias': "r_maximum_depth_m"},
            "mean_depth": {'alias': 'r_mean_depth_m'},
            "area_fractions": {'alias': "r_area_fractions"},
            "soil_carbon": {'alias': "r_msocs_kgperm2"},
            "mean_radiance": {'alias': "r_mghr_all_kwhperm2perday"},
            "mean_radiance_may_sept": {'alias': "r_mghr_may_sept_kwhperm2perday"},
            "mean_radiance_nov_mar": {'alias': "r_mghr_nov_mar_kwhperm2perday"},
            "mean_monthly_windspeed": {'alias': "r_mean_annual_windspeed"},
            "water_intake_depth": {}}


if __name__ == "__main__":
    """ """