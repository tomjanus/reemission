""" """
from __future__ import annotations
from typing import ClassVar, Dict, Tuple, List, Union, Sequence
from pydantic import validator, root_validator
import pandas as pd
from reemission.data_models.input_model import BuildStatusModel, \
    BiogenicFactorsModel, CatchmentModel, ReservoirModel, DamDataModel
from reemission.auxiliary import rollout_nested_list


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


class CatchmentModelHeet(CatchmentModel):
    """Model for Re-Emission catchment parameters adapted to read and
    parse model output from HEET"""

    catchment_landuse_map: ClassVar[Dict[str, int]] = {
        "BARE": 6,
        "SNOW_ICE": 8,
        "URBAN": 5,
        "WATER": 7,
        "WETLANDS": 4,
        "CROPS": 1,
        "SHRUBS": 3,
        "FOREST": 2,
        "NODATA": 0}

    @root_validator(pre=True)
    @classmethod
    def root_validator(cls, values):
        """Obtain a vector of c area fractions for RE-EMISSION from fractions
        in c_landcover_[i] fields in raw tabular HEET output data"""
        values["c_area_fractions"] = [
            values['c_landcover_'+str(ix)] for ix in 
            cls.catchment_landuse_map.values()]
        return values

    class Config:
        allow_population_by_field_name = False
        allow_extra_values = True
        fields = {
            "runoff": {'alias': 'c_mar_mm'},
            "area": {'alias': 'c_area_km2'},
            "riv_length": {'alias': "ms_length"},
            "population": {'alias': 'n_population'},
            "area_fractions": {'alias': 'c_area_fractions'},
            "slope": {'alias': 'c_mean_slope_pc'},
            "precip": {'alias': 'c_map_mm'},
            "etransp": {'alias': 'c_mpet_mm'},
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
        values['r_area_fractions'] = cls.map_r_landuse(
                r_landuses=[values['r_landcover_bysoil_'+str(ix)] for
                            ix in range(0, 27)])
        return values

    @classmethod
    def map_r_landuse(
            cls, r_landuses: List[float],
            aggregate: bool = False) -> Sequence:
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
        index_order: Dict["str", List[int]] = {
            "mineral": [6, 8, 5, 7, 4, 1, 3, 2, 0],
            "organic": [15, 17, 14, 16, 13, 10, 12, 11, 9],
            "nodata": [24, 26, 23, 25, 22, 19, 21, 20, 18]}
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


def test_dam_data_model() -> None:
    """Instantiate dam data from dummy data in a dictionary"""
    print("\nParsing dam data...")
    dam_data_dict = {
        "name": "Shweli 1",
        "dam_lon": 97.506,
        "dam_lat": 23.698,
        "r_mean_temp_1": 13.9,
        "r_mean_temp_2": 16.0,
        "r_mean_temp_3": 13.9,
        "r_mean_temp_4": 10.9,
        "r_mean_temp_5": 13.9,
        "r_mean_temp_6": 24,
        "r_mean_temp_7": 25,
        "r_mean_temp_8": 13.9,
        "r_mean_temp_9": 13.9,
        "r_mean_temp_10": 13.9,
        "r_mean_temp_11": 14.2,
        "r_mean_temp_12": 14.8}
    dam_data = DamDataModelHeet(**dam_data_dict)
    print(dam_data.json(indent=2))


def test_heet_build_status_model() -> None:
    """Instantiate build status from dummy data in a dictionary"""
    print("\nParsing build status data...")
    status_dict = {
        "r_status": "ExisTing",
        "r_construction_date": "2000",
        "construction_date": 1998}
    status = BuildStatusModelHeet(**status_dict)
    print(status.json(indent=2))


def test_heet_biogenic_factors_model() -> None:
    """Instantiate biogenic factors from dummy data in a dictionary"""
    print("\nParsing biogenic factors data...")
    biogenic_factors_dict = {
        "c_biome": "Tropical & Subtropical Dry Broadleaf Forests",
        "c_climate_zone": "10",
        "c_soil_type": "MINERAL",
        "c_treatment_factor": "primary (mechanical)",
        "c_landuse_intensity": "low intensity"}
    biogenic_factors = BiogenicFactorsModelHeet(**biogenic_factors_dict)
    print(biogenic_factors.json(indent=2))


def test_heet_catchment_model() -> None:
    """Instantiate catchment data from dummy data in a dictionary"""
    print("\nParsing catchment data...")
    catchment_data_dict = {
        "c_mar_mm": 1115.0,
        "c_area_km2": 12582.613,
        "ms_length": 0.0,
        "n_population": 1587658.0,
        "c_landcover_0": 0.0,
        "c_landcover_1": 0.0,
        "c_landcover_2": 0.003,
        "c_landcover_3": 0.002,
        "c_landcover_4": 0.001,
        "c_landcover_5": 0.146,
        "c_landcover_6": 0.391,
        "c_landcover_7": 0.457,
        "c_landcover_8": 0.0,
        "c_mean_slope_pc": 23.0,
        "c_map_mm": 1498.0,
        "c_mpet_mm": 1123.0,
        "c_masm_mm": 144.0,
        "c_mean_olsen": 5.85}
    catchment_data = CatchmentModelHeet(**catchment_data_dict)
    print(catchment_data.json(indent=2))


def test_heet_reservoir_model() -> None:
    """Instantiate reservoir data from dummy data in a dictionary"""
    print("\nParsing reservoir data...")
    reservoir_dict = {
        "r_volume_m3": 7238166.0,
        "r_area_km2": 1.604,
        "r_maximum_depth_m": 22.0,
        "r_mean_depth_m": 4.5,
        "r_landcover_bysoil_0": 0,
        "r_landcover_bysoil_1": 0,
        "r_landcover_bysoil_2": 0,
        "r_landcover_bysoil_3": 0,
        "r_landcover_bysoil_4": 0.5,
        "r_landcover_bysoil_5": 0,
        "r_landcover_bysoil_6": 0.3,
        "r_landcover_bysoil_7": 0,
        "r_landcover_bysoil_8": 0.2,
        "r_landcover_bysoil_9": 0,
        "r_landcover_bysoil_10": 0,
        "r_landcover_bysoil_11": 0,
        "r_landcover_bysoil_12": 0,
        "r_landcover_bysoil_13": 0,
        "r_landcover_bysoil_14": 0,
        "r_landcover_bysoil_15": 0,
        "r_landcover_bysoil_16": 0,
        "r_landcover_bysoil_17": 0,
        "r_landcover_bysoil_18": 0,
        "r_landcover_bysoil_19": 0,
        "r_landcover_bysoil_20": 0,
        "r_landcover_bysoil_21": 0,
        "r_landcover_bysoil_22": 0,
        "r_landcover_bysoil_23": 0,
        "r_landcover_bysoil_24": 0,
        "r_landcover_bysoil_25": 0,
        "r_landcover_bysoil_26": 0,   
        "r_msocs_kgperm2": 6.281,
        "r_mghr_all_kwhperm2perday": 4.66,
        "r_mghr_may_sept_kwhperm2perday": 4.328,
        "r_mghr_nov_mar_kwhperm2perday": 4.852,
        "r_mean_annual_windspeed": 1.08,
        "water_intake_depth": None}
    reservoir_data = ReservoirModelHeet(**reservoir_dict)
    print(reservoir_data.json(indent=2))


if __name__ == "__main__":
    """ """
    test_dam_data_model()
    test_heet_build_status_model()
    test_heet_biogenic_factors_model()
    test_heet_catchment_model()
    test_heet_reservoir_model()