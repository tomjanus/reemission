""" """
import unittest
from reemission.integration.geocaret.input_model_geocaret import (
    DamDataModelHeet, BuildStatusModelHeet, BiogenicFactorsModelHeet, 
    CatchmentModelHeet, ReservoirModelHeet)


class TestHeetInputModel(unittest.TestCase):
    """ """
    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def setUp(self):
        # 
        ...

    def tearDown(self):
        # 
        ...

    def test_dam_data_model(self) -> None:
        """Instantiate dam data from dummy data in a dictionary"""
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
        dam_data_txt = dam_data.json(indent=None)
        dam_data_expected = \
            '{"name": "Shweli 1", "longitude": 97.506, "latitude": 23.698, ' + \
            '"monthly_temps": [13.9, 16.0, 13.9, 10.9, 13.9, 24.0, 25.0, 13.9' + \
            ', 13.9, 13.9, 14.2, 14.8]}'
        self.assertEqual(dam_data_txt, dam_data_expected)

    def test_geocaret_build_status_model(self) -> None:
        """Instantiate build status from dummy data in a dictionary"""
        status_dict = {
            "r_status": "ExisTing",
            "r_construction_date": "2000",
            "construction_date": 1998}
        status = BuildStatusModelHeet(**status_dict)
        status_data_txt = status.json(indent=None)
        status_data_expected = '{"status": "existing", "construction_date": 2000}'
        self.assertEqual(status_data_txt, status_data_expected)

    def test_geocaret_biogenic_factors_model(self) -> None:
        """Instantiate biogenic factors from dummy data in a dictionary"""
        biogenic_factors_dict = {
            "c_biome": "Tropical & Subtropical Dry Broadleaf Forests",
            "c_climate_zone": "10",
            "c_soil_type": "MINERAL",
            "c_treatment_factor": "primary (mechanical)",
            "c_landuse_intensity": "low intensity"}
        biogenic_factors = BiogenicFactorsModelHeet(**biogenic_factors_dict)
        biogenic_factors_data_txt = biogenic_factors.json(indent=None)
        biogenic_factors_data_expected = '{"biome": "tropical dry broadleaf", ' \
        + '"climate": "temperate", "soil_type": "mineral", "treatment_factor": ' \
        + '"primary (mechanical)", "landuse_intensity": "low intensity"}'
        self.assertEqual(biogenic_factors_data_txt, biogenic_factors_data_expected)

    def test_geocaret_catchment_model(self) -> None:
        """Instantiate catchment data from dummy data in a dictionary"""
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
        catchment_data_txt = catchment_data.json(indent=None)
        catchment_data_expected = \
            '{"runoff": 1115.0, "area": 12582.613, "riv_length": 0.0, "population": 1587658.0, ' + \
            '"area_fractions": [0.391, 0.0, 0.146, 0.457, 0.001, 0.0, 0.002, 0.003, 0.0], ' + \
            '"slope": 23.0, "precip": 1498.0, "etransp": 1123.0, "soil_wetness": 144.0, "mean_olsen": 5.85}'
        self.assertEqual(catchment_data_txt, catchment_data_expected)

    def test_geocaret_reservoir_model(self) -> None:
        """Instantiate reservoir data from dummy data in a dictionary"""
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
        reservoir_data_txt = reservoir_data.json(indent=None)
        reservoir_data_expected = \
            '{"volume": 7238166.0, "area": 1.604, "max_depth": 22.0, "mean_depth": 4.5, ' + \
            '"area_fractions": [0.3, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ' + \
            '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ' + \
            '"soil_carbon": 6.281, "mean_radiance": 4.66, "mean_radiance_may_sept": 4.328, ' + \
            '"mean_radiance_nov_mar": 4.852, "mean_monthly_windspeed": 1.08, "water_intake_depth": null}'
        self.assertEqual(reservoir_data_txt, reservoir_data_expected)

    
if __name__ == '__main__':
    unittest.main()