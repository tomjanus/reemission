""" """
import itertools
import unittest
from unittest.mock import patch, Mock
from typing import Dict, List, Any
import jsonschema
import pandas as pd
from reemission.integration.heet.input_model_heet import map_c_landuse, map_r_landuse
from reemission.utils import get_package_file, load_yaml, load_json
from reemission.emissions import CarbonDioxideEmission, MethaneEmission
from reemission.reservoir import Reservoir
from reemission.catchment import Catchment
from reemission.temperature import MonthlyTemperature
from reemission.biogenic import BiogenicFactors, Climate, SoilType


class TestLanduseMapping(unittest.TestCase):
    """Checks if data mapping from heet to re-emission and pre-impoundment
    emission calculation in re-emission produce correct results"""
    
    @classmethod
    def setUpClass(cls):
        cls.catchment_fractions = pd.read_csv(
            get_package_file("../../tests/test_data/catchment_fractions.csv"))
        cls.reservoir_fractions = pd.read_csv(
            get_package_file("../../tests/test_data/reservoir_fractions.csv"))

    def get_c_landuse_heet(self) -> List[List[float]]:
        return self.catchment_fractions.iloc[:, :9].values.tolist()
    
    def get_c_landuse_reemission(self) -> List[List[float]]:
        return self.catchment_fractions.iloc[:, -9:].values.tolist()
    
    def get_r_landuse_heet(self) -> List[List[float]]:
        return self.reservoir_fractions.iloc[:, :27].values.tolist()
    
    def get_r_landuse_reemission(self) -> List[List[float]]:
        return self.reservoir_fractions.iloc[:, -27:].values.tolist()

    def test_catchment_landuse_mapping(self):
        """Test mapping between heet and re-emission
        Re-emission landuse map is shown below:
            "bare", 
            "snow and ice"
            "urban"
            "water"
            "wetlands"
            "crops"
            "shrubs"
            "forest"
            "no data"
        Heet landuse map is as follows:
            c_landcover_0	'No Data'
            c_landcover_1	'Croplands'
            c_landcover_2	'Grassland/Shrubland'
            c_landcover_3	'Forest'
            c_landcover_4	'Wetlands'
            c_landcover_5	'Settlements'
            c_landcover_6	'Bare Areas'
            c_landcover_7	'Water Bodies'
            c_landcover_8	'Permanent snow and ice'"""
        # Load input/output data
        heet_data = self.get_c_landuse_heet()
        reemission_des_outputs = self.get_c_landuse_reemission()

        for row in range(0,len(heet_data)):
            heet_c_fractions = {
                f'c_landcover_{str(ix)}': fraction for ix, fraction in enumerate(heet_data[row])} 
            reemission_fractions_desired = reemission_des_outputs[row]
            reemission_fractions_calc = map_c_landuse(heet_c_fractions)
            self.assertEqual(reemission_fractions_desired, reemission_fractions_calc)

    def test_reservoir_landuse_mapping(self):
        """Reservoir landuse mapping follows the same order as catchment landuse mapping
        but the vector is 3 x 9 = 27 long as the landuse mapping is divided into three
        categories based on soil type: Mineral -> Organic -> No Data"""
        heet_data = self.get_r_landuse_heet()
        reemission_des_outputs = self.get_r_landuse_reemission()

        for row in range(0,len(heet_data)):
            reemission_fractions_desired = reemission_des_outputs[row]
            reemission_fractions_calc = map_r_landuse(heet_data[row])
            try:
                self.assertEqual(reemission_fractions_desired, reemission_fractions_calc)
            except AssertionError as e:
                print("Failed at row %d" % row)
                raise 


class TestPreimpoundment(unittest.TestCase):
    """Checks calculation of pre-impoundment emissions with re-emission using 
    re-emission inputs.
    """

    @classmethod
    def setUpClass(cls):
        """Load pre-impoundment-emission-tables from the test environment"""
        cls.ch4_preimpoundment = load_yaml(
            get_package_file("../../tests/test_data/pre-impoundment_ch4.yaml"))
        cls.co2_preimpoundment = load_yaml(
            get_package_file("../../tests/test_data/pre-impoundment_co2.yaml"))
        # Load schema
        cls.preimpoundment_schema = load_json(
            get_package_file("schemas/pre_impoundment_yaml_schema.json"))
        # Validate the pre-impoundment emission files
        jsonschema.validate(
            instance=cls.ch4_preimpoundment, schema=cls.preimpoundment_schema)
        jsonschema.validate(
            instance=cls.co2_preimpoundment, schema=cls.preimpoundment_schema)
        # Instantiate catchment and reservoir objects
        r_area_fractions = [0]*27
        c_area_fractions = [0]*9
        r_area_fractions[0] = 1
        c_area_fractions[0] = 1
        b_factors = BiogenicFactors.fromdict({
            "biome": "TROPICALMOISTBROADLEAF",
            "climate": "TROPICAL",
            "soil_type": "MINERAL",
            "treatment_factor": "NONE",
            "landuse_intensity": "LOW"})
        cls.test_reservoir = Reservoir(
            (0,0), MonthlyTemperature([0]*12), 0, 0, 0, 0, 1, 0, 
            r_area_fractions, 0, 0, 0, 0, 0)
        cls.test_catchment = Catchment(
            0, 0, 0, 0, 0, 0, 0, 0, 0, c_area_fractions, b_factors)
        cls.test_climates = [Climate.BOREAL, Climate.SUBTROPICAL, Climate.TEMPERATE, Climate.TROPICAL]
        cls.test_soiltypes = [SoilType.MINERAL, SoilType.ORGANIC]


    def test_preimpoundment_co2_emissions(self):
        """ """
        co2_emission = CarbonDioxideEmission(self.test_catchment, self.test_reservoir, 20, "g-res")
        co2_emission.pre_impoundment_table = self.co2_preimpoundment
        co2_emission.par.weight_C = 1
        co2_emission.par.weight_CO2 = 1
        co2_emission.par.co2_gwp100 = 1
        test_data = pd.read_csv(
            get_package_file("../../tests/test_data/reservoir_preimpoundment_co2_test.csv"))
        r_landuse_data = test_data.iloc[:,:27].values.tolist()
        results_data = test_data.iloc[:,-8:].values.tolist()

        for row in range(0, len(r_landuse_data)):
            co2_emission.reservoir.area_fractions = r_landuse_data[row]
            output_ix: int = 0
            for climate, soil in itertools.product(self.test_climates, self.test_soiltypes):
                co2_emission.catchment.biogenic_factors.climate = climate
                co2_emission.catchment.biogenic_factors.soil_type = soil
                try:
                    # 0.01 is a coefficient that is applied in the method to convert area from km2 to ha
                    self.assertAlmostEqual(0.01 * co2_emission.pre_impoundment(), results_data[row][output_ix])
                    output_ix += 1
                except AssertionError as e:
                    print("Preimpoundment CH4 emission failed at row %d" % row)
                    raise e

        

    def test_preimpoundment_ch4_emissions(self):
        """ """
        ch4_emission = MethaneEmission(
            self.test_catchment, self.test_reservoir, 
            MonthlyTemperature([0,0,0,0,0,0,0,0,0,0,0,0]))
        ch4_emission.pre_impoundment_table = self.ch4_preimpoundment
        ch4_emission.par.weight_CH4 = 1
        ch4_emission.par.weight_CO2 = 1
        ch4_emission.par.ch4_gwp100 = 1

        test_data = pd.read_csv(
            get_package_file("../../tests/test_data/reservoir_preimpoundment_ch4_test.csv"))
        r_landuse_data = test_data.iloc[:,:27].values.tolist()
        results_data = test_data.iloc[:,-8:].values.tolist()
        

        for row in range(0, len(r_landuse_data)):
            ch4_emission.reservoir.area_fractions = r_landuse_data[row]
            output_ix: int = 0
            for climate, soil in itertools.product(self.test_climates, self.test_soiltypes):
                ch4_emission.catchment.biogenic_factors.climate = climate
                ch4_emission.catchment.biogenic_factors.soil_type = soil
                try:
                    # Multiplier 10 is a result of km2->ha coefficient 100 and second coefficient 0.001
                    self.assertAlmostEqual(10 * ch4_emission.pre_impoundment(), results_data[row][output_ix])
                    output_ix += 1
                except AssertionError as e:
                    print("Preimpoundment CH4 emission failed at row %d" % row)
                    raise e
            



if __name__ == '__main__':
    unittest.main()