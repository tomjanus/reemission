""" """
import itertools
import unittest
from unittest.mock import patch, Mock
from typing import Dict, List, Any
import jsonschema
import numpy as np
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

        def generate_input_c_landuse_array() -> np.ndarray:
            top = np.eye(9)
            bottom_1 = np.eye(9) * 0.5
            bottom_2 = np.roll(bottom_1, 1, axis=1)
            bottom = bottom_1 + bottom_2
            return np.vstack((top, bottom))
        
        cls.c_landuse_heet = generate_input_c_landuse_array()
        
        cls.c_landuse_reemission = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]
    
        cls.r_landuse_heet = np.eye(27)

        cls.r_landuse_reemission = [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        

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
        for row in range(0,len(self.c_landuse_heet)):
            heet_c_fractions = {
                f'c_landcover_{str(ix)}': fraction for ix, fraction in 
                enumerate(self.c_landuse_heet[row])} 
            reemission_fractions_desired = self.c_landuse_reemission[row]
            reemission_fractions_calc = map_c_landuse(heet_c_fractions)
            self.assertEqual(reemission_fractions_desired, reemission_fractions_calc)

    def test_reservoir_landuse_mapping(self):
        """Reservoir landuse mapping follows the same order as catchment landuse mapping
        but the vector is 3 x 9 = 27 long as the landuse mapping is divided into three
        categories based on soil type: Mineral -> Organic -> No Data"""
        heet_data = self.r_landuse_heet
        reemission_des_outputs = self.r_landuse_reemission

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
        """Load pre-impoundment-emission-tables"""

        cls.co2_preimpoundment = {
            'boreal': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': -0.4, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 2.8, 'crops': 7.9, 'forest': 0.6, 'shrubs': 5.7, 'urban': 6.4, 'wetlands': -0.5}}, 
            'subtropical': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': -1.4, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 2.0, 'crops': 11.7, 'forest': 2.6, 'shrubs': 9.6, 'urban': 6.4, 'wetlands': 0.1}}, 
            'temperate': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': -0.9, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0},
                'organic': {'bare': 2.8, 'crops': 7.9, 'forest': 0.0, 'shrubs': 5.0, 'urban': 6.4, 'wetlands': -0.5}}, 
            'tropical': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': -1.4, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 2.0, 'crops': 11.7, 'forest': 15.3, 'shrubs': 9.6, 'urban': 6.4, 'wetlands': 0.0}}}

        
        cls.ch4_preimpoundment = {
            'boreal': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': 0.0, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 6.1, 'crops': 0.0, 'forest': 4.5, 'shrubs': 1.4, 'urban': 19.6, 'wetlands': 89.0}}, 
            'subtropical': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': 0.0, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 7.0, 'crops': 11.7, 'forest': 2.5, 'shrubs': 7.0, 'urban': 19.6, 'wetlands': 116.3}}, 
            'temperate': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': 0.0, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 6.1, 'crops': 0.0, 'forest': 0.0, 'shrubs': 18.9, 'urban': 19.6, 'wetlands': 0.0}}, 
            'tropical': {
                'mineral': {'bare': 0.0, 'crops': 0.0, 'forest': 0.0, 'shrubs': 0.0, 'urban': 0.0, 'wetlands': 0.0}, 
                'organic': {'bare': 7.0, 'crops': 75.0, 'forest': 1.8, 'shrubs': 7.0, 'urban': 19.6, 'wetlands': 41.0}}}

        cls.r_landuse = np.eye(27)

        # Instantiate catchment and reservoir objects
        r_area_fractions, c_area_fractions = [0]*27, [0]*9
        r_area_fractions[0], c_area_fractions[0] = 1, 1
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
        cls.test_climates = [
            Climate.BOREAL, Climate.SUBTROPICAL, Climate.TEMPERATE, Climate.TROPICAL]
        cls.test_soiltypes = [SoilType.MINERAL, SoilType.ORGANIC]


    def test_preimpoundment_co2_emissions(self):
        """ """
        co2_emission = CarbonDioxideEmission(self.test_catchment, self.test_reservoir, 20, "g-res")
        co2_emission.pre_impoundment_table = self.co2_preimpoundment
        co2_emission.par.weight_C = 1
        co2_emission.par.weight_CO2 = 1
        co2_emission.par.co2_gwp100 = 1
        co2_emission_calc = [
            [0.0, 2.8, 0.0, 2.0, 0.0, 2.8, 0.0, 2.0], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 6.4, 0.0, 6.4, 0.0, 6.4, 0.0, 6.4], 
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, -0.5, 0.0, 0.1, 0.0, -0.5, 0.0, 0.0], 
            [0.0, 7.9, 0.0, 11.7, 0.0, 7.9, 0.0, 11.7], 
            [0.0, 5.7, 0.0, 9.6, 0.0, 5.0, 0.0, 9.6],
            [-0.4, 0.6, -1.4, 2.6, -0.9, 0.0, -1.4, 15.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.8, 0.0, 2.0, 0.0, 2.8, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 6.4, 0.0, 6.4, 0.0, 6.4, 0.0, 6.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.1, 0.0, -0.5, 0.0, 0.0],
            [0.0, 7.9, 0.0, 11.7, 0.0, 7.9, 0.0, 11.7],
            [0.0, 5.7, 0.0, 9.6, 0.0, 5.0, 0.0, 9.6],
            [-0.4, 0.6, -1.4, 2.6, -0.9, 0.0, -1.4, 15.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.8, 0.0, 2.0, 0.0, 2.8, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 6.4, 0.0, 6.4, 0.0, 6.4, 0.0, 6.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.1, 0.0, -0.5, 0.0, 0.0],
            [0.0, 7.9, 0.0, 11.7, 0.0, 7.9, 0.0, 11.7],
            [0.0, 5.7, 0.0, 9.6, 0.0, 5.0, 0.0, 9.6],
            [-0.4, 0.6, -1.4, 2.6, -0.9, 0.0, -1.4, 15.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        for row in range(0, len(self.r_landuse)):
            co2_emission.reservoir.area_fractions = self.r_landuse[row]
            output_ix: int = 0
            for climate, soil in itertools.product(self.test_climates, self.test_soiltypes):
                co2_emission.catchment.biogenic_factors.climate = climate
                co2_emission.catchment.biogenic_factors.soil_type = soil
                try:
                    # 0.01 is a coefficient that is applied in the method to convert area from km2 to ha
                    self.assertAlmostEqual(
                        0.01 * co2_emission.pre_impoundment(), 
                        co2_emission_calc[row][output_ix])
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
        ch4_emission_calc = [
            [0.0, 6.1, 0.0, 7.0, 0.0, 6.1, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 19.6, 0.0, 19.6, 0.0, 19.6, 0.0, 19.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 89.0, 0.0, 116.3, 0.0, 0.0, 0.0, 41.0],
            [0.0, 0.0, 0.0, 11.7, 0.0, 0.0, 0.0, 75.0],
            [0.0, 1.4, 0.0, 7.0, 0.0, 18.9, 0.0, 7.0],
            [0.0, 4.5, 0.0, 2.5, 0.0, 0.0, 0.0, 1.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 6.1, 0.0, 7.0, 0.0, 6.1, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 19.6, 0.0, 19.6, 0.0, 19.6, 0.0, 19.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 89.0, 0.0, 116.3, 0.0, 0.0, 0.0, 41.0],
            [0.0, 0.0, 0.0, 11.7, 0.0, 0.0, 0.0, 75.0],
            [0.0, 1.4, 0.0, 7.0, 0.0, 18.9, 0.0, 7.0],
            [0.0, 4.5, 0.0, 2.5, 0.0, 0.0, 0.0, 1.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 6.1, 0.0, 7.0, 0.0, 6.1, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 19.6, 0.0, 19.6, 0.0, 19.6, 0.0, 19.6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 89.0, 0.0, 116.3, 0.0, 0.0, 0.0, 41.0],
            [0.0, 0.0, 0.0, 11.7, 0.0, 0.0, 0.0, 75.0],
            [0.0, 1.4, 0.0, 7.0, 0.0, 18.9, 0.0, 7.0],
            [0.0, 4.5, 0.0, 2.5, 0.0, 0.0, 0.0, 1.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        
        for row in range(0, len(self.r_landuse)):
            ch4_emission.reservoir.area_fractions = self.r_landuse[row]
            output_ix: int = 0
            for climate, soil in itertools.product(self.test_climates, self.test_soiltypes):
                ch4_emission.catchment.biogenic_factors.climate = climate
                ch4_emission.catchment.biogenic_factors.soil_type = soil
                try:
                    # Multiplier 10 is a result of km2->ha coefficient 100 and second coefficient 0.001
                    self.assertAlmostEqual(
                        10 * ch4_emission.pre_impoundment(), 
                        ch4_emission_calc[row][output_ix])
                    output_ix += 1
                except AssertionError as e:
                    print("Preimpoundment CH4 emission failed at row %d" % row)
                    raise e


if __name__ == '__main__':
    unittest.main()