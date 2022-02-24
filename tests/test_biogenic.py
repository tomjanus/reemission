""" Tests for BiogenicFactors """
import sys
sys.path.append("..")
import unittest
from src.emissions.biogenic import BiogenicFactors
from src.emissions.constants import (Biome, Climate, SoilType, TreatmentFactor,
                                     LanduseIntensity)


class TestBiogenic(unittest.TestCase):
    """ Test functionality of BiogenicFactors class """

    @classmethod
    def setUpClass(cls):
        cls.data = {
            "biome": "TROPICALMOISTBROADLEAF",
            "climate": "TROPICAL",
            "soil_type": "MINERAL",
            "treatment_factor": "NONE",
            "landuse_intensity": "LOW"}

    def setUp(self):
        pass

    def test_read_from_dict(self):
        """ Test initialization of BiogenicFactors from dictionary """
        biogenic_factors = BiogenicFactors.fromdict(self.data)
        self.assertEqual(
            biogenic_factors.biome.value, Biome.TROPICALMOISTBROADLEAF.value)
        self.assertEqual(
            biogenic_factors.climate.value,
            Climate.TROPICAL.value)
        self.assertEqual(
            biogenic_factors.soil_type.value, SoilType.MINERAL.value)
        self.assertEqual(
            biogenic_factors.treatment_factor.value,
            TreatmentFactor.NONE.value)
        self.assertEqual(
            biogenic_factors.landuse_intensity.value,
            LanduseIntensity.LOW.value)

    def test_convert_to_dict(self):
        """ Test conversion to a dictionary """
        # Initialize BiogenicFactors from dictionary and then convert
        # the data to a dictionary and check if both are equal
        biogenic_factors = BiogenicFactors.fromdict(self.data)
        self.assertEqual(self.data, biogenic_factors.todict())


if __name__ == '__main__':
    unittest.main()
