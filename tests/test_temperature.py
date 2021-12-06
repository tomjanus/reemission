""" Tests for the Emission classes """
import sys
sys.path.append("..")
import unittest

from src.emissions.temperature import MonthlyTemperature
from src.emissions.reservoir import Reservoir


class TestEmissions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.good_temperatures = \
            [10.56, 11.99, 15.46, 18.29, 20.79, 22.09, 22.46,
             22.66, 21.93, 19.33, 15.03, 11.66]

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.temperature = MonthlyTemperature(self.good_temperatures)

    def tearDown(self):
        pass

    def test_sample(self):
        print(self.temperature.eff_temp())
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
