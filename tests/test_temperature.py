""" Tests for the Emission classes """
import unittest
from reemission.temperature import MonthlyTemperature
from reemission.reservoir import Reservoir


class TestTemperature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_temperatures = \
            [10.56, 11.99, 15.46, 18.29, 20.79, 22.09, 22.46,
             22.66, 21.93, 19.33, 15.03, 11.66]

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.temperature = MonthlyTemperature(self.test_temperatures)

    def tearDown(self):
        pass

    # TODO: add tests
    def test_coldest(self):
        self.assertEqual(
            self.temperature.coldest, min(self.temperature.temp_profile))


if __name__ == '__main__':
    unittest.main()
