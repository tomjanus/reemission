""" Tests for the Input and Inputs classes """
import sys
sys.path.append("..")
import unittest
from src.emissions.input import Input, Inputs
from types import SimpleNamespace
from src.emissions.utils import read_table


class TestInput(unittest.TestCase):
    """ Test class for testing Input class functionalities """
    input = None

    @classmethod
    def setUpClass(cls):
        cls.input_file = 'test_data/inputs.json'
        cls.test_name = "Reservoir 2"
        cls.test_dict = {
            "monthly_temps": [13.56, 14.99, 18.46, 21.29, 23.79, 25.09,
                              25.46, 25.66, 24.93, 22.33, 18.03, 14.66],
            "year_vector": [1, 5, 10, 20, 30, 40, 50, 65, 80, 100],
            "gasses": ["co2", "ch4"],
            "catchment":
            {
                "runoff": 3000,
                "area": 102203.0,
                "population": 38463,
                "area_fractions": [0.0, 0.2, 0.0, 0.0, 0.0, 0.01092, 0.11996,
                                   0.667257],
                "slope": 4.0,
                "precip": 5000.0,
                "etransp": 200.0,
                "soil_wetness": 100.0,
                "biogenic_factors":
                {
                    "biome": "MEDFORESTS",
                    "climate": "SUBTROPICAL",
                    "soil_type": "ORGANIC",
                    "treatment_factor": "PRIMARY",
                    "landuse_intensity": "HIGH"
                }
            },
            "reservoir": {
                "volume": 17663812,
                "area": 4.56470,
                "max_depth": 52.0,
                "mean_depth": 17.6,
                "area_fractions": [0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.4],
                "soil_carbon": 30.228
            }
        }

    def setUp(self) -> None:
        self.input = None

    def test_read_yaml(self) -> None:
        """ Test reading of yaml files """
        table = read_table(
            '../data/emissions/McDowell/landscape_TP_export.yaml')
        table_ns = SimpleNamespace(**table)
        self.assertIsInstance(table_ns.biome, dict)

    def test_input_fromfile(self) -> None:
        """ Test input object initialization """
        # Check if instiating with wrong key does not raise an error
        reservoir_name = "Reservoir 3"
        self.input = Input.fromfile(file=self.input_file,
                                    reservoir_name=reservoir_name)
        self.assertIsNone(self.input.data)

        # Assert that instantiating with the key creates a dict inside Input
        reservoir_name = "Reservoir 2"
        self.input = Input.fromfile(file=self.input_file,
                                    reservoir_name=reservoir_name)
        self.assertIsInstance(self.input.data, dict)
        self.assertEqual(self.input.name, reservoir_name)

    def test_input_from_dict(self) -> None:
        """ Test input object initialization from data in dict format """
        self.input = Input(self.test_name, self.test_dict)
        self.assertIsInstance(self.input.data, dict)

    def test_retrieve_reservoir_data(self) -> None:
        """ Test reservoir data retrieval """
        self.input = Input(self.test_name, self.test_dict)
        self.assertIsNotNone(self.input.reservoir_data)

    def test_retrieve_catchment_data(self) -> None:
        """ Test catchment data retrieval """
        self.input = Input(self.test_name, self.test_dict)
        self.assertIsNotNone(self.input.catchment_data)

    def test_retrieve_emission_factors(self) -> None:
        """ Test emission factors (list) data retrieval """
        self.input = Input(self.test_name, self.test_dict)
        self.assertIsNotNone(self.input.gasses)

    def test_retrieve_year_vector(self) -> None:
        """ Test retrieval of the vector of years (for calculation of
            emission profiles) """
        self.input = Input(self.test_name, self.test_dict)
        self.assertIsNotNone(self.input.year_vector)

    def test_retrieve_monthly_temps(self) -> None:
        """ Test retrieval of monthly temperatures (12x1 vector of floats) """
        self.input = Input(self.test_name, self.test_dict)
        self.assertIsNotNone(self.input.monthly_temps)


class TestInputs(unittest.TestCase):
    """ Test class for testing Inputs class functionalities """
    inputs = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.input_file = 'test_data/inputs.json'
        cls.new_reservoir = "Reservoir 3"
        cls.new_reservoir_data = {
            "monthly_temps": [13.56, 14.99, 18.46, 21.29, 23.79, 25.09,
                              25.46, 25.66, 24.93, 22.33, 18.03, 14.66],
            "year_vector": [1, 5, 10, 20, 30, 40, 50, 65, 80, 100],
            "emission_factors": ["co2", "ch4"],
            "catchment":
            {
                "runoff": 3000,
                "area": 102203.0,
                "population": 38463,
                "area_fractions": [0.0, 0.2, 0.0, 0.0, 0.0, 0.01092, 0.11996,
                                   0.667257],
                "slope": 4.0,
                "precip": 5000.0,
                "etransp": 200.0,
                "soil_wetness": 100.0,
                "biogenic_factors":
                {
                    "biome": "MEDFORESTS",
                    "climate": "SUBTROPICAL",
                    "soil_type": "ORGANIC",
                    "treatment_factor": "PRIMARY",
                    "landuse_intensity": "HIGH"
                }
            },
            "reservoir": {
                "volume": 17663812,
                "area": 4.56470,
                "max_depth": 52.0,
                "mean_depth": 17.6,
                "area_fractions": [0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.4],
                "soil_carbon": 30.228
            }
        }

    def setUp(self) -> None:
        self.inputs = None

    def test_inputs_fromfile(self) -> None:
        """ Test input object initialization """
        self.inputs = Inputs.fromfile(file=self.input_file)
        self.assertIsInstance(self.inputs.inputs, dict)

    def test_add_new_data(self) -> None:
        """ Initialize inputs from file and add a new reservoir dict """
        self.inputs = Inputs.fromfile(file=self.input_file)
        self.inputs.add_input(
            input_dict={self.new_reservoir: self.new_reservoir_data})
        self.assertEqual(len(self.inputs.inputs), 3)


if __name__ == '__main__':
    unittest.main()
