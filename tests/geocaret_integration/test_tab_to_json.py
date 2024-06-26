""" """
import unittest
import pathlib
import shutil
from reemission.utils import load_json
from reemission.app_logger import create_logger
from reemission.integration.geocaret.geocaret_tab_to_json import (
    TabToJSONConverter, LegacySavingStrategy)


TEST_OUTPUT_FOLDER = './test_output'
log = create_logger(logger_name="test_tab_to_json")


class TestTabToJson(unittest.TestCase):
    """ """
    @classmethod
    def setUpClass(cls):
        pathlib.Path(TEST_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        cls.saving_strategy = LegacySavingStrategy()

    @classmethod
    def tearDownClass(cls):
        # Clean up all created shape files after running all tests
        log.info("Clearing the %s outputs folder.", TEST_OUTPUT_FOLDER)
        shutil.rmtree(TEST_OUTPUT_FOLDER, ignore_errors=True)

    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_tab_to_json(self) -> None:
        """ """
        geocaret_output =pathlib.Path("test_data/all_geocaret_outputs.csv")
        converter = TabToJSONConverter(geocaret_output, self.saving_strategy)
        reemission_input_file_path = \
            pathlib.Path(TEST_OUTPUT_FOLDER)/"reemission_input.json"
        converter.to_json(reemission_input_file_path)
        # Load json and check if some of the fields read from 'all_geocaret_outputs.csv' exist
        reemission_input_data = load_json(reemission_input_file_path)
        self.assertEqual(
            list(reemission_input_data.keys()), 
            ['Kabaung', 'Kinda', 'Paung Laung (upper)', 'Chipwi'])

if __name__ == '__main__':
    unittest.main()