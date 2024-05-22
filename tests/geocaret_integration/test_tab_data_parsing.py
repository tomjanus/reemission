"""Tests for collating (merging) multiple csv files with output parameters from HEET
and adding additional (supplementary) data. Currently, the supplementary data addition
is made bespoke for the Myanmar case study and is loaded from the IFC database of dams
using class `SuppDataMyanmar`. 
TODO: Makse sure SuppDataMyanmar follows a pre-defined interface such that other supplementary
    data types can be merged with the tabular data in other projects.

NOTE: Fields `c_treatment_factor` and `c_landuse_intensity` are currently not output
    by HEET but are required by RE-EMISSION. They have to be added by hand. By default
    we assume "primary(mechanical)" treatment and "low intensity" landuse intensity.
"""
import pathlib
import unittest
import shutil
import pandas as pd
import geopandas as gpd
from reemission.utils import get_package_file, load_toml
from reemission.app_logger import create_logger
from reemission.integration.geocaret.geocaret_tab_parser import (
    HeetOutputReader, SuppDataMyanmar)


DEFAULT_HEET_OUTPUT_FILE = "output_parameters.csv"
TEST_OUTPUT_FOLDER = './test_output'
log = create_logger(logger_name="test_tab_data_parsing")


class TestTabDataParsing(unittest.TestCase):
    """ """
    @classmethod
    def setUpClass(cls):
        """Load geojson file and save it to a shape file before running the tests.
        The reason for this is that the tester SuppDataMyanmar class loads shp files
        but we want to avoid keeping binary files in the git repository and thus
        we keep the data in the geojson txt format."""
        ifc_db = gpd.read_file(pathlib.Path("test_data/ifc_db/ifc_test.geojson"))
        pathlib.Path(TEST_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        ifc_db.to_file(pathlib.Path(TEST_OUTPUT_FOLDER)/"ifc_test.shp")

        cls.geocaret_output_1 = pathlib.Path("./test_data/shp_1/") / DEFAULT_HEET_OUTPUT_FILE
        cls.geocaret_output_2 = pathlib.Path("./test_data/shp_2/") / DEFAULT_HEET_OUTPUT_FILE
        cls.geocaret_output_3 = pathlib.Path("./test_data/empty_folder/") / DEFAULT_HEET_OUTPUT_FILE

    @classmethod
    def tearDownClass(cls):
        # Clean up all created shape files after running all tests
        log.info("Clearing the %s outputs folder.", TEST_OUTPUT_FOLDER)
        shutil.rmtree(TEST_OUTPUT_FOLDER, ignore_errors=True)

    def setUp(self):
        # 
        ...

    def tearDown(self):
        # 
        ...

    def test_tab_data_parsing(self) -> None:
        """Parse tabular output data from HEET generated for demo purposes"""
        # Get the IFC database of dams (providing supplementary data)
        # Read the tabular output files
        output_reader = HeetOutputReader(
            file_paths=[self.geocaret_output_1, self.geocaret_output_2])
        geocaret_output = output_reader.read_files()
        geocaret_output.remove_duplicates(on_column="id")
        # Load supplementary data from the ifc database
        sup_data = SuppDataMyanmar.from_ifc_db(
            pathlib.Path(TEST_OUTPUT_FOLDER)/"ifc_test.shp")
        geocaret_output.handle_existing_reservoirs(sup_data)
        # Get the list of mandatory columns from config file
        tab_data_config = load_toml(
            get_package_file("./config/geocaret.toml"))['tab_data']
        geocaret_output.filter_columns(
            mandatory_columns=tab_data_config['mandatory_fields'],
            optional_columns=tab_data_config['unused_inputs'])
        # Add missing columns containing information about treatment factor and
        # landuse intensity that are not currently present in HEET
        # geocaret_output.set_index("id")
        geocaret_output.add_column(
            column_name="c_treatment_factor", default_value="primary (mechanical)")
        geocaret_output.add_column(
            column_name="c_landuse_intensity", default_value="low intensity")       
        # Save the combined and parsed outputs csv file
        test_output_file: pathlib.Path = pathlib.Path(TEST_OUTPUT_FOLDER)/"all_outputs.csv"
        geocaret_output.to_csv(test_output_file)
        # Read the saved csv file and assert that all IDs have been processed
        output_data = pd.read_csv(test_output_file)
        self.assertEqual(set(output_data['id']), {98, 35, 45, 14})


if __name__ == '__main__':
    unittest.main()