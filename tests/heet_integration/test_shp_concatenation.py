"""Tests for concatenating shape files that match a glob pattern and saving
the concetenated shape files into a single file

Shape concatenation functionality is used for processing reservoir and catchment
delineations coming as outputs from the HEET reservoir and catchment delineation
tool. The shape files representing individual reservoirs or dams need to be
concatenated into a single shape file (layer) with multiple reservoirs / dams / 
catchments / river fragments for easier post-processing, visualisation and 
merging with additional data, e.g. with the output data from RE-Emission.
"""
import pathlib
import unittest
import shutil
import geopandas as gpd
from reemission.integration.heet.heet_shp_parser import ShpConcatenator

TEST_OUTPUT_FOLDER = './test_output'


#TODO: Test for error handling, e.g. Value Error when nothing to concatenate, etc.


class TestShpConcatenation(unittest.TestCase):
    """Test re-emissions shape concatenation functionality used in a pipe-line
    process for integrating emission outputs with GIS data"""

    @classmethod
    def setUpClass(cls):
        # Set up any necessary resources before all test methods
        cls.shp_file_folders = [pathlib.Path(str_path) for str_path in [
            "./test_data/shp_1", "./test_data/shp_2", 
            "./test_data/empty_folder"]]
        cls.empty_folders = [pathlib.Path("./test_data/empty_folder")]

    @classmethod
    def tearDownClass(cls):
        # Clean up all created shape files after running all tests
        shutil.rmtree(TEST_OUTPUT_FOLDER, ignore_errors=True)

    def setUp(self):
        # Instantiate ShpConcatenetor object before each test
        self.shp_concat = ShpConcatenator()

    def tearDown(self):
        # Clean up any resources after each test method
        ...

    def test_shp_stats(self) -> None:
        """ """
        stats = self.shp_concat.find_in_folders(
            folders=self.shp_file_folders, glob_pattern="R_*.shp")
        self.assertEqual(set(stats["found ids"]), {98, 35, 45, 14})

    def test_shp_stats_empty_folder(self) -> None:
        """ """
        stats = self.shp_concat.find_in_folders(
            folders=self.empty_folders, glob_pattern="R_*.shp")
        self.assertEqual(set(stats["found ids"]), set())

    def test_shp_concatenation(self) -> None:
        """Test shp file concatenation and parsing using demo data"""
        self.shp_concat.find_in_folders(
            folders=self.shp_file_folders, glob_pattern="R_*.shp")
        g_data = self.shp_concat.concatenate()
        g_data.save(folder=pathlib.Path(TEST_OUTPUT_FOLDER), file_name='test.shp')
        test_shape = gpd.read_file(f"{TEST_OUTPUT_FOLDER}/test.shp")
        self.assertEqual(set(test_shape['id']), {98, 35, 45, 14})
    
    def test_shp_concatenation_empty(self) -> None:
        """ """
        self.shp_concat.find_in_folders(
            folders=self.empty_folders, glob_pattern="R_*.shp")
        with self.assertRaisesRegex(ValueError, r"No objects to concatenate"):
            _ = self.shp_concat.concatenate()
        

if __name__ == '__main__':
    unittest.main()