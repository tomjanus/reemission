""" """
import unittest
import pathlib
import shutil
import pandas as pd
import geopandas as gpd
from reemission.utils import load_shape, load_toml, get_package_file
from reemission.app_logger import create_logger
from reemission.postprocessing.data_processing import (
    append_data_to_shapes, ExtractFromJSON, TabToShpCopy)


TEST_OUTPUT_FOLDER = './test_output'
log = create_logger(logger_name="results postprocessing test")


class TestHeetReemissionIntegration(unittest.TestCase):
    """ """
    @classmethod
    def setUpClass(cls):
        cls.config = load_toml(get_package_file('config/heet.toml'))
        pathlib.Path(TEST_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up all created shape files after running all tests
        log.info("Clearing the %s outputs folder.", TEST_OUTPUT_FOLDER)
        shutil.rmtree(TEST_OUTPUT_FOLDER, ignore_errors=True)

    def setUp(self):
        # 
        self.shp_folder = pathlib.Path("test_data/merged_shp_files")
        self.data_file = pathlib.Path("test_data/heet_output/heet_outputs.csv")
        self.output_json_file = pathlib.Path(
            "test_data/reemission_output/reemission_output.json")

    def tearDown(self):
        # 
        ...

    def test_heet_outputs_to_shapes(self) -> None:
        """ Create shape files with extra fields with values obtained from the
        tabular HEET output csv file. Assert that the file has been created """
        append_data_to_shapes(
            self.shp_folder, self.data_file, self.config, TEST_OUTPUT_FOLDER)
        self.assertTrue(pathlib.Path.exists(
            pathlib.Path(TEST_OUTPUT_FOLDER, "reservoirs_updated.shp")))
    
    def test_updated_shp_fields(self) -> None:
        """Check that the fields in the shape file `reservoirs_updated.shp`
        contain all mandatory fields defined in the config file"""
        reservoir_updated: gpd.GeoDataFrame = gpd.read_file(
            pathlib.Path(TEST_OUTPUT_FOLDER, "reservoirs_updated.shp"))
        mandatory_fields = self.config['shp_output']['reservoirs']['fields']
        # Trim mandatory fields to 10 characters because only 10 characters
        # are supported by shape files
        mandatory_fields_shp = [field[:10] for field in mandatory_fields]
        fields_intersection = \
            set(mandatory_fields_shp).intersection(set(reservoir_updated.columns))
        # Assert that the fields in the shape file contain all mandatory fields
        self.assertSetEqual(set(fields_intersection), set(mandatory_fields_shp))
    
    def test_extract_reemission_outputs_to_df(self) -> None:
        """Extract data from RE-Emission output file and save it into reservoirs
        shape file"""
        json_extractor = ExtractFromJSON.from_file(self.output_json_file)
        reemission_outputs: pd.DataFrame = json_extractor.extract_outputs()
        data_columns = list(reemission_outputs.columns)
        data_columns.remove("name")
        # Assert if the dataframe constructed from the reemission outputs
        # JSON file contains the required columns
        required_fields = json_extractor.extracted_keys['outputs']
        required_fields.append("tot_em")
        self.assertEqual(data_columns, required_fields)

    def test_move_reemission_outputs_to_shp(self) -> None:
        """ """
        # 1. "Feed" the emission data into the reservoirs shape.
        json_extractor = ExtractFromJSON.from_file(self.output_json_file)
        reemission_outputs: pd.DataFrame = json_extractor.extract_outputs()
        res_shape_file = pathlib.Path(TEST_OUTPUT_FOLDER) / "reservoirs_updated.shp"
        data_columns = list(reemission_outputs.columns)
        data_columns.remove("name")
        re_outputs_to_res = TabToShpCopy(
            shp_data=load_shape(res_shape_file),
            tab_data=reemission_outputs)
        re_outputs_to_res.transfer(
            source_key_column="name", target_key_column="name", 
            fields=data_columns)
        re_outputs_to_res.save_shp(res_shape_file)
        # 2. Assert that the new shape file contains data_columns as fields
        # Trim column names to 10 chars for compatibility with shp file format
        # that only allows 10 characters for the column/field names
        data_columns_shp = [field[:10] for field in data_columns]
        shp_data: gpd.GeoDataFrame = gpd.read_file(res_shape_file)
        fields_intersection = \
            set(data_columns_shp).intersection(set(shp_data.columns))
        # Assert that the fields in the shape file contain all mandatory fields
        self.assertSetEqual(set(fields_intersection), set(data_columns_shp))


if __name__ == '__main__':
    unittest.main()