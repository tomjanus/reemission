""" """
import unittest
import pathlib
import shutil
from reemission.utils import load_geojson, load_shape
from reemission.app_logger import create_logger
from reemission.postprocessing.visualise import FoliumOutputMapper


TEST_OUTPUT_FOLDER = './test_output'
log = create_logger(logger_name="visualisation test")


class TestHeetReemissionResultVisualisation(unittest.TestCase):
    """ """
    @classmethod
    def setUpClass(cls):
        # Create a map from existing resources
        cls.reservoirs = load_shape(
            pathlib.Path("test_data/reservoirs_updated.shp"))
        cls.dams_ifc = load_geojson(
            pathlib.Path(
                "../heet_integration/test_data/ifc_db/ifc_test.geojson"))
        pathlib.Path(TEST_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up all created shape files after running all tests
        log.info("Clearing the %s outputs folder.", TEST_OUTPUT_FOLDER)
        shutil.rmtree(TEST_OUTPUT_FOLDER, ignore_errors=True)

    def setUp(self):
        """ """ 
        ...

    def tearDown(self):
        """ """ 
        ...

    def test_folium_map_creation(self) -> None:
        """Test creating interactive maps with folium using FoliumOutputMapper"""
        mapper = FoliumOutputMapper(self.reservoirs, self.dams_ifc)
        mapper.create_map()
        mapper.save_map(
            pathlib.Path(TEST_OUTPUT_FOLDER) / "index.html", show=False)


if __name__ == '__main__':
    unittest.main()