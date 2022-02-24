""" Utility functions for inundation area calculations"""
# Python packages
import os
#import gdal
from osgeo import gdal 
from typing import Optional, TypeVar, List
import geopandas as gpd

GeopandasDataFrame = TypeVar('geopandas.geodataframe.GeoDataFrame')


def specify_max_workers(n_work: Optional[int] = None) -> int:
    """ Return number of workers for parallel computing """
    if n_work is None:
        n_work = os.cpu_count()
    return n_work


def shapefile2geojson(infile: str, outfile: str,
                      proj: str = "EPSG:4326") -> None:
    """ Translate a shapefile to GEOJSON file using GDAL """
    options = gdal.VectorTranslateOptions(format="GeoJSON",
                                          dstSRS=proj)
    gdal.VectorTranslate(outfile, infile, options=options)


def open_geo(file_path: str) -> GeopandasDataFrame:
    """ Read shape file"""
    return gpd.read_file(file_path)


def remove_file(file_path: str) -> None:
    """Remove file_path if exists"""
    if os.path.exists(file_path):
        os.remove(file_path)


def find_files(path: str, ext: str) -> List[str]:
    """ Find files in path that have the specified extension """
    # Find all files in path matching exension
    files = [file for file in os.listdir(path) if ext in
             os.path.splitext(file)[1]]
    # Files with the full path
    paths = [os.path.join(path, file) for file in files]
    return paths
