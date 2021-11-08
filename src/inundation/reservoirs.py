""" Module for calculation of inundated reservoir area
    using digital elevation map and dam heights and locations
    Designed by Aung Kyaw Kyaw, 01/11/2021
"""
# Python packages
import os
import copy
from pathlib import Path
import json
from typing import Optional, TypeVar
# Numpy, pandas, geopandas
import numpy as np
import pandas as pd
import geopandas as gpd
# computer software library for reading and writing raster and vector geospatial
# data formats
from osgeo import gdal
# Parallel computing library that scales the existing Python ecosystem
import dask
from dask.distributed import Client
from dask import delayed
# Support for labelled multi-dimensional arrays
import xarray as xr
# rioxarray extends xarray with the rio accessor, used to clip, merge, and
# reproject rasters
import rioxarray as rxr
# Package for map visualisations
import folium
from folium.plugins import MarkerCluster
# GIS package for reading, writing, and converting between various common
# coordinate reference system
# (CRS) string and data source formats
import pycrs
# Fiona reads and writes spatial data files
from fiona.crs import from_epsg
# Set-theoretic analysis and manipulation of planar features
from shapely.geometry import box
# Raster data library with high-level bindings to GDAL
import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.plot import show, show_hist
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
# advanced geospatial data analysis platform with 480 tools, also for
# hydrologic analysis
from whitebox.whitebox_tools import WhiteboxTools

# Define new types for checking with typing
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
GeopandasDataFrame = TypeVar('geopandas.geodataframe.GeoDataFrame')
