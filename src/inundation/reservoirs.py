""" Module for calculation of inundated reservoir area
    using digital elevation map and dam heights and locations
    Methodology and coding by Aung Kyaw Kyaw, 01/11/2021
    Refactored by T. Janus and Aung Kyaw Kyaw
"""

# Currently known issues 09-Nov-2021
# Whitebox which is used in here as the raster processing toolbox
# https://www.whiteboxgeo.com/manual/wbt_book/
# doesn't output errors e.g. due to lack of memory
# Therefore, it is not possible to catch errors within Python and act upon them
# It is often the case that the file that is being processed and runs into
# some kind of problems, is not output and the number of outputs is less than
# the number of inputs (input files). It is most likely due to the way
# processing is carried out in Whitebox in Rust. The worker that runs out of
# memory is dropped while other (active) workers continue to produce results.

# Python packages
import os
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Optional, TypeVar, Dict, List, Any, Union
import numpy as np
import pandas as pd
import geopandas as gpd
# Parallel computing library that scales the existing Python ecosystem
import dask
from dask.distributed import Client
from dask import delayed
import xarray
# rioxarray extends xarray with the rio accessor, used to clip, merge, and
# reproject rasters
import rioxarray as rxr
# GIS package for reading, writing, and converting between various common
# coordinate reference system
# (CRS) string and data source formats
import pycrs
# Raster data library with high-level bindings to GDAL
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling
# advanced geospatial data analysis platform with 480 tools, also for
# hydrologic analysis
from whitebox.whitebox_tools import WhiteboxTools

from abc import ABC, abstractmethod
from .utils import remove_file, specify_max_workers, find_files
from .exceptions import NotRasterException, NotVectorException
from .error_retry import retry

# Define new types for checking with typing
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
GeopandasDataFrame = TypeVar('geopandas.geodataframe.GeoDataFrame')
RasterioRaster = TypeVar('rasterio.io.DatasetReader')


def find_pointers(dem_names: List[str], pointer_path: str) -> List[str]:
    # Find files matching the names of files in a list file_names
    # Return a list of paths to pointer files matching dems
    pointer_files = []
    for dem_name in dem_names:
        for pointer_file in os.listdir(pointer_path):
            pointer_name, ext = os.path.splitext(pointer_file)
            _str = ''
            _container = []

            for i in pointer_name:
                if len(_container) < 2:
                    if i == '_':
                        _container.append(i)
                    _str = _str + i

            if _str in dem_name:
                matching_pointer = os.path.join(pointer_path, pointer_file)
                pointer_files.append(matching_pointer)
    try:
        assert len(pointer_files) == len(dem_names)
    except AssertionError:
        return None
    return pointer_files

# TODOs:
# Add a flag for either sequential or parallel execution with Dask


@dataclass
class Layer(ABC):
    path: str
    name: str = ""
    _data: Optional[Any] = None

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def data(self) -> Any:
        pass


@dataclass
class Collection(ABC):
    """ Base Collection Class """
    collection: List[Layer] = field(default_factory=list)

    @abstractmethod
    def from_path(self, path: str) -> None:
        pass

    @property
    def names(self) -> List[str]:
        """ Get layer names """
        return [layer.name for layer in self.collection]

    @property
    def paths(self) -> List[str]:
        """ Get layer paths """
        return [layer.path for layer in self.collection]

    def to_dict(self) -> Dict[str, str]:
        """ Get a dictionary with names as keys and paths as values """
        return {layer.name: layer.path for layer in self.collection}


@dataclass
class Vector(Layer):
    """ Class for loading and storing a vector layer """
    _data: Optional[GeopandasDataFrame] = None

    def __post_init__(self):
        """ Set name from path, if name not provided during initialization """
        if not self.name:
            dir, file = os.path.split(self.path)
            base, ext = os.path.splitext(os.path.basename(file))
            if ext != '.shp':
                raise NotVectorException(ext)
            else:
                self.name = base

    def load(self) -> None:
        """ Load data from path with Geopandas """
        self._data = gpd.read_file(self.path)

    @property
    def data(self) -> GeopandasDataFrame:
        """ Defer loading data until data needs to be accessed """
        if self._data is None:
            self.load()
        return self._data

    def get_coordinates(self) -> list:
        """Parse features from GeoDataFrame so that they can be read by
           rasterio """
        if self._data is None:
            self.load()
        return [json.loads(self._data.to_json())['features'][0]['geometry']]

    # Can use with list of points which is even faster but the Whitebox
    # algorithm neglects the overlapping area.
    # There is overhead cost for each individual points
    def shp_individual(self, output_path: str) -> None:
        """ Get ID-ed shapefile for each dam (to be used later to find catchment
            areas via individual points)
            Function assumes that 'ID' column is present in the shape file
        """
        Path(output_path, self.name + 'fill_').mkdir(parents=True,
                                                     exist_ok=True)
        df = self.data
        for ID in df.ID:
            df_ = df[df.ID == ID]
            output_file = os.path.join(
                output_path, self.name + 'fill_', str(ID) + '.shp')
            remove_file(output_file)
            df_.to_file(output_file)

    def to_single_polygon(self, output_path: str) -> None:
        """ Get single polygon , get rid of islands
            This processing step may not be needed depending on the
            the outcome of an earlier hydrological correction/fill
            depressions step
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name + '.shp')
        remove_file(output_file)
        df_test = self.data
        df_test['new_column'] = 0
        df_test = df_test.dissolve(by='new_column')
        df_test.to_file(output_file)
        # df_test.dissolve(by='new_column').to_file(output_file)


@dataclass
class Dams(Vector):
    """ Class for storing and manipulating geodataframes with information
        about dams """

    @staticmethod
    def find_files(path: str, extension: str) -> List[str]:
        """ Return a list of files in path which have an extension
            given in parameter extension """
        list_of_files = []
        for file in os.listdir(path):
            base, ext = os.path.splitext(file)
            if ext == extension:
                file_with_path = os.path.join(path, file)
                list_of_files.append(file_with_path)
        return list_of_files

    # Potential problem in this function (write error handling) is when:
    # No dams found and potentially unable write None to shape file
    def divide_to_zones(self, path_to_zone_shapes: str,
                        output_path: str) -> None:
        """ Snaps dams into zones defined with separate shape files in
            path_to_zone_shapes and returns shape files of dams split
            into zones """
        dams_df = self.data
        # Create output path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        list_of_zones = self.find_files(path=path_to_zone_shapes,
                                        extension='.shp')
        for zone in list_of_zones:
            # Get the name of the zone
            _, filename = os.path.split(zone)
            zone_name = os.path.splitext(filename)[0]
            # Open the zone (shapefile) in geopandas
            zone_df = gpd.read_file(zone)
            # For each dam in data check if the dam is within the zone
            found_dams = []
            for dam in dams_df.itertuples():
                dam_index = dam.Index
                dam_location = dam.geometry
                if zone_df.geometry.values.contains(dam_location)[0]:
                    found_dams.append(dam_index)
            dams_in_zone = dams_df.loc[found_dams]
            # Write to shape file
            dam_shape_file = os.path.join(output_path,
                                          'dam_' + zone_name + '_fill_.shp')
            remove_file(dam_shape_file)
            dams_in_zone.to_file(dam_shape_file)


@dataclass
class Dem(Layer):
    """ Provide facility for digital elevation models """
    _data: Optional[RasterioRaster] = None

    def __post_init__(self):
        """ Set name from path, if name not provided during initialization """
        if not self.name:
            dir, file = os.path.split(self.path)
            base, ext = os.path.splitext(os.path.basename(file))
            if ext != '.tif':
                raise NotRasterException(ext)
            else:
                self.name = base

    def load(self) -> None:
        """ Load raster from path with rasterio """
        try:
            self._data = rasterio.open(self.path)
        except RasterioIOError as e:
            print(e, "Data could not be loaded")

    def load_array(self) -> Union[xarray.Dataset, xarray.DataArray,
                                  List[xarray.Dataset]]:
        """Load raster to xarray"""
        return rxr.open_rasterio(self.path)

    @property
    def data(self) -> RasterioRaster:
        """ Defer loading data until data needs to be accessed """
        if self._data is None:
            self.load()
        return self._data

    def resize(self, new_path: str, upscale_factor: float = 0.5,
               verbose: bool = False) -> None:
        """ Resize digital elevation model by an upscale factor
            values < 1 indicate that the resolution will be reduced """
        dataset = self.data
        profile = dataset.profile
        # resample data to target
        target = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ), resampling=Resampling.bilinear)
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / target.shape[-1]),
            (dataset.height / target.shape[-2])
        )
        new_height = int(dataset.height * upscale_factor)
        new_width = int(dataset.width * upscale_factor)
        profile.update(transform=transform, driver='GTiff',
                       height=new_height, width=new_width,
                       crs=dataset.crs)
        remove_file(new_path)
        with rasterio.open(new_path, 'w', **profile) as dst:
            dst.write(target)
        if verbose:
            print('Original DEM size: {}, Resampled DEM size: {}'.format(
                  dataset.shape, target.shape))

    def clip_raster_to_polygon(self, output_path: str) -> None:
        """ Clip raster to a polygon """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, output_file)
        remove_file(output_file)
        wbt = WhiteboxTools()

        wbt.clip_raster_to_polygon(self.path, output=output_path)

    def clip_to_polygon(self, zone: Vector, clipped_path: str,
                        parallel: bool = False,
                        copy_raster: bool = True) -> None:
        """
        Clip raster to a zone. Zone needs to be a vector polygon shape
        Needs to receive paths instead of objects because otherwise
        parallelization cannot work as rasterio objects cannot be serialized
        For consistency, the zone geopandas object is also created inside
        this function and hence, the function receives path to the zone
        shape file, not the object itself
        Source of info:
        https://autogis-site.readthedocs.io/en/latest/notebooks/Raster/clipping-raster.html
        """
        # TODO: The problem with parallelizing clipping is that in order to
        #       enable Dask paralellize, we need to open the raster in each
        #       process. This creates overhead. It is much faster to use
        #       a single data set (if parallel = True) but this leads to
        #       random access/read errors. It is possible to catch the error
        #       and wait until read can be resumed. Other possibility is lower-
        #       level parallelization with DASK
        if parallel and copy_raster:
            raster = rasterio.open(self.path)
        if parallel and not copy_raster:
            raster = self.data
        if not parallel:
            raster = self.data

        # Read epsg code; Need internet for pycrs information retrieval.
        epsg_code = int(raster.crs.data['init'][5:])
        epsg_string = pycrs.parse.from_epsg_code(epsg_code).to_proj4()
        # Obtain coordinates from the zone vector layer
        coords = zone.get_coordinates()
        # Crop raster
        out_img, out_transform = mask(
            dataset=raster, shapes=coords, crop=True)
        out_meta = raster.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": epsg_string})
        # Create path and write cropped raster in tif format
        Path(clipped_path).mkdir(parents=True, exist_ok=True)
        out_tif = os.path.join(clipped_path, zone.name + '.tif')
        remove_file(out_tif)
        # Write clipped dem as tif file
        with rasterio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)

    # TODO: WORKS INTERMITTENTLY. TEST WITH:
    # retry params and parallel = True and copy_raster = False
    # parallel = True and copy_raster = True
    # parallel = True copy_raster = False and starting a dask client before
    # running the script
    # Depending on the configuration, the methods leads to errors, i.e.
    # exceeded memory or rasterio is unable to access shared raster (dem) file

    @retry(RasterioIOError, tries=4)
    def clip_to_polygons(self, zones: Collection,
                         clipped_path: str,
                         parallel: bool = True,
                         copy_raster: bool = False) -> None:
        """ Implement clip_to_polygon to clip raster to multiple polygons
            (zones) using parallelization feature in Dask """
        # Prepare the list of parallel task to be executed by dask
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)
        compute_list = []
        for zone in zones.collection:
            process = delayed(self.clip_to_polygon)(zone=zone,
                                                    clipped_path=
                                                    clipped_path,
                                                    parallel=parallel,
                                                    copy_raster=copy_raster)
            compute_list.append(process)
        # Compute DEM clipping to basins/polygons with Dask
        _ = dask.compute(*compute_list)

    # TODO: All these processing steps produce (intermediary) raster files
    # that are saved to disk. This reduces disk space and incurs computational
    # time for saving those files to disk. An alternative would be to save in
    # memory but would possibly lead to out-of-memory errors. Other altenatives?
    # e.g. pickling?
    # IT IS PROBABLY BEST TO SPLIT FILL AND BREACH INTO SEPARATE FILL AND
    # BREACH METHODS
    # Arent fill and breach equivalent? If so, one could choose one or another

    def fill(self, output_path: str) -> None:
        """
        Fill dem to enforce flow direction (i.e., make it hydrologically
        correct)
        """
        wbt = WhiteboxTools()
        # Make paths to store filled and breached versions of DEMs
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(
            output_path, self.name + "_filled.tif")
        # Remove existing file if exist
        remove_file(output_file)
        wbt.fill_single_cell_pits(
            dem=self.path,
            output=output_file)

    def breach(self, output_path: str, dist: int = 5,
               fill: bool = True) -> None:
        """
        Breach DEM to enforce flow direction (i.e., make it hydrologically
        correct)
        """
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(
            output_path, self.name + "_breached.tif")
        # Remove existing file if exist
        remove_file(output_file)
        wbt.breach_depressions_least_cost(
            dem=self.path,
            output=output_file,
            dist=dist,
            fill=fill)

    def fill_depression(
            self, output_path: str, flat_increment: float = 0.001) -> None:
        """ Efficient and quick DEM filling (replace fill + breach steps by the
            two functions above)"""
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(
            output_path, self.name + "_fill_depression.tif"
        )
        remove_file(output_file)
        wbt.fill_depressions(
            dem=self.path, output=output_file, flat_increment=flat_increment)

    def accumulate(self, output_path: str) -> None:
        """Perform flow accumulation, necessary for stream extraction"""
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name + '_acc.tif')
        remove_file(output_file)
        # Perform flow accumulation with whitebox
        wbt.d8_flow_accumulation(
            i=self.path,
            output=output_file,
            log=False
        )

    def d8_pointer(self, output_path: str) -> None:
        """ Generate a flow pointer grid using the simple D8
            (O'Callaghan and Mark, 1984) algorithm """
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name + '_d8.tif')
        remove_file(output_file)
        wbt.d8_pointer(
            dem=self.path,
            output=output_file)

    def extract_streams(self, output_path: str, stream_order: int) -> None:
        """ Extract, or map, the likely stream cells from an input
            flow-accumulation image """
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name + '_stream.tif')
        remove_file(output_file)
        wbt.extract_streams(flow_accum=self.path,
                            output=output_file,
                            threshold=stream_order)

    def stream_to_vec(self, d8_flow_pointer: str, output_path: str) -> None:
        """ Convert a raster stream file into a vector file.
            Requires: 1) the name of the raster streams file
                      2) the name of the D8 flow pointer file
                      3) the name of the output vector file.
        """
        streams = self.path
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name + '_stream.shp')
        remove_file(output_file)
        wbt.raster_streams_to_vector(streams=streams,
                                     d8_pntr=d8_flow_pointer,
                                     output=output_file)

    def snap_pour_pt(self, pt_zone: Vector, output_path: str,
                     snap_dist: float = 0.6) -> None:
        """snapp the existing dam locations to nearest streams"""
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        out_shp = os.path.join(output_path, pt_zone.name + '_fill_.shp')
        remove_file(out_shp)
        # Careful with the usage of the below method. Know the units of your
        # data
        wbt.jenson_snap_pour_points(pour_pts=pt_zone.path,
                                    streams=self.path,
                                    output=out_shp,
                                    snap_dist=snap_dist)

    def watershed_individual(self, output_path: str,
                             snapped_pts: Vector) -> None:
        """ Find upper catchment/watershed based on a snapped point or list of
            snapped points
            If using the list of points at once, please set snapped_pts as
            zone-based pts rather than individual points"""
        Path(output_path, self.name+'_watershed').mkdir(
            parents=True, exist_ok=True)
        wbt = WhiteboxTools()
        output_file = os.path.join(output_path, self.name+'_watershed',
                                   snapped_pts.name+'.tif')
        remove_file(output_file)
        wbt.watershed(d8_pntr=self.path,
                      pour_pts=snapped_pts.path,
                      output=output_file)

    def raster_to_polygon(self, output_path: str) -> None:
        """vectorize ID-ed watersheds"""
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name+'.shp')
        remove_file(output_file)
        wbt.raster_to_vector_polygons(
            self.path, output=output_file, callback=None)

    def clip_raster_to_polygon(
            self, output_path: str, polygon: Vector) -> None:
        """ Clip raster to polygon
            Similar to clip_to_polygon.
            Introduced here since the former function seems slower """
        wbt = WhiteboxTools()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, polygon.name+'.tif')
        remove_file(output_file)
        wbt.clip_raster_to_polygon(
            self.path, polygons=polygon.path, output=output_file)

    def clip_raster_to_polygons(
            self, output_path: str, polygons: List[Vector],
            parallel: bool = True) -> None:
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)
        if not parallel:

            for vector in polygons:
                self.clip_raster_to_polygon(
                    output_path=output_path, polygon=vector)
        else:
            compute_list = []
            for vector in polygons:
                process = delayed(self.clip_raster_to_polygon)(
                    output_path=output_path, polygon=vector)
                compute_list.append(process)
            _ = dask.compute(*compute_list)

    def extract_inundated_area(
            self, gdf: Vector, output_path: str, output_path2: str) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, self.name+'_values.tif')
        remove_file(output_file)
        Path(output_path2).mkdir(parents=True, exist_ok=True)
        output_file2 = os.path.join(output_path2, self.name+'.tif')
        remove_file(output_file2)

        dem_data = self.load_array()
        result = dem_data.where(
            dem_data <= (
                gdf.data[gdf.data.ID == int(self.name)].DAM_HEIGHT.values +
                dem_data.min().values), drop=True)
        result.rio.to_raster(output_file, masked=False, dtype="uint16")

        """ The following second part is used to simpify raster-to-polygon
            process """
        result = dem_data.where(
            ~(dem_data <= (
                gdf.data[gdf.data.ID == int(self.name)].DAM_HEIGHT.values +
                dem_data.min().values)), other=100)
        result = result.where(result == 100, other=np.nan)
        result.rio.to_raster(output_file2, masked=False, dtype="uint16")


class VectorCollection(Collection):
    """ Collection Class for Dem Layers """
    collection: List[Vector] = field(default_factory=list)

    def from_path(self, path: str) -> None:
        """ Initialize collection from path instead of explicitly by providing
            a list of Dem objects """
        self.collection = []
        files = find_files(path=path, ext='.shp')
        self.collection = [Vector(file) for file in files]

    # can use with list of points which is even faster but the Whitebox
    # algorithm neglects the overlapping area.
    # There is overhead cost for each individual points
    def shp_individuals(self, output_path: str) -> None:
        """ Get ID-ed shapefile for each dam (to be used later in catchment
            area finding via individual points) """
        for vector in self.collection:
            vector.shp_individual(output_path)

    def to_single_polygons(self, output_path: str,
                           parallel: bool = True) -> None:
        """ Merge polygons to a single polygon """
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for vector in self.collection:
                vector.to_single_polygon(output_path=output_path)
        else:
            compute_list = []
            for vector in self.collection:
                process = delayed(vector.to_single_polygon)(
                    output_path=output_path)
                compute_list.append(process)
            _ = dask.compute(*compute_list)

    def merge_vectors(self, output_path: str,
                      name: str = 'combined_shape.shp') -> None:
        """ Combine  shapefiles """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, name)
        remove_file(output_file)
        gdf = pd.concat([
            gpd.read_file(shp)
            for shp in self.paths
        ]).pipe(gpd.GeoDataFrame)
        # change projection
        gdf = gdf.set_crs('epsg:4326')
        gdf.to_file(output_file)


class DemCollection(Collection):
    """ Collection Class for Dem Layers """
    collection: List[Dem] = field(default_factory=list)

    def from_path(self, path: str):
        """ Initialize collection from path instead of explicitly by providing
            a list of Dem objects """
        self.collection = []
        files = find_files(path=path, ext='.tif')
        self.collection = [Dem(file) for file in files]

    # Parellelization does not seem to work well in here
    # In my laptop I cannot get all the rasters to get converted
    # Perhaps a more low-level setting up of tasks with Dask is required
    # No errors are given from Whitebox so it is hard to say what the problem
    # is.
    def fill(self, output_path: str,
             parallel: bool = True, manual_split: bool = False) -> None:
        # TODO: Decide whether to specify workers and initialize dask client or
        # not
        """
        Run fill to enforce flow direction (i.e., make it
        hydrologically correct) on all Dems in collection`
        """
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for dem in self.collection:
                dem.fill(output_path=output_path)
        else:
            compute_list = []
            for index, dem in enumerate(self.collection):
                process = delayed(dem.fill)(output_path=output_path)
                compute_list.append(process)
                if manual_split and not index % n_workers:
                    _ = dask.compute(*compute_list)
                    compute_list = []
            _ = dask.compute(*compute_list)

        # TODO, add error handling with finally such that the dask_client
        # always gets closed
        dask_client.close()

    def breach(self, output_path: str, dist: int = 5, fill: bool = True,
               parallel: bool = True, manual_split: bool = False) -> None:
        # TODO: Decide whether to specify workers and initialize dask client or
        # not
        """
        Run breach to enforce flow direction (i.e., make it
        hydrologically correct) on all Dems in collection`
        """
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for dem in self.collection:
                dem.breach(output_path=output_path, dist=dist, fill=fill)
        else:
            compute_list = []
            for index, dem in enumerate(self.collection):
                process = delayed(dem.breach)(output_path=output_path,
                                              dist=dist, fill=fill)
                compute_list.append(process)
                if manual_split and not index % n_workers:
                    _ = dask.compute(*compute_list)
                    compute_list = []
                    print('dupa')
            _ = dask.compute(*compute_list)

        # TODO, add error handling with finally such that the dask_client
        # always gets closed
        dask_client.close()

    def fill_depressions(self, output_path: str, parallel: bool = True,
                         flat_increment: float = 0.001) -> None:
        """Performs fill depressions"""
        if parallel:
            compute_list = []
            for _, dem in enumerate(self.collection):
                process = delayed(dem.fill_depression)(output_path=output_path)
                compute_list.append(process)
            dask.compute(*compute_list)
        else:
            for _, dem in enumerate(self.collection):
                dem.fill_depression(
                    output_path=output_path, flat_increment=flat_increment)

    def rho8_accumulate(self, output_path: str, parallel: bool = True) -> None:
        """Performs fill depressions"""
        if parallel:
            compute_list = []
            for _, dem in enumerate(self.collection):
                process = delayed(dem.rho8_accumulate)(output_path=output_path)
                compute_list.append(process)
            dask.compute(*compute_list)
        else:
            for _, dem in enumerate(self.collection):
                dem.rho8_accumulate(output_path=output_path)

    def accumulate(self, output_path: str, parallel: bool = True) -> None:
        """ Performs flow accumulation on a collection of rasters """
        # TODO: Shall we start dask client in here for parallel computation?
        if parallel:
            compute_list = []
            for _, dem in enumerate(self.collection):
                process = delayed(dem.accumulate)(output_path=output_path)
                compute_list.append(process)
            dask.compute(*compute_list)
        else:
            for _, dem in enumerate(self.collection):
                dem.accumulate(output_path=output_path)

    def d8_pointer(self, output_path: str, parallel: bool = True) -> None:
        """ Generate a flow pointer grid using the simple D8
            (O'Callaghan and Mark, 1984) algorithm on a number of rasters
            in the collection """
        if parallel:
            compute_list = []
            for _, dem in enumerate(self.collection):
                process = delayed(dem.d8_pointer)(output_path=output_path)
                compute_list.append(process)
            dask.compute(*compute_list)
        else:
            for _, dem in enumerate(self.collection):
                dem.d8_pointer(output_path=output_path)

    def extract_streams(self, output_path: str, stream_order: int,
                        parallel: bool = True) -> None:
        """ Extract, or map, the likely stream cells from an input
            flow-accumulation image for a number of rasters (images)
            in the collection """
        if parallel:
            compute_list = []
            for _, dem in enumerate(self.collection):
                process = delayed(dem.extract_streams)(
                    output_path=output_path,
                    stream_order=stream_order)
                compute_list.append(process)
            dask.compute(*compute_list)
        else:
            for _, dem in enumerate(self.collection):
                dem.extract_streams(output_path=output_path,
                                    stream_order=stream_order)

    def stream_to_vec(self, output_path: str,
                      d8_flow_pntr_list: List[str],
                      parallel: bool = True) -> None:
        """ Convert a raster stream file into a vector file for a number
            of streams in the collection.
            We need to pass a list of d8_flow_pointer_rasters corresponding
            to raster stream files
        """
        if parallel:
            compute_list = []
            for index, stream in enumerate(self.collection):
                d8_flow_pointer = d8_flow_pntr_list[index]
                process = delayed(stream.stream_to_vec)(
                    d8_flow_pointer=d8_flow_pointer,
                    output_path=output_path)
                compute_list.append(process)
            dask.compute(*compute_list)
        else:
            for index, stream in enumerate(self.collection):
                d8_flow_pointer = d8_flow_pntr_list[index]
                stream.stream_to_vec(
                    d8_flow_pointer=d8_flow_pointer,
                    output_path=output_path)

    def snap_pour_pts(self, pt_zones: VectorCollection, output_path: str,
                      snap_dist: float = 0.6,
                      parallel: bool = True) -> None:
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for dem, vector in zip(self.collection, pt_zones.collection):
                dem.snap_pour_pt(vector, output_path=output_path,
                                 snap_dist=snap_dist)
        else:
            compute_list = []
            for dem, vector in zip(self.collection, pt_zones.collection):
                process = delayed(dem.snap_pour_pt)(
                    vector, output_path=output_path, snap_dist=snap_dist)
                compute_list.append(process)
            _ = dask.compute(*compute_list)

    def watershed_individuals(
            self, output_path: str, snapped_pts: VectorCollection,
            parallel: bool = True) -> None:
        """ Find upper catchment/watershed based on a snapped point or list of
            snapped points.
            If using the list of points at once, please set snapped_pts as
            zone_based pts rather than individual points """
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for dem, vector in zip(self.collection, snapped_pts.collection):
                dem.watershed_individual(
                    snapped_pts=vector, output_path=output_path)
        else:
            compute_list = []
            for dem, vector in zip(self.collection, snapped_pts.collection):
                process = delayed(dem.watershed_individual)(
                    snapped_pts=vector, output_path=output_path)
                compute_list.append(process)
            _ = dask.compute(*compute_list)

    def clip_raster_to_polygons(
            self, output_path: str, polygons: VectorCollection,
            parallel: bool = True) -> None:
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for dem, vector in zip(self.collection, polygons.collection):
                dem.clip_raster_to_polygon(
                    output_path=output_path, polygon=vector)
        else:
            compute_list = []
            for dem, vector in zip(self.collection, polygons.collection):
                process = delayed(dem.clip_raster_to_polygon)(
                    output_path=output_path, polygon=vector)
                compute_list.append(process)
            _ = dask.compute(*compute_list)

    def raster_to_polygons(self, output_path: str,
                           parallel: bool = True) -> None:
        """vectorize ID-ed watersheds"""
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)

        if not parallel:
            for dem in self.collection:
                dem.raster_to_polygon(output_path=output_path)
        else:
            compute_list = []
            for dem in self.collection:
                process = delayed(dem.raster_to_polygon)(output_path=output_path)
                compute_list.append(process)
            _ = dask.compute(*compute_list)

    def extract_inundated_areas(
            self, output_path: str, output_path2: str,
            gdf: Vector, parallel: bool = True) -> None:
        """ Run extract inundated area for a collection of points/dams """
        n_workers = specify_max_workers()
        dask_client = Client(n_workers=n_workers)
        if not parallel:
            for dem in self.collection:
                dem.extract_inundated_area(
                    output_path=output_path, output_path2=output_path2,
                    gdf=gdf)
        else:
            compute_list = []
            for dem in self.collection:
                process = delayed(dem.extract_inundated_area)(
                    output_path=output_path, output_path2=output_path2,
                    gdf=gdf)

                compute_list.append(process)
            _ = dask.compute(*compute_list)
