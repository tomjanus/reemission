import os
import dask
import pathlib
import geopandas as gpd
from dask.distributed import Client
import src.inundation
from src.inundation import utils
from src.inundation.reservoirs import Dem, Vector, Dams, Collection, DemCollection, find_pointers, VectorCollection
from dataclasses import dataclass
@dataclass
class config:
    directory = {}

    mother_directory: str = '/home/jojo/futuredam_update/inundation_code/dam_data'
    resolution_factor: float = 1/5
    snap_distance: float = 0.6
    dem_data: str = 'nasa_mosiac'
    dams_shape: str = 'IFC_location'
    dams: str = 'dams'
    basins_zones: str = 'sel_basin'
    

    def initiate_directory(self) -> None:
        
        modified_dams_shape_path =  os.path.join(mother_directory, dams_shape,[i for i in os.listdir(os.path.join(mother_directory,dams_shape)) if i[-4:]=='.shp'][0])
        original_dams_shape_path =  os.path.join(mother_directory, dams_shape,[i for i in os.listdir(os.path.join(mother_directory,dams_shape)) if i[-4:]=='.shp'][0])


# Specify paths to files
mother_directory = '/home/jojo/futuredam_update/inundation_code/dam_data'

original_dams_shape_path = os.path.join(mother_directory,'dams','all_dams_replaced_refactored.shp')
raster_path = os.path.join(mother_directory,'nasa_mosiac','original_resolution.tif')
reduced_raster_path = os.path.join(mother_directory,'nasa_mosiac','dem_downscaled.tif')
modified_dams_shape_path = os.path.join(mother_directory,'IFC_locations','IFC_points.shp')
basins_dir = os.path.join(mother_directory,'sel_basin')
clipped_dem_path = os.path.join(mother_directory,'nasa_clipped_dem')
filled_dem_path = os.path.join(mother_directory,'nasa_clipped_dem_filled')
breached_dem_path = os.path.join(mother_directory,'nasa_clipped_dem_breached')
d8_dem_path = os.path.join(mother_directory,'nasa_clipped_dem_d8')
flow_acc_path = os.path.join(mother_directory,'nasa_clipped_dem_acc')
stream_dem_path = os.path.join(mother_directory,'nasa_clipped_dem_stream')
stream_vector_path = os.path.join(mother_directory,'nasa_clipped_dem_stream_shape')
dam_per_zone_path = os.path.join(mother_directory,'to_snap_pts')
dams_snapped_per_zone_path = os.path.join(mother_directory,'nasa_snap_pts')
individual_snapped_dams_path = os.path.join(mother_directory, 'individual_snapped_dams')
individual_watershed_path = os.path.join(mother_directory,'individual_watershed')
watershed_shape_path = os.path.join(mother_directory,'watershed_shape')
watershed_shape_single_polygon_path = os.path.join(mother_directory,'watershed_single_polygon_shape')
breached_dem_whole_path = os.path.join(mother_directory,'breached_dem_whole')
#merged_water
watershed_clipped_path = os.path.join(mother_directory,'watershed_dem')
inundated_dem_value_path = os.path.join(mother_directory, 'inundated_dem_value')
inundated_dem_geometry_path = os.path.join(mother_directory, 'inundated_dem_geometry')
inundated_shape_path = os.path.join(mother_directory, 'inundated_shape')
inundated_shape_single_polygon_path = os.path.join(mother_directory, 'inundated_shape_singel_polygon')
inundated_shape_merged_path = os.path.join(mother_directory, 'inudated_shape_merged')
dams_snapped_merged_path = os.path.join(mother_directory,'nasa_dams_snapped_merged')


