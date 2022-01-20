import os
import dask
import pathlib
import geopandas as gpd
from dask.distributed import Client
import src.inundation
from src.inundation import utils
#from src.inundation.reservoirs_test import Dem, Vector, Dams, Collection, DemCollection, find_pointers
from src.inundation.reservoirs import Dem, Vector, Dams, Collection, DemCollection, find_pointers, VectorCollection
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Optional

# Specify paths to files
mother_directory = '/home/jojo/futuredam_update/inundation_code/dam_data/'
result_folder = 'resutls_1'

@dataclass
class  application(ABC):
    mother_directory:  str
    result_folder: str
    basin_directory: str
    upscale_factor: float
    snap_dist: float
    stream_order: int

    ## post init
    original_dams_shape_path : Optional[Any] = None
    raster_path : Optional[Any] = None
    reduced_raster_path : Optional[Any] = None
    modified_dams_shape_path :Optional[Any] = None
    basins_dir : Optional[Any] = None
    clipped_dem_path : Optional[Any] = None
    filled_dem_path :Optional[Any] = None
    breached_dem_path : Optional[Any] = None
    d8_dem_path : Optional[Any] = None
    flow_acc_path : Optional[Any] = None
    stream_dem_path : Optional[Any] = None
    stream_vector_path : Optional[Any] = None
    dam_per_zone_path : Optional[Any] = None
    dams_snapped_per_zone_path :Optional[Any] = None
    individual_snapped_dams_path : Optional[Any] = None
    individual_watershed_path : Optional[Any] = None
    watershed_shape_path : Optional[Any] = None
    watershed_shape_single_polygon_path : Optional[Any] = None
    breached_dem_whole_path : Optional[Any] = None
    #merged_water
    watershed_clipped_path : Optional[Any] = None
    inundated_dem_value_path : Optional[Any] = None
    inundated_dem_geometry_path : Optional[Any] = None
    inundated_shape_path : Optional[Any] = None
    inundated_shape_single_polygon_path : Optional[Any] = None
    inundated_shape_merged_path : Optional[Any] = None
    dams_snapped_merged_path : Optional[Any] = None

    # @abstractmethod
    # def choose_points(self)-> Any:
    #     pass 

    # @abstractmethod
    # def snap_stream_points(self)-> Any:
    #     pass

    @abstractmethod
    def watershed_run(self)-> Any:
        pass

    @abstractmethod
    def inundation_run(self) -> Any:
        pass

@dataclass
class watershed(application):

        def __post_init__(self):

            self.original_dams_shape_path = os.path.join(self.mother_directory,'dams','all_dams_replaced_refactored.shp')
            self.raster_path = os.path.join(self.mother_directory,'nasa_mosiac','original_resolution.tif')
            self.reduced_raster_path = os.path.join(self.mother_directory,'nasa_mosiac','dem_downscaled.tif')
            self.modified_dams_shape_path = os.path.join(self.mother_directory,'IFC_locations','IFC_points.shp')
            self.basins_dir = os.path.join(self.mother_directory,self.basin_directory)
            self.clipped_dem_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem')
            self.filled_dem_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem_filled')
            self.breached_dem_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem_breached')
            self.d8_dem_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem_d8')
            self.flow_acc_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem_acc')
            self.stream_dem_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem_stream')
            self.stream_vector_path = os.path.join(self.mother_directory,self.result_folder,'nasa_clipped_dem_stream_shape')
            self.dam_per_zone_path = os.path.join(self.mother_directory,self.result_folder,'to_snap_pts')
            self.dams_snapped_per_zone_path = os.path.join(self.mother_directory,self.result_folder,'nasa_snap_pts')
            self.individual_snapped_dams_path = os.path.join(self.mother_directory,self.result_folder, 'individual_snapped_dams')
            self.individual_watershed_path = os.path.join(self.mother_directory,self.result_folder,'individual_watershed')
            self.watershed_shape_path = os.path.join(self.mother_directory,self.result_folder,'watershed_shape')
            self.watershed_shape_single_polygon_path = os.path.join(self.mother_directory,self.result_folder,'watershed_single_polygon_shape')
            self.breached_dem_whole_path = os.path.join(self.mother_directory,self.result_folder,'breached_dem_whole')
            #merged_water
            self.watershed_clipped_path = os.path.join(self.mother_directory,self.result_folder,'watershed_dem')
            self.inundated_dem_value_path = os.path.join(self.mother_directory,self.result_folder, 'inundated_dem_value')
            self.inundated_dem_geometry_path = os.path.join(self.mother_directory, self.result_folder,'inundated_dem_geometry')
            self.inundated_shape_path = os.path.join(self.mother_directory, self.result_folder,'inundated_shape')
            self.inundated_shape_single_polygon_path = os.path.join(self.mother_directory, self.result_folder,'inundated_shape_singel_polygon')
            self.inundated_shape_merged_path = os.path.join(self.mother_directory, self.result_folder,'inudated_shape_merged')
            self.dams_snapped_merged_path = os.path.join(mother_directory,self.result_folder,'nasa_dams_snapped_merged')

        

        def watershed_run(self) -> Any:
            dem = Dem(path=self.raster_path)
            dem.resize(new_path=self.reduced_raster_path, verbose=True, upscale_factor = self.upscale_factor)
            print('/n')

            dem = Dem(path= self.reduced_raster_path)
            dem.fill_depression(output_path=self.breached_dem_whole_path)
            dem = DemCollection()
            dem.from_path(self.breached_dem_whole_path)
            dem = dem.collection[0]
            ref_dem = self.raster_path
            dem.raster_attr(new_path=self.breached_dem_whole_path, ref_data = ref_dem)
            print(f'DEM data filled using (hydrologciall corrected using data with resoltion {self.upscale_factor})')

            # Find shape files representing zones to clip DEM to
            zone_shapes = utils.find_files(path=self.basins_dir, ext='.shp')
            # Initialize DEM and clip it to zones
            
            dem = DemCollection()
            dem.from_path(self.breached_dem_whole_path)
            dem = dem.collection[0]
            dem.clip_to_polygons(
                zones=VectorCollection([Vector(shape) for shape in zone_shapes]),
                clipped_path=self.clipped_dem_path, parallel=False)
            print(f'Clipped DEM into using zones as per {self.basins_dir}')

            dem_col = DemCollection()
            dem_col.from_path(self.clipped_dem_path)
            dem_col.d8_pointer(output_path=self.d8_dem_path, parallel=False)
            dem_col = DemCollection()
            dem_col.from_path(self.clipped_dem_path)
            dem_col.accumulate(output_path=self.flow_acc_path,parallel=False)
            print('Flow accumulation step compeleted')

            dem_col = DemCollection()
            # Initialize with flow accumulation rasters
            dem_col.from_path(self.flow_acc_path)
            dem_col.extract_streams(output_path=self.stream_dem_path, stream_order=self.stream_order, parallel=False)
            dem_col = DemCollection()
            dem_col.from_path(self.stream_dem_path)
            # Create a d8_flow_pntr_list (decided for now to leavt this out of classes and have it as a separate
            # library function)
            d8_flow_pntr_list = find_pointers(dem_names=dem_col.paths, pointer_path=self.d8_dem_path,d8=True)
            # Convert raster streams to vectors
            dem_col.stream_to_vec(output_path=self.stream_vector_path, d8_flow_pntr_list=d8_flow_pntr_list, parallel=False)
            print('Stream extractions compleleted')

            # Dams divided into zones
            dams = Dams(path=self.modified_dams_shape_path)
            dams.divide_to_zones(path_to_zone_shapes=self.basins_dir, output_path=self.dam_per_zone_path)

            vector_col = VectorCollection()
            vector_col.from_path(self.dam_per_zone_path)
            dem_col = DemCollection()
            #dem_col.from_path(stream_dem_path)
            _list = find_pointers(dem_names=vector_col.paths, pointer_path=self.stream_dem_path,d8=False)
            
            
            dem_col.collection = [Dem(i) for i in _list]
            dem_col.snap_pour_pts(output_path = self.dams_snapped_per_zone_path, pt_zones = vector_col,      
            snap_dist = self.snap_dist, parallel=False)##snap_dist 0.6 in degree resolution

            Dams_col = VectorCollection()
            Dams_col.from_path(self.dams_snapped_per_zone_path)
            Dams_col.shp_individuals(self.individual_snapped_dams_path)
            print(f'Points snapped to the nearest stream using snapping distance of {self.snap_dist} \n')
            print(f'Please check with GIS visulatization using extracted streams in folder: {self.stream_vector_path}, and snapped points in folder: {self.individual_snapped_dams_path}')

            for folder in os.listdir(self.individual_snapped_dams_path):

                Dams_col = VectorCollection()
                Dams_col.from_path(os.path.join(self.individual_snapped_dams_path,folder))
                d8_flow_pntr_list = find_pointers(dem_names=Dams_col.paths, pointer_path=self.d8_dem_path, d8 = False)
                
                Dem_col = DemCollection()
                Dem_col.collection = [Dem(file) for file in d8_flow_pntr_list]
                Dem_col.watershed_individuals(output_path = self.individual_watershed_path,snapped_pts=Dams_col,parallel=False) 

            for folder in os.listdir(self.individual_watershed_path):
                _path = os.path.join(mother_directory,self.individual_watershed_path,folder)
                print(_path)
                dem_col = DemCollection()
                dem_col.from_path(_path)
                dem_col.raster_to_polygons(output_path=self.watershed_shape_path,parallel=False)

            print(f'Delineated waterhed shapefiles created in folder: {self.watershed_shape_path}')

            ## single polygons for each watershed shape
            vector_col = VectorCollection()
            vector_col.from_path(self.watershed_shape_path)
            vector_col.to_single_polygons(output_path=self.watershed_shape_single_polygon_path,parallel=False)

            ## raster ; TO DO later about resolution. 
            vector_col = VectorCollection()
            vector_col.from_path(self.watershed_shape_single_polygon_path)
            dem = DemCollection()
            dem.from_path(self.breached_dem_whole_path)
            dem = dem.collection[0]
            dem.clip_raster_to_polygons(output_path=self.watershed_clipped_path, parallel=False,polygons=vector_col.collection)

        
        def inundation_run(self) -> Any:
            gdf = Vector(path=self.modified_dams_shape_path)
            Dem_col = DemCollection()
            Dem_col.from_path(self.watershed_clipped_path)

            Dem_col.extract_inundated_areas(output_path = self.inundated_dem_value_path,
                                        output_path2 = self.inundated_dem_geometry_path,gdf=gdf, 
                                        parallel=False)
            dem_col = DemCollection()
            dem_col.from_path(self.inundated_dem_geometry_path)
            dem_col.raster_to_polygons(output_path=self.inundated_shape_path,parallel=False)
            vector_col = VectorCollection()
            vector_col.from_path(self.inundated_shape_path)
            vector_col.to_single_polygons(output_path=self.inundated_shape_single_polygon_path)
            vector_col = VectorCollection()
            vector_col.from_path(self.inundated_shape_single_polygon_path)
            vector_col.merge_vectors(output_path=self.inundated_shape_merged_path)

            print(f'Inudation area shapefiles create. Pelase check fodler: {self.inundated_shape_merged_path}')

