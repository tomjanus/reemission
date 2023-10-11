"""Visualisation of reservoirs and additional layers as a folium interactive
map"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, ClassVar, Optional, Tuple
import pathlib
from dataclasses import dataclass, field
import geopandas as gpd
import folium
from folium import plugins
import webbrowser
from branca.element import Figure
from reemission.app_logger import create_logger


logger = create_logger(logger_name=__name__)


@dataclass
class FoliumLayer(ABC):
    """ """

    @classmethod
    def fromfile(cls, file: pathlib.Path, *args, **kwargs) -> FoliumLayer:
        """Instantiate from geodata file using geopandas"""
        return cls(data=gpd.read_file(file), *args, **kwargs)

    @abstractmethod
    def toFoliumGeoJSON(self) -> folium.GeoJson | folium.FeatureGroup:
        ...


@dataclass
class FoliumOutputMapper:
    """ """
    layers: List[FoliumLayer] | None = None
    map: folium.Map = field(init=False, repr=False)
    location: List[float] = field(default_factory=list)
    init_zoom: float = 6
    map_layers: ClassVar[List[str]] = ['Open Street Map']
    show_minimap: bool = False
    
    def __post_init__(self) -> None:
        """ """
        default_location = [19.749883, 96.080650]
        if not self.location:
            self.location = default_location
        if not self.layers:
            self.layers = []
        self.map = folium.Map(
            location=self.location, zoom_start=self.init_zoom, min_zoom=5,
            max_zoom=13, control_scale=True)
        
    def add_layer(self, layer: FoliumLayer) -> None:
        self.layers.append(layer)

    def remove_layer(self, layer_classname: str) -> None:
        for layer in self.layers:
            if layer.__class__.__name__ == layer_classname:
                self.layers.remove(layer)
    
    #  v_min = 100, v_max = 3000
    def create_map(
            self, dropna: bool = True, 
            v_minmax: Optional[Tuple[float, float]] = None) -> None:
        """ """
        fig = Figure(width='100%')
        fig.add_child(self.map)
        fig = self.map

        # Add tiles to map
        for layer in self.map_layers:
            folium.raster_layers.TileLayer(layer).add_to(self.map)

        logger.info("Creating map...")

        for layer in self.layers:
            layer.toFoliumGeoJSON().add_to(self.map)
        
        if self.show_minimap:
            # plugin for mini map
            minimap = plugins.MiniMap(toggle_display=True)
            # add minimap to map
            self.map.add_child(minimap)

        # Add full screen button and layer control
        plugins.Fullscreen(position='topright').add_to(self.map)
        folium.LayerControl(collapsed=False).add_to(self.map)
        
    def save_map(self, file_path: pathlib.Path, show: bool = True) -> None:
        """Save map to a (.html) file and open in a web-browser (if show == True)"""
        try:
            self.map.save(file_path)
            logger.info("Saving map to %s", file_path.as_posix())
        except AttributeError:
            logger.warning(
                "Map does not exist. Create a map and try again.")
        else:
            if show:
                webbrowser.open(file_path.as_posix())


if __name__ == "__main__":
    """ """
