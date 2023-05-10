""" """
from __future__ import annotations
from typing import List, ClassVar
import pathlib
from dataclasses import dataclass, field
import geopandas as gpd
import folium
from folium import plugins
import webbrowser
import branca.colormap as cm
from reemission.utils import get_package_file, load_shape
from reemission.app_logger import create_logger


logger = create_logger(logger_name=__name__)


tooltip_style = """
    background-color: #F0EFEF;
    border: 1px solid grey;
    border-radius: 3px;
    box-shadow: 6px;
    """

popup_style = """
    background-color: #F0EFEF;
    border: 1px solid grey;
    border-radius: 6px;
    box-shadow: 6px;
    """


@dataclass
class FoliumOutputMapper:
    """ """
    reservoirs_df: gpd.GeoDataFrame
    dams_df: gpd.GeoDataFrame
    map: folium.Map = field(init=False)
    location: List[float] = field(default_factory=list)
    init_zoom: float = 6
    map_layers: ClassVar[List[str]] = [
        'Open Street Map', 'Stamen Terrain', 'Stamen Toner']
    
    def __post_init__(self) -> None:
        """ """
        if not self.location:
            self.location = [19.749883, 96.080650]
        self.map = folium.Map(
            location=self.location, zoom_start=self.init_zoom, min_zoom=5,
            max_zoom=13, control_scale=True)
    
    @staticmethod
    def dam_style_function(feature):
        props = feature.get('properties')
        markup = f"""
                <div style="font-size: 1.2em;">
                <div style="width: 11px;
                            height: 11px;
                            border: 1px solid black;
                            border-radius: 5px;
                            background-color: grey;">
                </div>
                {props.get('DAM_NAME')}
            </div>
        """
        return {"html": markup}
    
    @staticmethod
    def reservoir_popup() -> folium.GeoJsonPopup:
        popup_res = folium.GeoJsonPopup(
            fields=["name", "co2_net", "ch4_net", "n2o_mean", "tot_em"],
            aliases=[
                "Name", "Net CO2 emission, gCO2/m2/yr", 
                "Net CH4 emission, gCO2/m2/yr", "Net N2O emission, gCO2/m2/yr",
                "Net total emission, gCO2/m2/yr"],
            localize=True,
            labels=True,
            style=popup_style)
        return popup_res
    
    @staticmethod
    def reservoir_tooltip() -> folium.GeoJsonTooltip:
        tooltip_res = folium.GeoJsonTooltip(
            fields=["name", "id", "r_volume_m", "r_area_km2"],
            aliases=["Name", "ID", "Volume, m3", "Area, km2"],
            localize=True,
            sticky=False,
            labels=True,
            style=tooltip_style,
            max_width=800)
        return tooltip_res
    
    @staticmethod
    def dam_popup() -> folium.GeoJsonPopup:
        """ """
        popup_dam = folium.GeoJsonPopup(
            fields=["Inst_cap", "Annual Gen", "Firm Power"],
            aliases=["Installed capacity, MW", "Annual Generation, GWh", 
                     "Firm Power, MW"],
            localize=True,
            labels=True,
            style=popup_style)
        return popup_dam
        
    @staticmethod
    def dam_tooltip() -> folium.GeoJsonTooltip:
        """ """
        tooltip_dam = folium.GeoJsonTooltip(
            fields=["IFC_ID", "DAM_NAME", "Basin", "RoR or Sto", "Status", "Status 2"],
            aliases=["ID", "Name", "Basin", "Type", "Status Long", "Status Short"],
            style=tooltip_style,
            localize=True,
            sticky=False,
            labels=True,
            max_width=800)
        return tooltip_dam
    
    def create_map(self, dropna: bool = True) -> None:
        """ """
        logger.info("Creating map...")   
        map_ = folium.Map(location=self.location, zoom_start=self.init_zoom)
        if dropna:
            emissions = self.reservoirs_df[self.reservoirs_df['tot_em'].notna()]
        else:
            emissions = self.reservoirs_df
        v_min = min(emissions['tot_em'])
        v_max = max(emissions['tot_em'])
        # Override scale TODO: Fix this so that the scale is 'aligned' automatically
        v_min = 100
        v_max = 3000
        linear = cm.LinearColormap(
            ["green", "yellow", "orange", "red"], vmin=v_min, vmax=v_max)
        folium.GeoJson(
            data=emissions,
            name="Reservoirs",
            tooltip=self.reservoir_tooltip(),
            popup=self.reservoir_popup(),
            style_function=lambda feature: {
                "fillColor": linear(feature['properties'].get("tot_em")),
                "color": "black",
                "fillOpacity": 0.7,
                "weight": 1},
            highlight_function=lambda x: {"fillOpacity": 1.0},
            zoom_on_click=True
        ).add_to(map_)

        folium.GeoJson(
            data=self.dams_df,
            name="Dams",
            marker=folium.Marker(icon=folium.DivIcon()),
            tooltip=self.dam_tooltip(),
            popup=self.dam_popup(),
            highlight_function=lambda x: {"fillOpacity": 1},
            style_function=self.dam_style_function
        ).add_to(map_)
        # Add tiles to map
        for layer in self.map_layers:
            folium.raster_layers.TileLayer(layer).add_to(map_)
        self.map = map_
        
        # Add full screen button and layer control
        plugins.Fullscreen(position='topright').add_to(self.map)
        folium.LayerControl(collapsed=False).add_to(self.map)
        
    def save_map(self, file_path: pathlib.Path, show: bool = True) -> None:
        """ """
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
    # Create a map from existing resources
    reservoirs = load_shape(
        get_package_file("../../input_data/reservoirs_updated.shp"))
    dams = load_shape(
        get_package_file("../../input_data/dams_updated.shp"))
    dams_ifc = load_shape(
        get_package_file(
            "../../examples/demo/ifc_db/all_dams_replaced_refactored.shp")
    )
    mapper = FoliumOutputMapper(reservoirs, dams_ifc)
    mapper.create_map()
    mapper.save_map(pathlib.Path("index.html"), show=True)