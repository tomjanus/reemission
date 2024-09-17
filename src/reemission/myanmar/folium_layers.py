""" """
from typing import Dict, Optional, ClassVar, List, Tuple
from dataclasses import dataclass
import pathlib
import geojson
import folium
import geopandas as gpd
from folium.features import CustomIcon
import branca.colormap as cm
from reemission.postprocessing.visualise import FoliumLayer


tooltip_style = """
    background-color: #F0EFEF;
    border: 1px solid grey;
    border-radius: 3px;
    font-size: 15px;
    box-shadow: 6px;
    """

popup_style = """
    background-color: #F0EFEF;
    border: 1px solid grey;
    border-radius: 6px;
    font-size: 15px;
    box-shadow: 6px;
    """


class GeoJSONLoader:
    """Class for saving geojson FeatureCollection objects to .geojson files"""
    @staticmethod
    def load(file_name: pathlib.Path) -> Optional[Dict]:
        """ """
        with open(file_name, 'r', encoding='utf-8') as f:
            data = geojson.load(f)
            return data


@dataclass
class MyaConflicDataLayer(FoliumLayer):
    """Point data of settlements"""
    data: gpd.GeoDataFrame

    def toFoliumGeoJSON(self) -> folium.GeoJson:
        raise NotImplementedError


@dataclass
class MyaSettlementLayer(FoliumLayer):
    """Geodata representing conflic history in Myanmar"""
    data: gpd.GeoDataFrame

    @staticmethod
    def _style_function(feature) -> Dict:
        props = feature.get('properties')
        markup = f"""
                <div style="font-size: 1.4em;">
                <div style="width: 11px;
                            height: 11px;
                            border: 1px solid black;
                            border-radius: 5px;
                            background-color: red;">
                </div>
                {props.get('Name')}
            </div>
        """
        return {"html": markup}
    
    @staticmethod
    def _tooltip() -> folium.GeoJsonTooltip:
        """ """
        tooltip_village = folium.GeoJsonTooltip(
            fields=["Name", "Population"],
            aliases=["Name", "Population"],
            style=tooltip_style,
            localize=True,
            sticky=False,
            labels=True,
            max_width=800)
        return tooltip_village
    

    def toFoliumGeoJSON(self) -> folium.GeoJson:
        return folium.GeoJson(
            data=self.data,
            name="Villages",
            marker=folium.Marker(icon=folium.DivIcon()),
            tooltip=self._tooltip(),
            #popup=self._popup(),
            highlight_function=lambda x: {"fillOpacity": 0.2},
            style_function=self._style_function
        )    


@dataclass
class MyaIFCDamsLayer(FoliumLayer):
    """IFC hydropower database for Myanmar (points) as a Folium layer"""
    data: gpd.GeoDataFrame
    filter_noname: bool = True
    name_field: ClassVar[str] = 'DAM_NAME'

    def __post_init__(self) -> None:
        # Filter out the dams that do not have names
        try:
            self.data = self.data[self.data[self.name_field].notna()]
        except KeyError:
            pass

    @staticmethod
    def _style_function(feature) -> Dict:
        props = feature.get('properties')
        markup = f"""
                <div style="font-size: 1.4em;">
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
    def _tooltip() -> folium.GeoJsonTooltip:
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
    

    @staticmethod
    def _popup() -> folium.GeoJsonPopup:
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
    def _tooltip() -> folium.GeoJsonTooltip:
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

    def toFoliumGeoJSON(self) -> folium.GeoJson:
        return folium.GeoJson(
            data=self.data,
            name="IFC Dams",
            marker=folium.Marker(icon=folium.DivIcon()),
            tooltip=self._tooltip(),
            popup=self._popup(),
            highlight_function=lambda x: {"fillOpacity": 1},
            style_function=self._style_function
        )


@dataclass
class DelineationsPolyLayer(FoliumLayer):
    """GeoCARET-delineated reservoirs as a Folium layer"""
    data: gpd.GeoDataFrame
    v_minmax: Optional[Tuple[float, float]] = None
    drop_na_emissions: bool = True
    emission_field: ClassVar[str] = 'tot_em'
    color_scale: ClassVar[List[str]] = ["green", "yellow", "orange", "red"]
    default_scale: ClassVar[Tuple[float, float]] = (100, 3_000)
    fill_opacity: float = 0.6
    highlight_opacity: float = 0.8

    def __post_init__(self) -> None:
        self.data['r_volume_mcm'] = self.data['r_volume_m'] / 10**6
        # Remove fields that have not been assigned emission values
        if self.drop_na_emissions:
            self.data = self.data[self.data[self.emission_field].notna()]
        if self.v_minmax:
            v_min, v_max = self.v_minmax[0], self.v_minmax[1]
        else:
            try:
                v_min, v_max = \
                    min(self.data[self.emission_field]), \
                    max(self.data[self.emission_field])
            except KeyError:
                v_min, v_max = self.default_scale[0], self.default_scale[1]
        self.linear = cm.LinearColormap(colors=self.color_scale, vmin=v_min, vmax=v_max)

    @staticmethod
    def _popup() -> folium.GeoJsonPopup:
        """ """
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
    def _tooltip() -> folium.GeoJsonTooltip:
        tooltip_res = folium.GeoJsonTooltip(
            fields=[
                "name", "id", "type", "r_volume_mcm", "r_area_km2", "r_mean_dep"],
            aliases=[
                "Name", "ID", "Type", "Volume, mcm", "Area, km2", 
                "Mean depth, m"],
            localize=True,
            sticky=False,
            labels=True,
            style=tooltip_style,
            max_width=900)
        return tooltip_res

    def toFoliumGeoJSON(self) -> folium.GeoJson:
        """ """
        return folium.GeoJson(
            data=self.data,
            name="Reservoirs",
            tooltip=self._tooltip(),
            popup=self._popup(),
            style_function=lambda feature: {
                "fillColor": \
                    self.linear(feature['properties'].get(self.emission_field)),
                "color": "black",
                "fillOpacity": self.fill_opacity,
                "weight": 1},
            highlight_function=lambda x: {"fillOpacity": self.highlight_opacity},
            zoom_on_click=True
        )


@dataclass
class HydroLakesPolyLayer(FoliumLayer):
    """Hydrolakes geodata as a Folium layer"""
    data: gpd.GeoDataFrame

    def __post_init__(self) -> None:
        """ """
        # Remap lake types from numeric to text
        map = {1: 'Lake', 2: 'Reservoir', 3: 'Lake with control'}
        self.data['Lake_type'] = self.data['Lake_type'].replace(map)
        self.data['Lake_name'] = self.data['Lake_name'].replace({None: 'No name'})

    @staticmethod
    def _tooltip() -> folium.GeoJsonTooltip:
        """ """
        tooltip_dam = folium.GeoJsonTooltip(
            fields=[
                "Lake_name", "Lake_type", "Lake_area", "Vol_total", "Depth_avg", "Dis_avg", 
                "Elevation"],
            aliases=[
                "Lake_name", "Lake type", "Area (km2)", "Volume (mcm)", "Average depth (m)", 
                "Average discharge (m3/s)", "Elevation (masl)"],
            style=tooltip_style,
            localize=True,
            sticky=False,
            labels=True,
            max_width=900)
        return tooltip_dam

    def toFoliumGeoJSON(self) -> folium.GeoJson:
        return folium.GeoJson(
            data=self.data,
            name="Hydrolakes",
            tooltip=self._tooltip(),
            style_function=lambda feature: {
                "fillColor": 'blue',
                "color": "black",
                "fillOpacity": 0.3,
                "weight": 1},
            highlight_function=lambda x: {"fillOpacity": 0.35},
            zoom_on_click=False
        )

rivers_style_function = lambda x: {
        'color' :  'black',
        'opacity' : 0.50,
        'weight' : 2}

@dataclass
class MyaPywrModelLayer(FoliumLayer):
    """Pywr water resources model for Myanmer as a Folium layer"""
    dam_icon_image: pathlib.Path
    turbine_icon_image: pathlib.Path
    model_folder: pathlib.Path
    model_name: str

    def __post_init__(self) -> None:
        """ """
        self.model_feature = folium.FeatureGroup(name='Water Resource Model')

    tooltip_style: ClassVar[str] = """
        background-color: #F0EFEF;
        border: 1px solid grey;
        border-radius: 3px;
        box-shadow: 6px;
        """
    popup_style: ClassVar[str] = """
        background-color: #F0EFEF;
        border: 1px solid grey;
        border-radius: 6px;
        box-shadow: 6px;
        """
    
    basins: ClassVar[List[str]] = ["Irrawaddy", "Salween", "Sittaung"]

    # Markers for GeoJson data
    @property
    def link_node_marker(self) -> folium.CircleMarker:
        return folium.CircleMarker(
            radius=6,  # Radius in pixels
            weight=0,  # Outline weight
            fill_color='grey',
            fill_opacity=0.9)

    @property
    def input_node_marker(self) -> folium.CircleMarker:
        return folium.CircleMarker(
            radius=6,  # Radius in pixels
            weight=0,  # Outline weight
            fill_color='green',
            fill_opacity=0.8)

    @property
    def output_node_marker(self) -> folium.CircleMarker:
        return folium.CircleMarker(
            radius=6,  # Radius in pixels
            weight=0,  # Outline weight
            fill_color='orange',
            fill_opacity=0.8)

    @staticmethod
    def _reservoir_popup(feature_data: Dict) -> folium.Popup:
        """Dynamic popup creation for reservoir nodes"""
        _res_fields = [
            "name", "comment", "min_volume", "max_volume", "min_level", 
            "max_level", "min_area", "max_area"]
        parsed_data = {}
        for field_ in _res_fields:
            try:
                parsed_data[field_] = feature_data[field_]
            except KeyError:
                parsed_data[field_] = "-"
        popup_text = folium.Html(
        """<!DOCTYPE html>
            <html>
            <head>
            <style>
            table {
            font-family: arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            }

            td, th {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
            }

            tr:nth-child(even) {
            background-color: #dddddd;
            }
            </style>
            </head>
            """
            +
            f"""
            <body>

            <h2>{parsed_data['name']}</h2>

            <table>
            <tr>
                <td><b>Comment</b></td>
                <td>{parsed_data["comment"]}</td>
                <td>-</td>
            </tr>
            <tr>
                <td><b>Min Volume</b></td>
                <td>{parsed_data["min_volume"]}</td>
                <td>Mm3</td>
            </tr>
            <tr>
                <td><b>Max Volume</b></td>
                <td>{parsed_data["max_volume"]}</td>
                <td>Mm3</td>
            </tr>
            <tr>
                <td><b>Min Level</b></td>
                <td>{parsed_data["min_level"]}</td>
                <td>m</td>
            </tr>
            <tr>
                <td><b>Max Level</b></td>
                <td>{parsed_data["max_level"]}</td>
                <td>m</td>
            </tr>
            <tr>
                <td><b>Min Area</b></td>
                <td>{parsed_data["min_area"]}</td>
                <td>km2</td>
            </tr>
            <tr>
                <td><b>Max Area</b></td>
                <td>{parsed_data["max_area"]}</td>
                <td>km2</td>
            </tr>
            </table>
            </body>
            </html>""", script=True) 
        return folium.Popup(popup_text, max_width=800, style=popup_style)

    @staticmethod
    def _reservoir_tooltip(feature_data: Dict) -> folium.Tooltip:
        """Dynamic tooltip creation for reservoir nodes"""
        tooltip = folium.Tooltip(
            text=f"<b>Reservoir </b>{feature_data['name']}")
        return tooltip
        
    @staticmethod
    def _turbine_popup(feature_data: Dict) -> folium.Popup:
        """Dynamic popup creation for turbine nodes"""
        _fields = [
            "name", "turbine_capacity", "turbine_efficiency", "turbine_elevation"]
        parsed_data = {}
        for field_ in _fields:
            try:
                parsed_data[field_] = feature_data[field_]
            except KeyError:
                parsed_data[field_] = " "
        popup_text = folium.Html(
        """<!DOCTYPE html>
            <html>
            <head>
            <style>
            table {
            font-family: arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            }

            td, th {
            border: 1px solid #dddddd;
            text-align: left;
            padding: 8px;
            }

            tr:nth-child(even) {
            background-color: #dddddd;
            }
            </style>
            </head>
            """
            +
            f"""
            <body>

            <h2>{parsed_data['name']}</h2>

            <table>
            <tr>
                <td><b>Turbine capacity</b></td>
                <td>{parsed_data["turbine_capacity"]}</td>
                <td>MW</td>
            </tr>
            <tr>
                <td><b>Turbine Elevation</b></td>
                <td>{parsed_data["turbine_elevation"]}</td>
                <td>m</td>
            </tr>
            <tr>
                <td><b>Turbine Efficiency</b></td>
                <td>{parsed_data["turbine_efficiency"]}</td>
                <td>-</td>
            </tr>
            </table>
            </body>
            </html>""", script=True)
        return folium.Popup(popup_text, max_width=800, style=popup_style)
        
    @staticmethod
    def _turbine_tooltip(feature_data: Dict) -> folium.Tooltip:
        """Dynamic tooltip creation for turbine nodes"""
        tooltip = folium.Tooltip(
            text=f"<b>Turbine </b>{feature_data['name']}")
        return tooltip

    def toFoliumGeoJSON(self) -> folium.FeatureGroup:
        """Make map of the Burmese water resources system"""
        feature_group = folium.FeatureGroup(name=self.model_name + " Water Model")
        # Get paths to model components
        edges_path = self.model_folder / 'edges.geojson'
        link_nodes_path = self.model_folder / 'link_nodes.geojson'
        input_nodes_path = self.model_folder / 'input_nodes.geojson'
        output_nodes_path = self.model_folder / 'output_nodes.geojson'
        turbine_nodes_path = self.model_folder / 'turbine_nodes.geojson'
        storage_nodes_path = self.model_folder / 'storage_nodes.geojson'
        # Add edges
        folium.GeoJson(
            data=GeoJSONLoader.load(edges_path), 
            name="Network edges",
            style_function=rivers_style_function).add_to(feature_group)
        # Add turbine nodes
        turbines = GeoJSONLoader.load(turbine_nodes_path)
        if turbines is not None:
            for feature in turbines['features']:
                geometry = feature['geometry']
                properties = feature['properties']
                lon, lat = geometry['coordinates']
                icon_custom = CustomIcon(
                    self.turbine_icon_image,
                    icon_size=(32, 32),
                    icon_anchor=(16, 16),
                    popup_anchor=(0, 0))
                marker = folium.map.Marker(
                    [lat, lon], icon=icon_custom,
                    popup=self._turbine_popup(properties),
                    tooltip=self._turbine_tooltip(properties))
                feature_group.add_child(marker)
        # Add reservoir nodes
        reservoirs = GeoJSONLoader.load(storage_nodes_path)
        if reservoirs is not None:
            for feature in reservoirs['features']:
                geometry = feature['geometry']
                properties = feature['properties']
                lon, lat = geometry['coordinates']
                icon_custom = CustomIcon(
                    self.dam_icon_image,
                    icon_size=(32, 32),
                    icon_anchor=(16, 16),
                    popup_anchor=(0, 0))
                marker = folium.map.Marker(
                    [lat, lon], icon=icon_custom,  
                    popup=self._reservoir_popup(properties),
                    tooltip=self._reservoir_tooltip(properties))
                feature_group.add_child(marker)
        # Add link nodes
        folium.GeoJson(
            GeoJSONLoader.load(link_nodes_path),
            name="Link nodes",
            popup=folium.GeoJsonPopup(
                fields=['name'],
                aliases=["Name"],
                style=popup_style,
                localize=True,
                labels=True),
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["Link"],
                style=tooltip_style,
                localize=True,
                sticky=False,
                labels=True,
                max_width=800),
            marker=self.link_node_marker).add_to(feature_group)
        # Add input nodes
        folium.GeoJson(
            GeoJSONLoader.load(input_nodes_path),
            name="Input nodes",
            popup=folium.GeoJsonPopup(
                fields=['name'],
                aliases=["Name"],
                style=popup_style,
                localize=True,
                labels=True),
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["Input"],
                style=tooltip_style,
                localize=True,
                sticky=False,
                labels=True,
                max_width=800),
            marker=self.input_node_marker).add_to(feature_group)           
        # Add output nodes
        folium.GeoJson(
            GeoJSONLoader.load(output_nodes_path),
            name="Output nodes",
            popup=folium.GeoJsonPopup(
                fields=['name', 'comment'],
                aliases=["Name", "Comment"],
                style=popup_style,
                localize=True,
                labels=True),
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["Output"],
                style=tooltip_style,
                localize=True,
                sticky=False,
                labels=True,
                max_width=800),
            marker=self.output_node_marker).add_to(feature_group)
        return feature_group
    

@dataclass
class MyanmarOutlineLayer(FoliumLayer):
    """ """
    data: gpd.GeoDataFrame

    def toFoliumGeoJSON(self) -> folium.GeoJson:
        return folium.GeoJson(
            data=self.data,
            name="Myanmar Outline",
            style_function=lambda feature: {
                "fillColor": "grey",
                "color": "black",
                "fillOpacity": 0.08,
                "weight": 0.7},
            highlight_function=lambda x: {"fillOpacity": 0.12},
            zoom_on_click=False
        )  


if __name__ == "__main__":
    ...
