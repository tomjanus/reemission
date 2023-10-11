"""Toolbox with functions to assist GIS operations"""
from __future__ import annotations
from dataclasses import dataclass, field
import pathlib
from typing import List, Any, Dict, Optional
import pandas as pd
import geopandas as gpd
from reemission.utils import (
    load_toml, get_package_file, load_shape, load_csv, load_json)
from reemission.app_logger import create_logger


# Create a logger
logger = create_logger(logger_name=__name__)
# Read toml configuration file
config = load_toml(get_package_file('config/heet.toml'))


@dataclass
class TabToShpCopy:
    """A class for transferring data from a tabular data source to a shape (GIS) 
    data source.

    Parameters
        shp_data : GeoDataFrame of shape data
        tab_data : DataFrame of tabular data
    """
    shp_data: gpd.GeoDataFrame
    tab_data: pd.DataFrame

    @classmethod
    def from_files(cls, shp_file: pathlib.Path, 
                   tab_file: pathlib.Path) -> TabToShpCopy:
        """Loads a shape and tabular data source from files"""
        return cls(shp_data=load_shape(shp_file), tab_data=load_csv(tab_file))

    def find_missing_tab_keys(
            self, shp_column: Any, tab_column: Any) -> List[Any]:
        """Find a list of keys that are present in tabular data but are 
        absent in the shape (GIS) data. Common keys are keys that are shared
        between two datasets in shp_column and tab_column, respectively.
        """
        tab_data_ids = set(self.tab_data[tab_column].tolist())
        shp_data_ids = set(self.shp_data[shp_column].tolist())
        missing_keys = list(tab_data_ids - shp_data_ids)
        return missing_keys

    def transfer(
            self, source_key_column: Any, target_key_column: Any, 
            fields: List[Any], inplace: bool = True) -> gpd.GeoDataFrame:
        """Transfers data from tabular data source to shape (GIS) data source.
        Matching data is based on two columns: source_key_column for source data
        and target_key_column for target data.
        
        Returns
            A GeoDataFrame of the modified shape data.  
            
        If inplace is True, the shape data will be modified in place. Defaults 
        to True."""
        fields = [field.replace("\\ufeff", "") for field in fields.copy()]
        fields.append(source_key_column)
        for field_ in fields.copy():
            if field_ not in self.tab_data.columns:
                logger.warning("Field %s not present in source data", field_)
                fields.remove(field_)
        out_df = self.shp_data.join(
            self.tab_data[fields].set_index(source_key_column), 
            on=target_key_column)
        if inplace:
            self.shp_data = out_df
        return out_df

    def save_shp(self, shp_file_name: pathlib.Path) -> None:
        """Saves the shape data to a file"""
        self.shp_data.to_file(shp_file_name)
        logger.info("Shape file saved to %s", shp_file_name.as_posix())


@dataclass
class ShpToShpCopy:
    """Opens source and target shape, finds sommon rows (data) based on the
    match key. For matching keys, adds new fields existing in the source shape
    and copy the values from source shape to target shape.
    Overwrites the target shape."""
    source_data: gpd.GeoDataFrame
    target_data: gpd.GeoDataFrame
    joint_data: gpd.GeoDataFrame = field(init=False)

    @classmethod
    def from_files(
            cls, source_shp_file: pathlib.Path, 
            target_shp_file: pathlib.Path) -> ShpToShpCopy:
        """Loads source and target shape files from files"""
        return cls(
            source_data=load_shape(path=source_shp_file),
            target_data=load_shape(path=target_shp_file))

    def copy_data(
            self, source_key: Any, target_key: Any, fields: List[str]) -> None:
        """Join two dataframes (source_data and target_data and produce the 
        joined object"""
        fields.append(source_key)
        self.joint_data = self.target_data.join(
            self.source_data[fields].set_index(source_key), on=target_key)
        
    def save_joint_data(self, file_name: pathlib.Path) -> None:
        """Save joint data to a shap file"""
        try:
            self.target_data.to_file(file_name)
            logger.info("Output saved to file: %s", file_name)
        except AttributeError:
            logger.warning("Joint data does not exist")


@dataclass
class ExtractFromJSON:
    """Extract data from a JSON file and present them in a tabular data format"""
    # inputs are keys found in the inputs section of the reemission output json file
    # outputs are keys found in th eoutputs section of the reemission output json file
    reemission_output: Dict
    extracted_keys: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Provide default extracted keys if empty dictionary given as an argument"""
        if not self.extracted_keys:
            self.extracted_keys = {
                "inputs": [],
                "outputs": ["co2_net", "ch4_net", "n2o_mean"]
            }
    
    @classmethod
    def from_file(
            cls, reemission_file: pathlib.Path, 
            extracted_keys: Optional[Dict[str, List[str]]] = None) -> ExtractFromJSON:
        """Instantiate object from data files.
        Args:
            reemission_file: Output json file from Re-Emission
        """
        if extracted_keys is None:
            return cls(reemission_output=load_json(reemission_file))
        return cls(
            reemission_output=load_json(reemission_file),
            extracted_keys=extracted_keys)
    
    def extract_outputs(self) -> pd.DataFrame:
        """Extract selected RE-Emission gas emission estimation outputs and
        save them to a dataframe. Calculates and adds additional field 'tot_em'
        """
        # 1. Extract data given in class variable extracted_keys
        extracted_data = {}
        for reservoir_name, output_data in self.reemission_output.items():
            for category, variables in self.extracted_keys.items():
                var_value_pairs = {
                    variable: output_data[category][variable]['value'] for 
                    variable in variables}
            extracted_data[reservoir_name] = var_value_pairs
        # 2. Convert dictionary with extracted data to dataframe
        output_df = pd.DataFrame.from_dict(
            data=extracted_data, orient='index')
        output_df.index.rename(name="name", inplace=True)
        output_df['tot_em'] = \
            output_df[self.extracted_keys['outputs']].sum(axis=1)
        output_df.reset_index(inplace=True)
        return output_df


def append_data_to_shapes(
        shp_folder: pathlib.Path, data_file: pathlib.Path,
        config_dict: Dict, output_folder: Optional[pathlib.Path]=None, 
        suffix: str = "_updated") -> None:
    """Appends data from the loaded tabular data into shape files using config
    information. Uses TabToShpCopy class to copy the selected tabular data
    to the shape file. Saves modified shape files in the same directory as
    the original shape files but under a different name. The new name is equal
    to the original name plus the suffix provided as an argument.
    
    Used to transfer data from HEET csv output file to the shape files."""
    config_shape_categories = config_dict['shp_output'].keys()
    tab_data = load_csv(data_file)
    for shape in shp_folder.glob('*.shp'):
        # Such as catchments, dams, etc.
        file_category = pathlib.Path(shape).resolve().stem
        if file_category in config_shape_categories:
            shp_data = load_shape(shape)
            tab_shp_copy = TabToShpCopy(shp_data, tab_data)
            match_keys = config['shp_output'][file_category]['match_keys']
            tab_shp_copy.transfer(
                source_key_column=match_keys[0], 
                target_key_column=match_keys[1],
                fields=config['shp_output'][file_category]['fields'])
            if output_folder is None:
                output_folder = shape.parent
            new_shp_file = pathlib.Path(
                output_folder, shape.stem + suffix + ".shp")
            tab_shp_copy.save_shp(shp_file_name=new_shp_file)


if __name__ == '__main__':
    """ """