"""
Presenter Module

This module provides functionality for presenting Greenhouse Gas (GHG) emissions 
computation results in various output formats such as Excel, JSON, LaTeX, and HTML.

Classes:
    ExcelWriter:
        Formats and writes data to an Excel file.
    JSONWriter:
        Formats and writes data to a JSON file.
    LatexWriter:
        Formats and writes data to a LaTeX file using PyLaTeX.
    HTMLWriter:
        Formats and writes data to an HTML file.
    Presenter:
        Reads and processes GHG emission calculation results and outputs them 
        using various writers.

Functions:
    enforce_unity_sum(vector: List[float], epsilon: float = 0.001) -> List[float]:
        Ensures the sum of values in a vector is approximately 1.0 within a given epsilon tolerance.
    landcover_pie(axis: plt.Axes, values: List[float], labels: List[str], colors: List[Any], 
                  title: Optional[str] = None, show_legend: bool = False) -> None:
        Creates a pie chart of land cover composition using matplotlib.
    parse_landcover_composition(vector: List[float]) -> List[float]:
        Parses a vector representing land cover composition into a standardized format.
    write_par_to_dict(input_name: str, parameter: Any, par_dict: Dict, reservoir_name: str, 
                      precision: int = 3) -> Dict:
        Writes a parameter to a dictionary of parameter name-value pairs.
    rollout(var_name: str, var_vector: Union[Sequence, Dict]) -> Iterator[Tuple[str, Any]]:
        Creates variable names and values from a sequence or dictionary.
    round_parameter(number: Union[float, list], number_decimals: int) -> Union[float, list]:
        Rounds a number or each element in a list to the specified number of decimals.

Notes:
    - Each writer class (ExcelWriter, JSONWriter, LatexWriter, HTMLWriter) is designed to handle 
      specific output formats, ensuring the presentation of inputs, outputs, and internal variables 
      in a structured manner.
    - All writer classes inherit from the abstract base class Writer, which enforces the implementation 
      of the `write` method and provides utility methods such as `round_parameter`, `rollout`, and 
      `write_par_to_dict`.

Usage:
    [To be added later]
"""
import collections.abc
from collections.abc import Iterable
from functools import reduce
from dataclasses import dataclass, field
from typing import (
    Dict, Tuple, List, Type, Union, Sequence, Iterator, Any, Optional)
from abc import ABC, abstractmethod
import configparser
import os
from io import BytesIO
import base64
import math
import logging
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from pylatex.utils import bold
from pylatex.errors import CompilerError
from pylatex import (
    PageStyle,
    Head,
    simple_page_number,
    Figure,
    NoEscape,
    Tabu,
    Center,
    Quantity,
    Description,
    Document,
    Section,
    Subsection,
    Command,
    MultiColumn,
)
from reemission.utils import is_latex_installed, safe_cast
from reemission.input import Inputs
from reemission.constants import Landuse
from reemission.auxiliary import rollout_nested_list
from reemission.document_compiler import BatchCompiler
from reemission import registry

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Format pyplot
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    }
)

# Plot parameters
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
TICK_FONTSIZE = 10
ANNOTATION_FONTSIZE = 10

# JSONWriter parameters
JSON_NUMBER_DECIMALS = 4
EXCEL_NUMBER_DECIMALS = 4

# TODO: Create this list from True/False values in the input config YAML file
INPUT_NAMES = [
    'coordinates', 'id', 'type', 'monthly_temps', 'biogenic_factors',
    'year_profile', 'catchment_inputs', 'reservoir_inputs',
    'gasses']

FACTOR_NAMES = {
    "biome": "Biome",
    "climate": "Climate",
    "soil_type": "Soil Type",
    "treatment_factor": "Treatment Factor",
    "landuse_intensity": "Landuse Intensity",
}

landcover_names: List[str] = [
    landcover.value for landcover in Landuse.__dict__['_member_map_'].values()]

def _get_latex_compiler() -> str:
    """
    Retrieves the LaTeX compiler to be used based on application configuration.

    Returns:
        str: The LaTeX compiler to be used. Defaults to 'pdflatex' if no specific compiler
             is configured or if the configured compiler is not recognized.

    Note:
        This function reads the LaTeX compiler setting from the application configuration
        (app_config['latex']['compiler']). If the configured compiler is 'pdflatex' or 
        'latexmk', it returns that compiler. Otherwise, it falls back to the default 
        compiler 'pdflatex'.

    """
    default_compiler = 'pdflatex'
    app_config = registry.main_config.get("app_config")
    try:
        latex_options = app_config['latex']
        compiler = latex_options['compiler']
        if compiler in ('pdflatex', 'latexmk'):
            return compiler
        else:
            return default_compiler
    except KeyError:
        return 'pdflatex'


def format_number(x) -> str:
    """ Formats a number for display in a human-readable way.
    Args:
        x (Any): The number to format. It can be an int, float, or None.
    Returns:
        str: A string representation of the number formatted for readability.
    Note:
        - If `x` is None, it returns "—" (or "N/A").
        - If `x` is an integer, it returns the integer as a string.
        - For floats, it formats them based on their magnitude:
            - Very large or very negative values are formatted with 1 decimal point.
            - Moderate values are formatted with up to 5 significant digits.
            - Very small values are formatted with 3 significant digits.
            - Extremely small values are formatted in engineering notation.
    """
    if x is None:
        return "—"  # or "N/A"
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"

    abs_x = abs(x)

    # Very large or very negative values: 1 decimal point
    if abs_x >= 1e4:
        return f"{x:.1f}"

    # Moderate range: up to 5 significant digits
    elif abs_x >= 1e-3:
        return f"{x:.5g}"

    # Very small values
    elif abs_x >= 1e-5:
        return f"{x:.3g}"

    # Extremely small → engineering notation
    else:
        exponent = int(math.floor(math.log10(abs_x) / 3) * 3)
        mantissa = x / (10**exponent)
        return f"{mantissa:.3g}e{exponent:+d}"


def parse_landcover_composition(vector: List[float]) -> List[float]:
    """
    Parses a vector representing land cover composition into a standardized format.

    Args:
        vector (List[float]): 
            A vector representing land cover composition. 
            It should be either a 9-element vector or a 27-element vector.

    Returns:
        List[float]: 
            A 9-element vector representing the parsed land cover composition.

    Raises:
        ValueError: If the provided vector does not have a length of 9 or 27.

    Note:
        - If the input vector length is 9, it is returned as-is.
        - If the input vector length is 27, it is split into three equal parts,
          each part is summed element-wise, resulting in a 9-element vector.
    """
    vector_length = len(vector)
    supported_vector_lengths = (9,27)
    try:
        assert vector_length in supported_vector_lengths
    except AssertionError as exc:
        vector_sizes: str = ", ".join(map(str, supported_vector_lengths))
        raise ValueError(
            f"Supported vector sizes: {vector_sizes}. "
            f"Provided vector has size {vector_length}") from exc
    if vector_length == 9:
        return vector
    # Fold the 27x1 vector into three parts and sum the 9x1 vectors
    len_list = vector_length // 3
    sublist1 = vector[:len_list]
    sublist2 = vector[len_list:2*len_list]
    sublist3 = vector[2*len_list:]
    return reduce(
        lambda acc, val: 
            [acc[i] + val[i] for i in range(len(acc))], 
            [sublist1, sublist2, sublist3])


def enforce_unity_sum(vector: List[float], epsilon: float = 0.001) -> List[float]:
    """
    Ensures the sum of values in the vector is approximately 1.0 within a given epsilon 
    tolerance.

    Args:
        vector (List[float]): The input vector of float values to be normalized.
        epsilon (float, optional): 
            Tolerance level around 1.0 within which the sum of the vector
            should lie. Defaults to 0.001.

    Returns:
        List[float]: A normalized vector where the sum of values is approximately 1.0.

    Note:
        - If the sum of values in the input vector is not within the range [1.0 - epsilon, 1.0 + epsilon],
          the vector is normalized by dividing each element by the current sum of the vector.

    """
    if not 1.0 - epsilon <= sum(vector) <= 1.0 + epsilon:
        # TODO: Add warning, maybe to a logger.
        return [value/sum(vector) for value in vector]
    return vector


landcover_cmap = plt.get_cmap('Spectral')
landcover_colors = [landcover_cmap(i) for i in np.linspace(0, 1, 9)]


def landcover_pie(
        axis: plt.Axes, values: List[float], labels: List[str], colors: List[Any],
        title: str | None = None, show_legend: bool = False) -> None:
    """
    Creates a pie chart of land cover composition.

    Args:
        axis (plt.Axes): The matplotlib axis object where the pie chart will be drawn.
        values (List[float]): List of float values representing the proportions of each land cover type.
        labels (List[str]): List of strings representing labels for each land cover type.
        colors (List[Any]): List of colors corresponding to each land cover type.
        title (str, optional): Title of the pie chart. Defaults to None.
        show_legend (bool, optional): Whether to display the legend. Defaults to False.

    Returns:
        None

    Note:
        - The function creates a pie chart using matplotlib on the provided axis (`axis`).
        - Only land cover types with non-zero values (`values > 0`) are included in the chart.
        - The pie is exploded to highlight the land cover type with the highest proportion.
        - Labels and percentages are formatted and positioned for readability.

    """
    explode_offset: float = 0.05
    labels_tr = [label for label, value in zip(labels, values) if value >0]
    values_tr = [value for value in values if value >0]
    colors_tr = [color for color, value in zip(colors, values) if value >0]
    # Explode the pie with the highest value
    max_value_index = values_tr.index(max(values_tr))
    one_hot_list = [0] * len(values_tr)
    one_hot_list[max_value_index] = 1
    explode = [explode_offset * item for item in one_hot_list]

    axis.pie(values_tr, 
        labels=labels_tr, 
        explode=explode,
        pctdistance = 0.6, 
        labeldistance = 1.1,
        wedgeprops = {"edgecolor":"k",'linewidth': 0.5},
        startangle=180,
        autopct=lambda x: f'{x:.1f}%',
        textprops={"size": 8}, 
        radius = 1.0,
        colors = colors_tr
        )
    if title:
        axis.set_title(title, fontsize=9)
    if show_legend:
        legend = axis.legend(
            loc="lower right", frameon=False, bbox_to_anchor=(1.5,0), borderaxespad=0.5)
        for text in legend.get_texts():
            text.set_fontstyle("italic")
            text.set_fontsize(8)


def emission_profile(
        axis: plt.Axes,
        years: List[int],
        emissions: List[float],
        title: str,
        y_label: str,
        annotate: bool = True) -> None:
    """ Plots a profile of emissions over time.

    Args:
        axis (plt.Axes): Matplotlib axes object.
        years (List[int]): List of years corresponding to the emissions.
        emissions (List[float]): List of emissions corresponding to the years.
        title (str): Title of the plot.
        y_label (str): Label for the y-axis.
        annotate (bool): Flag setting whether emission values are added to plot. Default is True.
    Raises:
        ValueError: If the lengths of years and emissions do not match, or if they contain non-numeric values.
    """
    plot_options = {
        'linewidth': 1.0, 
        'axiswidth': 2, 
        'tickwidth': 2,
        'labelpad': 5, 
        'titlepad': 15,
        'label_fontsize': LABEL_FONTSIZE,
        'title_fontsize': TITLE_FONTSIZE,
        'tick_fontsize': TICK_FONTSIZE,
        'annotation_fontsize': ANNOTATION_FONTSIZE}

    # Get the x and y data
    if len(years) != len(emissions):
        raise ValueError(
            f"Years and emissions must have the same length, got {len(years)} and {len(emissions)}")
    if not all(isinstance(year, (int, float)) for year in years):
        raise ValueError("All emissions must be numeric (int or float).")
    if not all(isinstance(emission, (int, float)) for emission in emissions):
        raise ValueError("All emissions must be numeric (int or float).")
    x_data = years
    y_data = emissions
    # Format plot
    axis.plot(x_data, y_data, '-', color='k',
                linewidth=plot_options['linewidth'])
    axis.plot(x_data, y_data, marker='o', color='r')
    axis.set_ylabel(y_label, fontsize=plot_options['label_fontsize'],
                    labelpad=plot_options['labelpad'])
    axis.set_xlabel('Time, years', fontsize=plot_options['label_fontsize'],
                    labelpad=plot_options['labelpad'])
    axis.set_title(title, fontsize=plot_options['title_fontsize'],
                    pad=plot_options['titlepad'])
    axis.tick_params(axis="both", labelsize=plot_options['tick_fontsize'])
    # Make the (visible) axes thicker
    for axis_pos in ['bottom', 'left']:
        axis.spines[axis_pos].set_linewidth(plot_options['axiswidth'])
    # Increase tick width
    axis.tick_params(width=plot_options['tickwidth'])
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    plt.grid(True)
    # Add emission values (numbers) to the plot
    if annotate:
        for x_coord, y_coord in zip(x_data, y_data):
            axis.annotate(
                r'${:.1f}$'.format(y_coord),
                xy=(x_coord, y_coord),
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=plot_options['annotation_fontsize'])
            
def emission_bars(
        axis: plt.Axes,
        gases: List[str],
        values: List[float],
        output_name: str) -> None:
    """Visualize total emissions (unit x surface area) for the calculated gases.

    Args:
        axis (plt.Axes): Matplotlib axes object.
        gases (List[str]): List of gas names to be displayed on the y-axis.
        values (List[float]): List of total annual emissions for each gas.
        output_name (str): Name of the output/reservoir.
    """
    plot_options = {
        'linewidth': 1.0,
        'axiswidth': 2,
        'tickwidth': 2,
        'labelpad': 5, 
        'titlepad': 15,
        'label_fontsize': LABEL_FONTSIZE,
        'title_fontsize': TITLE_FONTSIZE,
        'tick_fontsize': TICK_FONTSIZE,
        'annotation_fontsize': ANNOTATION_FONTSIZE}

    y_pos = np.arange(len(gases))[::-1]
    axis.barh(y_pos, values, color=(0.2, 0.4, 0.6, 0.6), edgecolor='blue')
    axis.set_yticks(y_pos, gases)
    #axis.tick_params(fontsize=plot_options['tick_fontsize'])
    axis.tick_params(axis="both", labelsize=plot_options['tick_fontsize'])
    axis.set_xlabel("Total annual emission, tonnes CO$_{2,eq}$ yr$^{-1}$",
                    fontsize=plot_options['label_fontsize'])
    axis.set_ylabel("Gas", fontsize=plot_options['label_fontsize'])
    axis.set_title("Total annual emission, {}".format(output_name),
                    fontsize=plot_options['title_fontsize'],
                    pad=plot_options['titlepad'])
    # Make the (visible) axes thicker
    for axis_pos in ['bottom', 'left']:
        axis.spines[axis_pos].set_linewidth(plot_options['axiswidth'])
    # Increase tick width
    axis.tick_params(width=plot_options['tickwidth'])
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)


@dataclass  # type: ignore[misc]
class Writer(ABC):
    """
    Abstract base class for all writers.
    """

    @staticmethod
    def round_parameter(
            number: Union[float, list],
            number_decimals: int) -> Union[float, list]:
        """
        Rounds a number or each element in a list to the specified number of decimals.

        Args:
            number (Union[float, list]): The number or list of numbers to round.
            number_decimals (int): Number of decimal places to round to.

        Returns:
            Union[float, list]: Rounded number or list of rounded numbers.
        """
        try:
            if isinstance(number, list):
                number = [round(num, number_decimals) for num in number]
            else:
                number = round(number, number_decimals)
        except TypeError:
            pass
        return number

    @staticmethod
    def rollout(
            var_name: str,
            var_vector: Union[Sequence, Dict]) -> Iterator[Tuple[str, Any]]:
        """
        Creates variable names and values from a sequence or dictionary.

        Args:
            var_name (str): Base name for the variables.
            var_vector (Union[Sequence, Dict]): Sequence or dictionary containing variable values.

        Yields:
            Iterator[Tuple[str, Any]]: Iterator yielding tuples of variable names and values.
            
        Example:
            a variable var = [3,4,5] becomes: ('var_0', 3), ('var_1', 4),
            ('var_2', 5).
        """
        if isinstance(var_vector, Sequence):
            for iter_var, var in enumerate(var_vector):
                yield '_'.join([var_name, str(iter_var)]), var

    @staticmethod
    def write_par_to_dict(input_name: str, parameter: Any, par_dict: Dict,
                          reservoir_name: str, precision: int = 3) -> Dict:
        """
        Writes a parameter to a dictionary of parameter name-value pairs.

        Args:
            input_name (str): Name of the input parameter.
            parameter (Any): Parameter data. It can be a single value, sequence, or nested sequence.
            par_dict (Dict): Dictionary of parameter names and values in the form {reservoir_name: {par_name: par_values}}.
            reservoir_name (str): Name of the reservoir for which the parameter is to be added.
            precision (int, optional): Number of decimal points for the parameter. Defaults to 3.

        Returns:
            Dict: Updated dictionary of parameter names and values.
        """
        if isinstance(parameter, str):
            param_dict = {input_name: parameter}
            par_dict[reservoir_name].update(param_dict)
        elif isinstance(parameter, collections.abc.Sequence):
            nested_list: bool = False
            for val in parameter:
                if isinstance(val, collections.abc.Sequence):
                    nested_list = True
                    break
            if nested_list is True:
                parameter = rollout_nested_list(parameter, None)
            for name, value in Writer.rollout(input_name, parameter):
                param_dict = {name: Writer.round_parameter(
                    value, precision)}
                par_dict[reservoir_name].update(param_dict)
        else:
            param_dict = {input_name: Writer.round_parameter(
                parameter, precision)}
            par_dict[reservoir_name].update(param_dict)
        return par_dict

    @abstractmethod
    def write(self) -> None:
        """
        Writes outputs to the format of choice.
        """


@dataclass  # type: ignore[misc]
class ExcelWriter(Writer):
    """
    Formats and writes data to an Excel file.

    Attributes:
        inputs (Inputs): Inputs object containing input data.
        outputs (Dict): Outputs dictionary.
        intern_vars (Dict): Internal variables dictionary.
        output_file_path (str): Path where the output Excel file will be saved.
        output_config (Dict): Configuration file with formatting settings for outputs.
        input_config (Dict): Configuration file with formatting settings for inputs.
        intern_vars_config (Dict): Configuration dictionary with formatting settings for internal variables.
        parameter_config (Dict): Configuration file with formatting settings for parameters.
        config_ini (Dict): Global parameter configuration dict (usually from .ini file).
        author (str): Author of the Excel document.
        title (str): Title of the Excel document.
        output_df (pd.DataFrame): DataFrame containing model outputs.
        input_df (pd.DataFrame): DataFrame containing model inputs.
        intern_vars_df (pd.DataFrame): DataFrame containing internal variables.
    """

    inputs: Inputs
    outputs: Dict
    intern_vars: Dict

    output_file_path: str

    output_config: Dict
    input_config: Dict
    intern_vars_config: Dict

    parameter_config: Dict
    config_ini: Dict
    author: str
    title: str

    output_df: pd.DataFrame = field(init=False)
    input_df: pd.DataFrame = field(init=False)
    intern_vars_df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """
        Initialize output DataFrames which will be output as Excel sheets.
        """
        self.output_df = pd.DataFrame()
        self.input_df = pd.DataFrame()
        self.intern_vars_df = pd.DataFrame()

    def add_inputs(self, reservoir_name: str) -> None:
        """
        Create an inputs DataFrame for a given reservoir and append it to the main input DataFrame.

        Args:
            reservoir_name (str): Name of the reservoir for which inputs are to be added.
        """
        input_data = self.inputs.inputs[reservoir_name].data
        input_dict: Dict = {reservoir_name: {}}

        # Find out which inputs to include in the presentation
        included_inputs = []
        for input_name in INPUT_NAMES:
            input_data_conf = self.input_config[input_name]
            if input_data_conf['include']:
                included_inputs.append(input_name)
        if len(included_inputs) < 1:
            return None

        # Mapping between input names and input names in the Input.data dict.
        config_to_data = {"coordinates": "coordinates",
                          "id": "id",
                          "type": "type",
                          "monthly_temps": "monthly_temps",
                          "year_profile": "year_vector",
                          "gasses": "gasses"}
        for input_name, input_name_mapped in config_to_data.items():
            if input_name in included_inputs:
                try:
                    parameter_value = input_data[input_name_mapped]
                except (KeyError, AttributeError):
                    log.error("Input name %s not found. Skipping",
                              input_name_mapped)
                input_dict = self.write_par_to_dict(
                    input_name=input_name,
                    parameter=parameter_value,
                    par_dict=input_dict,
                    reservoir_name=reservoir_name,
                    precision=EXCEL_NUMBER_DECIMALS)

        # Add catchment inputs
        if 'catchment_inputs' in included_inputs:
            # Get input data
            for input_name, input_value in input_data['catchment'].items():
                if input_name == 'biogenic_factors':
                    # Iterate through biogenic facors
                    # TODO: Fix this piece of code to work with the Biogenic object data
                    # Currently switched off as it produces errors
                    break
                    for factor_name, factor_value in \
                            input_data['catchment'][input_name].items():
                        log.info(factor_name)
                        input_dict = self.write_par_to_dict(
                            input_name=factor_name,
                            parameter=factor_value,
                            par_dict=input_dict,
                            reservoir_name=reservoir_name,
                            precision=EXCEL_NUMBER_DECIMALS)
                else:
                    input_dict = self.write_par_to_dict(
                        input_name='_'.join(['catch', input_name]),
                        parameter=input_value,
                        par_dict=input_dict,
                        reservoir_name=reservoir_name,
                        precision=EXCEL_NUMBER_DECIMALS)

        # Add reservoir inputs
        if 'reservoir_inputs' in included_inputs:
            # Get input data
            for input_name, input_value in input_data['reservoir'].items():
                input_dict = self.write_par_to_dict(
                    input_name='_'.join(['res', input_name]),
                    parameter=input_value,
                    par_dict=input_dict,
                    reservoir_name=reservoir_name,
                    precision=EXCEL_NUMBER_DECIMALS)

        reservoir_df = pd.DataFrame.from_dict(input_dict, orient='index')
        reservoir_df.index.names = ['Name']
        if self.input_df.empty:
            self.input_df = reservoir_df
        else:
            self.input_df = pd.concat([self.input_df, reservoir_df])

    def dict_data_to_df(
            self, id: str, data: Dict, config: Dict, 
            index_name: str = 'Name'
        ) -> pd.DataFrame:
        """
        Parse data in a dictionary format into pandas DataFrame format.

        Args:
            id (str): Identifier for the data (e.g., reservoir name).
            data (Dict): Dictionary containing data to be parsed into DataFrame.
            config (Dict): Configuration dictionary specifying which parameters to include.
            index_name (str, optional): Name of the index column in the DataFrame. Defaults to 'Name'.

        Returns:
            pd.DataFrame: DataFrame containing parsed data.
        """
        output: Dict = {id: {}}
        for parameter, parameter_value in data.items():
            # Add parameter if the parameter is marked for presentation
            if config[parameter]['include']:
                # If variable is a sequence, rollout to multiple variables
                # stored in individual columns
                if isinstance(parameter_value, collections.abc.Sequence) and \
                        not isinstance(parameter_value, str):
                    for name, value in self.rollout(
                            parameter, parameter_value):
                        param_dict = {name: self.round_parameter(
                            value, EXCEL_NUMBER_DECIMALS)}
                        output[id].update(param_dict)
                else:
                    param_dict = {parameter: self.round_parameter(
                        parameter_value, EXCEL_NUMBER_DECIMALS)}
                    output[id].update(param_dict)
        output_df = pd.DataFrame.from_dict(output, orient='index')
        output_df.index.names = [index_name]
        return output_df
    
    def add_outputs(self, reservoir_name: str) -> None:
        """
        Add outputs selected for presentation for a given reservoir.

        Args:
            reservoir_name (str): Name of the reservoir for which outputs are to be added.
        """
        config = self.output_config['outputs']
        data = self.outputs[reservoir_name]
        reservoir_df = self.dict_data_to_df(
            id=reservoir_name, data=data, config=config, index_name='Name')
        if self.output_df.empty:
            self.output_df = reservoir_df
        else:
            self.output_df = pd.concat([self.output_df, reservoir_df])

    def add_internals(self, reservoir_name: str) -> None:
        """
        Add internal variables selected for presentation for a given reservoir.

        Args:
            reservoir_name (str): Name of the reservoir.
        """
        config = self.intern_vars_config
        data=self.intern_vars[reservoir_name]
        reservoir_df = self.dict_data_to_df(
            id=reservoir_name, data=data, config=config, index_name='Name')
        if self.intern_vars_df.empty:
            self.intern_vars_df = reservoir_df
        else:
            self.intern_vars_df = pd.concat([self.intern_vars_df, reservoir_df])

    def write(self) -> None:
        """
        Write input/output data (all reservoirs) to an Excel file.
        """
        if not bool(self.outputs):
            log.error("Attempting to write before generating outputs")
            return None
        for reservoir_name in self.outputs:
            self.add_inputs(reservoir_name=reservoir_name)
            self.add_outputs(reservoir_name=reservoir_name)
            self.add_internals(reservoir_name=reservoir_name)
        with pd.ExcelWriter(self.output_file_path) as writer:
            self.input_df.to_excel(writer, sheet_name="inputs", index=True)
            self.output_df.to_excel(writer, sheet_name="outputs", index=True)
            self.intern_vars_df.to_excel(writer, sheet_name="internals", index=True)
        log.info("Created a Excel file with outputs.")
        return None


@dataclass  # type: ignore[misc]
class JSONWriter(Writer):
    """
    Format and write data to a JSON file.

    Attributes:
        inputs (Inputs): Inputs object with input data.
        outputs (Dict): Outputs dictionary.
        intern_vars (Dict): Internal variables dictionary.
        output_file_path (str): Path where the output file is to be saved/written to.
        output_config (Dict): Configuration dictionary with formatting settings for outputs.
        input_config (Dict): Configuration dictionary with formatting settings for inputs.
        intern_vars_config (Dict): Configuration dictionary with formatting settings for internal variables.
        parameter_config (Dict): Configuration file with formatting settings for parameters.
        config_ini (Dict): Global parameter configuration dict (usually from .ini file).
        author (str): Author of the document.
        title (str): Title of the document.
        json_dict (Dict): Output dictionary saved to the output JSON file. Automatically initialized.
    """

    inputs: Inputs
    outputs: Dict
    intern_vars: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    intern_vars_config: Dict
    parameter_config: Dict
    config_ini: Dict
    author: str
    title: str
    json_dict: Dict = field(init=False)

    def __post_init__(self):
        """
        Initialize json_dict which will be output as JSON.
        """
        self.json_dict = {}

    def add_inputs(self, reservoir_name: str) -> None:
        """
        Add inputs selected for presentation for a given reservoir to json_dict.

        Args:
            reservoir_name (str): Name of the reservoir.
        """
        input_data = self.inputs.inputs[reservoir_name].data
        input_dict: dict = {'inputs': {}}
        # Find out which inputs to include in the presentation
        included_inputs = []
        for input_name in INPUT_NAMES:
            input_data_conf = self.input_config[input_name]
            if input_data_conf['include']:
                included_inputs.append(input_name)
        if len(included_inputs) < 1:
            return None
        # Mapping between input names and input names in the Input.data dict.
        config_to_data = {"coordinates": "coordinates",
                          "id": "id",
                          "type": "type",
                          "monthly_temps": "monthly_temps",
                          "year_profile": "year_vector",
                          "gasses": "gasses"}
        for input_name, input_name_mapped in config_to_data.items():
            if input_name in included_inputs:
                param_dict = {}
                param_dict['name'] = self.input_config[input_name]['name']
                param_dict['unit'] = self.input_config[input_name]['unit']
                param_dict['value'] = input_data[input_name_mapped]
                input_dict['inputs'][input_name] = param_dict
        # Add biogenic factors
        if 'biogenic_factors' in included_inputs:
            param_dict = {}
            param_dict['name'] = self.input_config['biogenic_factors']['name']
            # Check if biogenic factors are of BiogenicFactors type
            # instead of a dictionary - the data is a dictionary in
            # text input files but then is converted to BiogenicFactors
            # type
            biogenic_factors = input_data['catchment']['biogenic_factors']
            if not isinstance(biogenic_factors, dict):
                try:
                    biogenic_factors = biogenic_factors.todict()
                except AttributeError:
                    log.error('Variable biogenic factors cannot be ' +
                              'converted to a dictionary')
            for input_name, input_value in biogenic_factors.items():
                param_dict[input_name] = {}
                param_dict[input_name]['name'] = FACTOR_NAMES[input_name]
                param_dict[input_name]['unit'] = ''
                param_dict[input_name]['value'] = input_value
            input_dict['inputs']['biogenic_factors'] = param_dict
        # Add catchment inputs
        if 'catchment_inputs' in included_inputs:
            param_dict = {}
            param_dict['name'] = self.input_config['catchment_inputs']['name']
            # Get input data
            for input_name, input_value in input_data['catchment'].items():
                if input_name == 'biogenic_factors':
                    break
                param_dict[input_name] = {}
                # Get input name and unit from config
                conf_input = self.input_config[
                    'catchment_inputs']['var_dict'][input_name]
                param_dict[input_name]['name'] = conf_input['name']
                param_dict[input_name]['unit'] = conf_input['unit']
                if isinstance(input_value, Iterable) and not \
                        isinstance(input_value, str):
                    input_value = ', '.join([str(item) for item in input_value])
                param_dict[input_name]['value'] = input_value
            input_dict['inputs']['catchment_inputs'] = param_dict
        # Add reservoir inputs
        if 'reservoir_inputs' in included_inputs:
            param_dict = {}
            param_dict['name'] = self.input_config['reservoir_inputs']['name']
            # Get input data
            for input_name, input_value in input_data['reservoir'].items():
                param_dict[input_name] = {}
                # Get input name and unit from config
                conf_input = self.input_config[
                    'reservoir_inputs']['var_dict'][input_name]
                param_dict[input_name]['name'] = conf_input['name']
                param_dict[input_name]['unit'] = conf_input['unit']
                if isinstance(input_value, Iterable) and not \
                        isinstance(input_value, str):
                    input_value = ', '.join([str(item) for item in input_value])
                param_dict[input_name]['value'] = input_value
            input_dict['inputs']['reservoir_inputs'] = param_dict
        try:
            self.json_dict[reservoir_name].update(input_dict)
        except KeyError:
            self.json_dict[reservoir_name] = input_dict

    def parse_dict_data(self, config: Dict, data: Dict, item_names: Tuple) -> Dict:
        """
        Parse data dictionary into a structured format based on configuration.

        Args:
            config (Dict): Configuration dictionary indicating which parameters to include.
            data (Dict): Dictionary containing data to be parsed.
            item_names (Tuple): Tuple of item names to include in the parsed output.

        Returns:
            Dict: Parsed dictionary data.
        """
        output_dict: Dict = {}
        for parameter, parameter_value in data.items():
            if config[parameter]['include']:
                param_dict: Dict = {parameter: {}}
                for item_name in item_names:
                    item = config[parameter].get(item_name)
                    param_dict[parameter][item_name] = item
                param_dict[parameter]['value'] = self.round_parameter(
                    parameter_value, JSON_NUMBER_DECIMALS)
                output_dict.update(param_dict)
        return output_dict
            
    def add_outputs(self, reservoir_name: str) -> None:
        """
        Add outputs selected for presentation for a given reservoir to `json_dict`.

        Args:
            reservoir_name (str): Name of the reservoir.
        """
        config = self.output_config['outputs']
        data = self.outputs[reservoir_name]
        parsed_data = self.parse_dict_data(
            config=config ,data=data,
            item_names=('name', 'gas_name', 'unit', 'long_description'))
        try:
            self.json_dict[reservoir_name].update({'outputs': parsed_data})
        except KeyError:
            self.json_dict[reservoir_name] = {'outputs': parsed_data}

    def add_internals(self, reservoir_name: str) -> None:
        """
        Add internal variables selected for presentation for a given reservoir to `json_dict`.

        Args:
            reservoir_name (str): Name of the reservoir.
        """
        config = self.intern_vars_config
        data = self.intern_vars[reservoir_name]
        parsed_data = self.parse_dict_data(
            config=config ,data=data,
            item_names=('name', 'unit', 'long_description'))
        try:
            self.json_dict[reservoir_name].update({'intern_vars': parsed_data})
        except KeyError:
            self.json_dict[reservoir_name] = {'intern_vars': parsed_data}

    def write(self) -> None:
        """
        Write output data (all reservoirs) to a JSON file.
        """
        if not bool(self.outputs):
            log.error("Attempting to write before generating outputs.")
            return None
        for reservoir_name in self.outputs:
            self.add_inputs(reservoir_name=reservoir_name)
            self.add_outputs(reservoir_name=reservoir_name)
            self.add_internals(reservoir_name=reservoir_name)
        with open(self.output_file_path,
                  'w', encoding='utf-8') as file_pointer:
            json.dump(self.json_dict, file_pointer, indent=4)
        log.info("Created a json file with outputs.")
        return None


@dataclass  # type: ignore[misc]
class LatexWriter(Writer):
    """Format and write data to a LaTeX file using PyLaTeX.

    Attributes:
        inputs (Inputs): Inputs object with input data.
        outputs (Dict): Outputs dictionary.
        intern_vars (Dict): Internal variables dictionary.
        output_file_path (str): Path where the output file is to be saved/written to.
        output_config (Dict): Configuration file with formatting settings.
        input_config (Dict): Configuration file with formatting settings.
        intern_vars_config (Dict): Configuration dictionary with formatting settings.
        parameter_config (Dict): Configuration file with formatting settings.
        config_ini (Dict): Global parameter configuration dict (usually from .ini file).
        author (str): Author of the document.
        title (str): Title of the document.
        document (Document): PyLaTeX Document object.
    """

    inputs: Inputs
    outputs: Dict
    intern_vars: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    intern_vars_config: Dict
    parameter_config: Dict
    config_ini: Dict
    author: str
    title: str

    def __post_init__(self) -> None:
        """Initialize the PyLaTeX document with specified geometry."""
        path_no_ext = os.path.splitext(self.output_file_path)[0]
        self.document = Document(path_no_ext, geometry_options=self.geometry())

    @staticmethod
    def geometry() -> Dict:
        """Create document geometry structure.

        Returns:
            Dict: Dictionary containing the document geometry options.
        """
        document_geometry = {
            "head": "0.0in", "margin": "0.75in", "top": "0.55in",
            "bottom": "0.55in", "includeheadfoot": True}
        return document_geometry

    def plot_profile(
            self, 
            axis: plt.Axes, 
            emission: str, 
            output_name: str,
            annotate: bool = True) -> None:
        """Plot an emission profile using matplotlib.

        Args:
            axis (plt.Axes): Matplotlib axes object.
            emission (str): Name of the gas/emission to plot.
            output_name (str): Name of the output/reservoir.
            annotate (bool): Flag setting whether emission values are added to plot. Default is True.
        """
        emission_var = {
            "co2": "co2_profile",
            "ch4": "ch4_profile",
            "n2o": "n2o_profile"}       
        emission_unit = self.output_config['outputs'][
            emission_var[emission]]['unit_latex']
        data = self.outputs[output_name]
        title = ", ".join((self.output_config['outputs'][
            emission_var[emission]]['name_latex'], output_name))
        # Create title and y_label
        y_label = "Emission, " + emission_unit
        # Get the x and y data
        try:
            y_data = data[emission_var[emission]]
            x_data = self.inputs.inputs[output_name].data["year_vector"]
        except KeyError:
            log.warning(
                "Problem with plotting emission profiles: X or Y data for emission profiles not found.")
            return None
        emission_profile(
            axis=axis, years=x_data, emissions=y_data,
            title=title, y_label=y_label, annotate=annotate)

    def plot_emission_bars(self, axis: plt.Axes, output_name: str) -> None:
        """Visualize total emissions (unit x surface area) for the calculated gases.

        Args:
            axis (plt.Axes): Matplotlib axes object.
            output_name (str): Name of the output/reservoir.
        """
        data = self.outputs[output_name]
        vars_to_plot = ('co2_net', 'ch4_net', 'n2o_mean')
        bars = [self.output_config['outputs'][var]['gas_name_latex'] for
                var in vars_to_plot if var in data]
        # Get reservoir area from inputs (convert from km2 to m2)
        area = self.inputs.inputs[output_name].data['reservoir']['area'] * \
            10**6
        values = [data[var] * area * 10 ** (-6) for var in vars_to_plot
                  if var in data]
        emission_bars(
            axis=axis, gases=bars, values=values,
            output_name=output_name)


    def plot_landcover_piecharts(self, axes: np.ndarray, output_name: str) -> None:
        """Plot pie charts of landcover compositions for reservoir and catchment.

        Args:
            axes (np.ndarray): Array of matplotlib axes objects.
            output_name (str): Name of the output/reservoir.
        """
        #data = self.inputs.data[output_name]
        input_data = self.inputs.inputs[output_name]
        landcovers_reservoir = parse_landcover_composition(
            input_data.data['reservoir']['area_fractions'])
        landcovers_catchment = parse_landcover_composition(
            input_data.data['catchment']['area_fractions'])
        # Make sure that the sums are equal to one (they should be because of earlier steps)
        # But just for the piece of mind, make an additional check
        landcovers_reservoir = enforce_unity_sum(landcovers_reservoir)
        landcovers_catchment = enforce_unity_sum(landcovers_catchment)
        landcover_pie(
            axis = axes[0], values=landcovers_reservoir, labels=landcover_names,
            colors=landcover_colors, 
            show_legend=True, title="Reservoir\nLandcover Composition")
        landcover_pie(
            axis = axes[1], values=landcovers_catchment, labels=landcover_names,
            colors=landcover_colors, 
            show_legend=True, title="Catchment\nLandcover Composition")


    def add_plots(self, output_name: str, plot_fraction: float = 0.75,
                  dpi: int = 300) -> None:
        """Generate and add emission plots to the document.

        Args:
            output_name (str): Name of the output/reservoir.
            plot_fraction (float): Fraction of text width the plot takes on the page. Default is 0.75.
            dpi (int): Resolution of the created figure within PDF. Default is 300.
        """
        profile_plots = bool(self.output_config['global']['plot_profiles'])
        bar_plot = bool(self.output_config['global']['plot_emission_bars'])
        # Find out how many subplots to define
        no_plots = 0
        if profile_plots:
            # Find a number of gases that are computed (for the
            # specified output name)
            no_plots += len(self.inputs.inputs[output_name].data['gasses'])
        if bar_plot:
            no_plots += 1
        if no_plots == 0:
            return None
        fig = plt.figure()
        if no_plots <= 2:
            subplot_dim = (1, 2)
        elif no_plots <= 4:
            subplot_dim = (2, 2)
        with self.document.create(Subsection('Emission plots')):
            fig_index = 1
            if profile_plots:
                for gas in ['co2', 'ch4', 'n2o']:
                    if gas not in self.inputs.inputs[output_name].data['gasses']:
                        continue
                    axis = fig.add_subplot(*subplot_dim, fig_index)
                    self.plot_profile(
                        axis=axis, emission=gas, output_name=output_name,
                        annotate=False)
                    fig_index += 1
            if bar_plot:
                axis = fig.add_subplot(*subplot_dim, fig_index)
                self.plot_emission_bars(axis=axis, output_name=output_name)

            fig.tight_layout(pad=1.0)
            with self.document.create(Figure(position='htbp')) as plot:
                width = r'{}\textwidth'.format(plot_fraction)
                plot.add_plot(width=NoEscape(width), dpi=dpi)
        # TODO: There's an intermittent error with profiles not being plotted and only
        # bar plot appearing in the report. Could be sth. to do with overwriting axes
        # as profile plot is added with add_subplot method. 
        plt.close()
        return None
        
    def add_landcover_charts(self, output_name: str, plot_fraction: float = 0.95,
                  dpi: int = 300) -> None:
        """Add pie charts with landcover proportions for reservoir and catchment.

        Args:
            output_name (str): Name of the output/reservoir.
            plot_fraction (float): Fraction of text width the plot takes on the page. Default is 0.95.
            dpi (int): Resolution of the created figure within PDF. Default is 300.
        """
        plot_piecharts = bool(self.output_config['global']['plot_landcover_piecharts'])
        if not plot_piecharts: # Playing the non-indent guy.
            return None
        fig, axs = plt.subplots(1,2)
        self.plot_landcover_piecharts(axes=axs, output_name=output_name)
        fig.tight_layout(pad=1.0)
        #with self.document.create(Subsection('Landcover composition plots')):
        with self.document.create(Figure(position='h')) as plot:
            width = r'{}\textwidth'.format(plot_fraction)
            plot.add_plot(width=NoEscape(width), dpi=dpi)
        plt.close()
        return None
        
    def add_parameters(self, precision: int = 4) -> None:
        """Add information about model parameters such as conversion factors
        and other useful details to the report.

        Args:
            precision (int): Number of decimal points in output parameters. Default is 4.
        """
        round_options = {'round-precision': precision,
                         'round-mode': 'figures'}
        # TODO: Currently only supports two parameters
        #       If we would like to output more/all parameters, this will have
        #       to be made more generic. Potentially .ini file could be
        #       translated into yaml
        if self.parameter_config['parameters']['gwp100']['include']:
            with self.document.create(
                    Subsection(
                        'Global Warming Potentials (GWPs) over 100 years')):
                with self.document.create(Description()) as desc:
                    try:
                        desc.add_item(
                            NoEscape("GWP100 for CO$_2$: "),
                            safe_cast(self.config_ini['CARBON_DIOXIDE']['co2_gwp100'], float))
                        desc.add_item(
                            NoEscape("GWP100 for CH$_4$: "),
                            safe_cast(self.config_ini['METHANE']['ch4_gwp100'], float))
                        desc.add_item(
                            NoEscape("GWP100 for N$_2$O: "),
                            safe_cast(self.config_ini['NITROUS_OXIDE']['nitrous_gwp100'], float))
                    except configparser.NoSectionError:
                        pass
        if self.parameter_config['parameters']['conv_factors']['include']:
            with self.document.create(Subsection("Unit conversion factors")):
                try:
                    with self.document.create(Description()) as desc:
                        conv_coeff = safe_cast(self.config_ini['CARBON_DIOXIDE']['conv_coeff'], float)
                        desc.add_item(
                            NoEscape('Conversion from mg~CO$_2$-C~m$^{-2}$~' +
                                     'd$^{-1}$ to g~CO$_{2,eq}$~m$^{-2}$~' +
                                     'yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options),
                        )
                    with self.document.create(Description()) as desc:
                        conv_coeff = safe_cast(self.config_ini['METHANE']['conv_coeff'], float)
                        desc.add_item(
                            NoEscape('Conversion from mg CH$_4$~m$^{-2}$~' +
                                     'd$^{-1}$ to g~CO$_{2,eq}$~m$^{-2}$~' +
                                     'yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options),
                        )
                    with self.document.create(Description()) as desc:
                        conv_coeff = safe_cast(self.config_ini['NITROUS_OXIDE']['conv_coeff'], float)
                        desc.add_item(
                            NoEscape('Conversion from $\\mu$g~N$_2$O~' +
                                     'm$^{-2}$~d$^{-1}$ to g~CO$_{2,eq}$~' +
                                     'm$^{-2}$~yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options),
                        )
                except configparser.NoSectionError:
                    pass

    def add_intern_vars_table(self, output_name: str, precision: int = 4) -> None:
        """Adds internal variables table to the LaTeX document.

        Args:
            output_name (str): Name of the internal variable/reservoir.
            precision (int): Number of decimal points in the output values.
        """
        round_options = {'round-precision': precision,
                         'round-mode': 'figures'}
        column_names = ["Name", "Unit", "Value"]
        table_format = "p{10cm} X[c] X[l]"
        data = self.intern_vars[output_name]
        with self.document.create(Center()) as centered:
            with centered.create(Tabu(table_format, booktabs=True,
                                      row_height=1.0)) as data_table:
                data_table.add_row(column_names, mapper=[bold])
                data_table.add_hline()
                # Iterate through outputs and generate rows of data to be
                # entered into the table
                for parameter, value in data.items():
                    parameter_name = NoEscape(
                        self.intern_vars_config[parameter]['name_latex'])
                    unit = NoEscape(
                        self.intern_vars_config[parameter]['unit_latex'])
                    if isinstance(value, np.float64):
                        value = float(value)
                    if self.intern_vars_config[parameter]['include'] \
                            and not isinstance(value, Iterable):
                        # Lists describe profiles and should not be put in the
                        # table
                        value = Quantity(value, options=round_options)
                        row = [parameter_name, unit, value]
                        data_table.add_row(row)

    def add_outputs_table(self, output_name: str, precision: int = 4) -> None:
        """Adds outputs table to the LaTeX document.

        Args:
            output_name (str): Name of the output/reservoir.
            precision (int): Number of decimal points in the output values.
        """
        round_options = {'round-precision': precision,
                         'round-mode': 'figures'}
        column_names = ["Name", "Unit", "Value"]
        table_format = "p{10cm} X[c] X[l]"
        data = self.outputs[output_name]
        with self.document.create(Center()) as centered:
            with centered.create(Tabu(table_format, booktabs=True,
                                      row_height=1.0)) as data_table:
                data_table.add_row(column_names, mapper=[bold])
                data_table.add_hline()
                # Iterate through outputs and generate rows of data to be
                # entered into the table
                for parameter, value in data.items():
                    parameter_name = NoEscape(
                        self.output_config['outputs'][parameter]['name_latex'])
                    unit = NoEscape(
                        self.output_config['outputs'][parameter]['unit_latex'])
                    if self.output_config['outputs'][parameter]['include'] \
                            and not isinstance(value, list):
                        # Lists describe profiles and should not be put in the
                        # table
                        value = Quantity(value, options=round_options)
                        row = [parameter_name, unit, value]
                        data_table.add_row(row)
                # Add summed (composite) emission values
                if all(self.output_config['outputs'][key]['include'] for key in \
                    ['co2_ch4', 'co2_net', 'ch4_net']):
                    try:
                        value = Quantity(data['co2_net'] + data['ch4_net'],
                                         options=round_options)
                        unit = NoEscape(
                            self.output_config['outputs']['co2_ch4'][
                                'unit_latex'])
                        row = [NoEscape('CO$_2$+CH$_4$ net emissions'), unit,
                               value]
                        data_table.add_hline()
                        data_table.add_row(row)
                    except KeyError:
                        # Do not output anything if one of the data (either)
                        # CO2 net or CH4 net are not included in the results
                        pass
                if all(self.output_config['outputs'][key]['include'] for key in \
                    ['co2_ch4_n2o', 'co2_net', 'ch4_net', 'n2o_mean']):
                    try:
                        value = Quantity(
                            data['co2_net'] + data['ch4_net'] +
                            data['n2o_mean'], options=round_options)
                        unit = NoEscape(
                            self.output_config['outputs'][
                                'co2_ch4_n2o']['unit_latex'])
                        row = [NoEscape('CO$_2$+CH$_4$+N$_2$O net emissions'),
                               unit, value]
                        data_table.add_hline()
                        data_table.add_row(row)
                    except KeyError:
                        # Do not output anything if one of the data (either)
                        # CO2 net or CH4 net are not included in the results
                        pass

    def add_inputs_table(self, output_name: str, precision: int = 4) -> None:
        """Add information with model inputs (for each reservoir).

        Args:
            output_name (str): Name of the output/reservoir.
            precision (int): Number of decimal points in the output input values.
        """
        round_options = {'round-precision': precision,
                         'round-mode': 'figures'}
        column_names = ["Input Name", "Unit", "Value(s)"]
        table_format = "X[l] X[c] X[l]"
        # Get inputs and input config
        input_data = self.inputs.inputs[output_name].data
        included_inputs = []
        for input_name in INPUT_NAMES:
            input_data_conf = self.input_config[input_name]
            if input_data_conf['include']:
                included_inputs.append(input_name)
        if len(included_inputs) < 1:
            return None
        # If there are inputs to be added to the table, proceed with
        # constructing the table
        with self.document.create(Center()) as centered:
            with centered.create(Tabu(table_format, booktabs=True,
                                      row_height=1.0)) as data_table:
                data_table.add_row(column_names, mapper=[bold])
                data_table.add_hline()
                # Add stuff to the table if they're to be included
                printout = False
                if "id" in included_inputs:
                    name = self.input_config['id']['name']
                    unit = NoEscape(
                        self.input_config['id']['unit_latex'])
                    input_value_str = str(input_data["id"])
                    row = [name, unit, input_value_str]
                    data_table.add_row(row)
                    printout = True
                if 'type' in included_inputs:
                    name = self.input_config['type']['name']
                    unit = NoEscape(
                        self.input_config['type']['unit_latex'])
                    input_value_str = str(input_data["type"])
                    row = [name, unit, input_value_str]
                    data_table.add_row(row)
                    printout = True                    
                if "coordinates" in included_inputs:
                    name = self.input_config['coordinates']['name']
                    unit = NoEscape(
                        self.input_config['coordinates']['unit_latex'])
                    input_value = input_data["coordinates"]
                    input_value_str = "LAT: " + str(input_value[0]) + \
                        ", LON: " + str(input_value[1])
                    row = [name, unit, input_value_str]
                    data_table.add_row(row)
                    printout = True
                if "monthly_temps" in included_inputs:
                    name = self.input_config['monthly_temps']['name']
                    unit = NoEscape(
                        self.input_config['monthly_temps']['unit_latex'])
                    input_value = input_data["monthly_temps"]
                    input_value = ', '.join([str(item) for item in
                                             input_value])
                    row = [name, unit, input_value]
                    data_table.add_row(row)
                    printout = True
                if "year_profile" in included_inputs:
                    name = self.input_config['year_profile']['name']
                    unit = NoEscape(
                        self.input_config['year_profile']['unit_latex'])
                    input_value = input_data["year_vector"]
                    input_value = ', '.join([str(item) for item in
                                             input_value])
                    row = [name, unit, input_value]
                    data_table.add_row(row)
                    printout = True
                if "gasses" in included_inputs:
                    gas_name_latex = {'co2': 'CO$_2$', 'ch4': 'CH$_4$',
                                      'n2o': 'N$_2$O'}
                    name = self.input_config['gasses']['name']
                    unit = NoEscape(self.input_config['gasses']['unit_latex'])
                    input_value = input_data["gasses"]
                    gases_latex = ', '.join(
                        [gas_name_latex[gas] for gas in input_value])
                    row = [name, unit, NoEscape(gases_latex)]
                    data_table.add_row(row)
                    printout = True
                if printout:
                    data_table.add_hline()
                # Add biogenic factors
                if 'biogenic_factors' in included_inputs:
                    row_name = self.input_config['biogenic_factors']['name']
                    data_table.add_row(
                        (MultiColumn(3, align='c', data=row_name),))
                    data_table.add_hline()
                    # Check if biogenic factors are of BiogenicFactors type
                    # instead of a dictionary - the data is a dictionary in
                    # text input files but then is converted to BiogenicFactors
                    # type
                    biogenic_factors = input_data[
                        'catchment']['biogenic_factors']
                    if not isinstance(biogenic_factors, dict):
                        try:
                            biogenic_factors = biogenic_factors.todict()
                        except AttributeError:
                            log.error('Variable biogenic factors cannot be ' +
                                      'converted to a dictionary')

                    for input_name, input_value in biogenic_factors.items():
                        name = FACTOR_NAMES[input_name]
                        unit = '-'
                        row = [name, unit, input_value]
                        data_table.add_row(row)
                    data_table.add_hline()

                # Add catchment inputs
                if 'catchment_inputs' in included_inputs:
                    row_name = self.input_config['catchment_inputs']['name']
                    data_table.add_row(
                        (MultiColumn(3, align='c', data=row_name),))
                    data_table.add_hline()
                    # Get input data
                    for input_name, input_value in \
                            input_data['catchment'].items():
                        if input_name == 'biogenic_factors':
                            break
                        # Get input name and unit from config
                        conf_input = self.input_config[
                            'catchment_inputs']['var_dict'][input_name]
                        name = conf_input['name']
                        unit = NoEscape(conf_input['unit_latex'])
                        if not isinstance(input_value, Iterable):
                            input_value = Quantity(
                                input_value, options=round_options)
                        elif isinstance(input_value, str):
                            pass
                        else:
                            input_value = ', '.join(
                                [str(item) for item in input_value])
                        row = [name, unit, input_value]
                        data_table.add_row(row)
                    data_table.add_hline()

                # Add reservoir inputs
                if 'reservoir_inputs' in included_inputs:
                    row_name = self.input_config['reservoir_inputs']['name']
                    data_table.add_row(
                        (MultiColumn(3, align='c', data=row_name),))
                    data_table.add_hline()
                    # Get input data
                    for input_name, input_value in \
                            input_data['reservoir'].items():
                        # Get input name and unit from config
                        conf_input = self.input_config[
                            'reservoir_inputs']['var_dict'][input_name]
                        name = conf_input['name']
                        unit = NoEscape(conf_input['unit_latex'])
                        if input_value is None:
                            # Convert None into empty string
                            input_value = "N/A"
                        elif not isinstance(input_value, Iterable):
                            input_value = Quantity(
                                input_value, options=round_options)
                        elif isinstance(input_value, str):
                            pass
                        else:
                            input_value = ', '.join(
                                [str(item) for item in input_value])
                        row = [name, unit, input_value]
                        data_table.add_row(row)
        return None

    def add_header(self, header_title: str = "GHG emission estimation results") -> None:
        """
        Adds a header to the LaTeX document source code.

        Args:
            header_title (str): The title to be added in the header. Defaults to "GHG emission estimation results".

        Returns:
            None
        """
        header = PageStyle("header")
        # Create center header
        with header.create(Head("C")):
            header.append(header_title)
        # Create right header
        with header.create(Head("R")):
            header.append(simple_page_number())
        self.document.preamble.append(header)
        self.document.change_document_style("header")

    def add_title_section(self, title: str, author: str) -> None:
        """
        Writes the title section to the LaTeX document source code.

        Args:
            title (str): The title of the document.
            author (str): The author of the document.

        Returns:
            None
        """
        self.document.preamble.append(Command('title', title))
        self.document.preamble.append(Command('author', author))
        self.document.preamble.append(Command('date', NoEscape(r'\today')))
        self.document.append(NoEscape(r'\maketitle'))

    def add_parameters_section(self) -> None:
        """
        Writes model parameter information to the LaTeX document source code.

        Args:
            None

        Returns:
            None
        """
        with self.document.create(Section('Global parameters')):
            self.add_parameters()
        self.document.append(NoEscape(r'\pagebreak'))

    def add_inputs_subsection(self, reservoir_name: str) -> None:
        """
        Writes inputs information to the LaTeX document source code.

        Args:
            reservoir_name (str): The name of the reservoir.

        Returns:
            None
        """
        # Add inputs section to the document
        with self.document.create(Subsection('Inputs')):
            self.add_inputs_table(output_name=reservoir_name)

    def add_outputs_subsection(self, reservoir_name: str) -> None:
        """
        Writes outputs information to the LaTeX document source code.

        Args:
            reservoir_name (str): The name of the reservoir.

        Returns:
            None
        """
        plot_fraction = 0.75
        # Add inputs section to the document
        with self.document.create(Subsection('Outputs')):
            self.add_outputs_table(output_name=reservoir_name)
            self.add_plots(output_name=reservoir_name,
                           plot_fraction=plot_fraction)
            
    def add_intern_var_subsection(self, reservoir_name: str) -> None:
        """
        Writes internal variable information to the LaTeX document source code.

        Args:
            reservoir_name (str): The name of the reservoir.

        Returns:
            None
        """
        with self.document.create(Subsection('Intermediate variables')):
            self.add_intern_vars_table(output_name=reservoir_name)

    def write(self) -> None:
        """
        Writes output data (all reservoirs) to .tex and .pdf files.

        Args:
            None

        Returns:
            None
        """
        # Read data from configuration
        app_config = registry.main_config.get("app_config")
        clean_tex = app_config['latex']['clean_tex']
        compilations = app_config['latex']['compilations']
        if not bool(self.outputs):
            # If outputs are None or an empty dictionary.
            return None
        if is_latex_installed():
            self.add_header()
            self.add_title_section(title=self.title, author=self.author)
            self.add_parameters_section()
            # Iterate through all reservoirs in outputs and write to tex
            for reservoir_name in self.outputs:
                with self.document.create(Section(reservoir_name)):
                    self.add_inputs_subsection(reservoir_name=reservoir_name)
                    self.add_landcover_charts(reservoir_name)
                    self.add_outputs_subsection(reservoir_name=reservoir_name)
                    self.add_intern_var_subsection(reservoir_name=reservoir_name)
            # Generate a PDF (requires a LaTeX compiler present in the system)
            BatchCompiler(self.document).generate_pdf(
                clean_tex=clean_tex, compiler=_get_latex_compiler(), 
                compilations=compilations)
            log.info("Created a PDF file with outputs.")
        else:
            log.error(
                "LaTeX compiler cannot be found in your environment." +
                "'.tex' and '.pdf' files could not be created.")
        return None


@dataclass
class HTMLWriter(Writer):
    """Format and write data to an HTML file."""

    inputs: Inputs
    outputs: Dict
    intern_vars: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    intern_vars_config: Dict
    parameter_config: Dict
    config_ini: Dict
    author: str
    title: str
    html_fragments: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Initialize the HTML content with a basic structure."""
        self.html_fragments.append(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{self.title}</title>
                <style>
                    :root {{
                        --bg-color: #f9f9f9;
                        --text-color: #333;
                        --panel-bg: #ffffff;
                        --border-color: #ccc;
                        --reservoir-links-bg: #eaeaea; /* Default light mode background */
                    }}
                    body {{
                        background-color: var(--bg-color);
                        color: var(--text-color);
                        transition: background-color 0.3s, color 0.3s;
                    }}
                    .dark-mode {{
                        --bg-color: #121212;
                        --text-color: #e0e0e0;
                        --panel-bg: #1e1e1e;
                        --border-color: #444;
                        --reservoir-links-bg: #333333; /* Dark mode background */
                    }}
                    h1, h2, h3 {{
                        color: var(--text-color);
                        border-bottom: 1px solid var(--border-color);
                        padding-bottom: 0.2em;
                    }}
                    h1 {{ font-size: 2.5rem; }}
                    h2 {{ font-size: 2rem; }}
                    h3 {{ font-size: 1.5rem; }}
                    table td, table th {{ font-size: 1.3rem; }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                        background-color: #fff;
                        box-shadow: 0 0 4px rgba(0,0,0,0.1);
                    }}
                    th, td {{
                        border: 1px solid #ccc;
                        padding: 8px 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: var(--panel-bg); /* or a slight variation */
                        color: var(--text-color);
                    }}
                    figure {{
                        margin: 20px 0;
                        padding: 10px;

                        background-color: var(--panel-bg);
                        color: var(--text-color);
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        box-shadow: 0 0 4px rgba(0,0,0,0.05);
                    }}
                    figcaption {{
                        margin-top: 8px;
                        font-style: italic;
                        color: var(--text-color);
                        font-size: 1em;
                    }}
                    .export-btn {{
                        margin-top: 10px;
                        margin-bottom: 20px;
                        padding: 5px 10px;
                        background: #3498db;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }}
                    .toggle-theme {{
                        
                        background: var(--text-color);
                        color: var(--bg-color);
                        position: fixed;
                        top: 10px;
                        right: 20px;
                        z-index: 1001;
                        cursor: pointer;
                        padding: 6px 12px;
                        border: none;
                        border-radius: 4px;
                    }}
                    #reservoir-links {{
                        display: flex;
                        overflow-x: auto;
                        white-space: nowrap;
                        gap: 10px;
                        padding: 10px 20px;
                        background-color: var(--reservoir-links-bg); /* Use the CSS variable */
                        border-bottom: 1px solid var(--border-color);
                        position: sticky;
                        top: 0;
                        z-index: 500;
                         margin-bottom: 20px;
                    }}                    
                    #reservoir-links a {{
                        padding: 6px 12px;
                        background-color: #3498db;
                        color: white;
                        border-radius: 4px;
                        text-decoration: none;
                        font-size: 0.9em;
                        flex: 0 0 auto;
                        transition: background-color 0.2s;
                    }}
                    #reservoir-links a:hover {{
                        background-color: #217dbb;
                    }}
                    details {{
                        margin-bottom: 1.5em;
                        background-color: #ffffff;
                        padding: 1em;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                    }}
                    summary {{
                        font-size: 1.3em;
                        font-weight: bold;
                        cursor: pointer;
                    }}
                    details,
                    table,
                    figure,
                    summary,
                    th,
                    td,
                    figcaption {{
                        background-color: var(--panel-bg);
                        color: var(--text-color);
                        border-color: var(--border-color);
                    }}
                    table, th, td {{
                        border: 1px solid var(--border-color);
                    }}
                    th {{
                        background-color: rgba(255, 255, 255, 0.05); /* subtle header contrast */
                    }}
                    figcaption {{
                        font-style: italic;
                        opacity: 0.8;
                    }}
                </style>
                <script>
                function toggleTheme() {{
                    document.body.classList.toggle('dark-mode');
                }}
                function exportSection(id, name) {{
                    const el = document.getElementById(id);
                    const blob = new Blob([el.innerHTML], {{ type: 'text/html' }});
                    const link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = name + '.html';
                    link.click();
                }}
                MathJax = {{
                    tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }},
                    svg: {{ fontCache: 'global' }}
                }};
                </script>
                <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
            </head>
            <body>
                <h1>{self.title}</h1>
                <p><strong>Author:</strong> {self.author}</p>
                <p><strong>Date:</strong> {self._get_current_date()}</p>
        """)

    @staticmethod
    def _get_current_date() -> str:
        """Get the current date in a readable format."""
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")

    def add_section(self, title: str) -> None:
        """Add a section to the HTML content."""
        self.html_fragments.append(f"<h2>{title}</h2>")

    def add_subsection(self, title: str) -> None:
        """Add a subsection to the HTML content."""
        self.html_fragments.append(f"<h3>{title}</h3>")

    def add_table(self, data: Dict, config: Dict, column_names: Tuple[str, str, str]) -> None:
        """Add a table to the HTML content."""
        self.html_fragments.append("<table>")
        self.html_fragments.append("<tr>" + "".join(f"<th>{col}</th>" for col in column_names) + "</tr>")
        for parameter, value in data.items():
            if parameter in config and config[parameter].get('include', False):
                name = config[parameter]['name']
                unit = config[parameter]['unit_latex']
                if isinstance(value, (list, tuple)):
                    value = ", ".join(map(format_number, value))
                if isinstance(value, str):
                    self.html_fragments.append(f"<tr><td>{name}</td><td>{unit}</td><td>{value}</td></tr>")
                else:
                    # Assume that if not string, it must be numerical
                    self.html_fragments.append(f"<tr><td>{name}</td><td>{unit}</td><td>{format_number(value)}</td></tr>")
        self.html_fragments.append("</table>")

    def add_plot(self, fig, caption: str, dpi: int = 200) -> None:
        """Add a plot to the HTML content using PNG format for faster rendering."""
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        self.html_fragments.append(
            f'<figure><img src="data:image/png;base64,{encoded_image}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>'
        )
        plt.close(fig)

    def add_inputs_table(self, output_name: str) -> None:
        """Add information with model inputs (for each reservoir) to the HTML content."""
        input_data = self.inputs.inputs[output_name].data
        included_inputs = [name for name in INPUT_NAMES if self.input_config[name]['include']]
        if not included_inputs:
            return

        self.add_subsection("Inputs")
        self.html_fragments.append("<table>")
        self.html_fragments.append("<tr><th>Input Name</th><th>Unit</th><th>Value(s)</th></tr>")

        # Add general inputs
        for input_name in included_inputs:
            if input_name not in input_data:
                continue
            name = self.input_config[input_name]['name']
            unit = self.input_config[input_name]['unit_latex']
            value = input_data[input_name]
            if isinstance(value, (list, tuple)):
                value = ', '.join(map(str, value))
            self.html_fragments.append(f"<tr><td>{name}</td><td>{unit}</td><td>{value}</td></tr>")

        # Add biogenic factors
        if "biogenic_factors" in included_inputs:
            self.html_fragments.append("<tr><td colspan='3' style='text-align:center; font-weight:bold;'>Biogenic Factors</td></tr>")
            biogenic_factors = input_data['catchment']['biogenic_factors']
            if not isinstance(biogenic_factors, dict):
                try:
                    biogenic_factors = biogenic_factors.todict()
                except AttributeError:
                    log.error("Variable biogenic factors cannot be converted to a dictionary")
            for factor_name, factor_value in biogenic_factors.items():
                name = FACTOR_NAMES[factor_name]
                unit = "-"
                self.html_fragments.append(f"<tr><td>{name}</td><td>{unit}</td><td>{factor_value}</td></tr>")

        # Add catchment inputs
        if "catchment_inputs" in included_inputs:
            self.html_fragments.append("<tr><td colspan='3' style='text-align:center; font-weight:bold;'>Catchment Inputs</td></tr>")
            for input_name, input_value in input_data['catchment'].items():
                if input_name == "biogenic_factors":
                    continue
                conf_input = self.input_config['catchment_inputs']['var_dict'][input_name]
                name = conf_input['name']
                unit = conf_input['unit_latex']
                if isinstance(input_value, (list, tuple)):
                    value = ', '.join(map(str, input_value))
                else:
                    value = str(input_value)
                self.html_fragments.append(f"<tr><td>{name}</td><td>{unit}</td><td>{value}</td></tr>")

        # Add reservoir inputs
        if "reservoir_inputs" in included_inputs:
            self.html_fragments.append("<tr><td colspan='3' style='text-align:center; font-weight:bold;'>Reservoir Inputs</td></tr>")
            for input_name, input_value in input_data['reservoir'].items():
                conf_input = self.input_config['reservoir_inputs']['var_dict'][input_name]
                name = conf_input['name']
                unit = conf_input['unit_latex']
                if input_value is None:
                    value = "N/A"
                elif isinstance(input_value, (list, tuple)):
                    value = ', '.join(map(str, input_value))
                else:
                    value = str(input_value)
                self.html_fragments.append(f"<tr><td>{name}</td><td>{unit}</td><td>{value}</td></tr>")

        self.html_fragments.append("</table>")

    def add_outputs_table(self, output_name: str) -> None:
        """Add outputs table to the HTML content."""
        output_data = self.outputs[output_name]
        self.add_subsection("Outputs")
        self.add_table(output_data, self.output_config['outputs'], ("Output Name", "Unit", "Value"))

    def add_intern_vars_table(self, output_name: str) -> None:
        """Add internal variables table to the HTML content."""
        intern_data = self.intern_vars[output_name]
        self.add_subsection("Internal Variables")
        self.add_table(intern_data, self.intern_vars_config, ("Variable Name", "Unit", "Value"))

    def add_landcover_piecharts(self, output_name: str) -> None:
        """Add landcover piecharts to the HTML content."""
        input_data = self.inputs.inputs[output_name]
        landcovers_reservoir = parse_landcover_composition(
            input_data.data["reservoir"]["area_fractions"])
        landcovers_catchment = parse_landcover_composition(
            input_data.data["catchment"]["area_fractions"])
        fig, axes = plt.subplots(1, 2, figsize=(5, 3))
        landcover_pie(
            axes[0], landcovers_reservoir, landcover_names, landcover_colors, 
            title="Reservoir Landcover Composition")
        landcover_pie(
            axes[1], landcovers_catchment, landcover_names, landcover_colors, 
            title="Catchment Landcover Composition")
        fig.tight_layout()
        self.add_plot(fig, "Landcover Composition")
        plt.close(fig)

    def add_plots(self, output_name: str) -> None:
        """Generate and add emission plots to the HTML content."""
        fig, axes = plt.subplots(2, 2, figsize=(7, 5))
        axes = axes.flatten()
        emission_var = {
            "co2": "co2_profile",
            "ch4": "ch4_profile",
            "n2o": "n2o_profile"}
                
        for ix, gas in enumerate(emission_var.keys()):
            if gas not in self.inputs.inputs[output_name].data["gasses"]:
                continue
            data = self.outputs[output_name]
            years = self.inputs.inputs[output_name].data["year_vector"]
            emission_unit = self.output_config['outputs'][
                emission_var[gas]]['unit_latex']
            data = self.outputs[output_name]
            title = ", ".join((self.output_config['outputs'][
                emission_var[gas]]['name_latex'], output_name))
            # Create title and y_label
            y_label = "Emission, " + emission_unit
            emissions = data.get(f"{gas}_profile", [])
            title = ", ".join((self.output_config['outputs'][
                emission_var[gas]]['name_latex'], output_name))
            emission_profile(
                axes[ix],
                years=years, 
                emissions=emissions, 
                title=title,
                y_label = y_label,
                annotate=False)
        self.plot_emission_bars(axes[ix+1], output_name)
        fig.tight_layout()
        # Clear unused axes if the number of gases is less than the number of axes
        for ax in axes[len(emission_var.keys())+1:]:
            fig.delaxes(ax)
        self.add_plot(fig, "Emission Profiles and Totals", dpi=175)
        fig.clear()  # Clear all axes from the figure
        plt.close(fig)
        

    def plot_emission_bars(self, axis: plt.Axes, output_name: str) -> None:
        """Plot total emissions as a bar chart."""
        data = self.outputs[output_name]
        vars_to_plot = ('co2_net', 'ch4_net', 'n2o_mean')
        bars = [self.output_config['outputs'][var]['gas_name_latex'] for
                var in vars_to_plot if var in data]
        # Get reservoir area from inputs (convert from km2 to m2)
        area = self.inputs.inputs[output_name].data['reservoir']['area'] * \
            10**6
        values = [data[var] * area * 10 ** (-6) for var in vars_to_plot
                  if var in data]
        emission_bars(
            axis=axis, gases=bars, values=values,
            output_name=output_name)

    def write(self) -> None:
        """Write the HTML content to the output file."""
        if not bool(self.outputs):
            log.error("Attempting to write before generating outputs.")
            return None
        self.html_fragments.append(
            '<button class="toggle-theme" onclick="toggleTheme()">Toggle Dark Mode</button>'
        )
        self.html_fragments.append('<div id="reservoir-links">')
        for _ix, reservoir_name in enumerate(self.outputs):
            self.html_fragments.append(
                f'<a href="#res-{reservoir_name}" title="Jump to {reservoir_name}">{reservoir_name}</a>'
            )
        self.html_fragments.append('</div>')  # Close #reservoir-links container
        
        for _ix, reservoir_name in enumerate(self.outputs):
            reservoir_type = self.inputs.inputs[reservoir_name].data["type"]
            log.info(f"Adding data for reservoir: {reservoir_name}, Progress: {_ix+1} out of {len(self.outputs)}")
            self.html_fragments.append(f'<section id="res-{reservoir_name}">')
            self.html_fragments.append(f"<details open><summary>{reservoir_name}</summary>")
            self.add_section(f"{reservoir_name} {reservoir_type} reservoir")
            self.add_inputs_table(reservoir_name)
            self.add_landcover_piecharts(reservoir_name)
            self.add_outputs_table(reservoir_name)
            self.add_plots(reservoir_name)
            self.add_intern_vars_table(reservoir_name)
            self.html_fragments.append(f'<button class="export-btn" onclick="exportSection(\'res-{reservoir_name}\', \'{reservoir_name}\')">Export</button>')
            self.html_fragments.append("</details></section>")
        self.html_fragments.append("</body></html>")
        # Join all fragments into a single string
        html_content = "".join(self.html_fragments)

        with open(self.output_file_path, "w", encoding="utf-8") as file:
            file.write(html_content)

        log.info(f"Created an HTML file at {self.output_file_path}.")


@dataclass
class Presenter:
    """Reads and processes results of GHG emission calculations and outputs
    them in different formats.

    Attributes:
        inputs (Inputs): An instance of Inputs class containing input data.
        outputs (Dict): Dictionary containing model outputs.
        intern_vars (Dict): Dictionary of internal variables.
        writers (Optional[List[Writer]]): Optional list of Writer objects for output.
        author (str): Name of the author. Default is 'Anonymous'.
        title (str): Title of the document. Default is 'Results'.
    """

    inputs: Inputs
    outputs: Dict
    intern_vars: Dict
    writers: Optional[List[Writer]] = field(default=None)
    author: str = field(default='Anonymous')
    title: str = field(default='Results')

    @classmethod
    def fromfiles(cls, input_file: str, output_file: str, interns_file: str, **kwargs):
        """Creates an instance of Presenter from JSON files.

        Args:
            input_file (str): Path to JSON file containing input data.
            output_file (str): Path to JSON file containing model outputs.
            interns_file (str): Path to JSON file containioutput_file_pathng internal variables.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            Presenter: An instance of Presenter initialized with data loaded from files.
        """
        inputs = Inputs.fromfile(input_file)
        with open(output_file, 'r', encoding='utf-8') as json_file:
            output_dict = json.load(json_file)
        with open(interns_file, 'r', encoding='utf-8') as json_file:
            intern_dict = json.load(json_file)
        return cls(
            inputs=inputs, outputs=output_dict, intern_vars=intern_dict, **kwargs)

    def __post_init__(self):
        """Loads configuration files after instance initialization."""
        self.input_config = registry.presenter_config.get("report_inputs")
        self.output_config = registry.presenter_config.get("report_outputs")
        self.intern_vars_config = registry.presenter_config.get("report_internal")
        self.parameter_config = registry.presenter_config.get("report_parameters")
        self.config_ini = registry.main_config.get("model_config")

    def add_writer(self, writer: Type[Writer], output_file: str) -> None:
        """Adds a Writer object to the list of writers.

        Args:
            writer (Type[Writer]): Type of Writer to instantiate.
            output_file (str): Path to the output file for the Writer.

        Returns:
            None
        """
        if self.writers is None:
            self.writers = []
        self.writers.append(
            writer(
                output_file_path=output_file,
                outputs=self.outputs,
                intern_vars=self.intern_vars,
                output_config=self.output_config,
                input_config=self.input_config,
                intern_vars_config=self.intern_vars_config,
                parameter_config=self.parameter_config,
                config_ini=self.config_ini,
                inputs=self.inputs,
                author=self.author,
                title=self.title,
            )
        )

    def output(self) -> None:
        """Outputs GHG emission calculation results using writers."""
        if self.writers is None:
            log.info("No writers specified. Results could not be output.")
            return None
        for writer in self.writers:
            writer.write()
        return None


if __name__ == "__main__":
    pass
