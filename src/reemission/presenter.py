"""Output result presentation layer for the GHG emissions computation engine.
"""
import collections.abc
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import (Dict, Tuple, List, Type, Union, Sequence, Iterator, Any,
                    Optional)
from abc import ABC, abstractmethod
import pathlib
import configparser
import os
import logging
import json
import matplotlib.pyplot as plt
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
import reemission.utils
from reemission.input import Inputs
from reemission.auxiliary import rollout_nested_list

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

# Get the file paths
MODULE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.abspath(os.path.join(MODULE_DIR, 'config'))
INPUT_CONFIG_PATH = pathlib.Path(CONFIG_DIR, 'inputs.yaml')
OUTPUT_CONFIG_PATH = pathlib.Path(CONFIG_DIR, 'outputs.yaml')
PARAMETER_CONFIG_PATH = pathlib.Path(CONFIG_DIR, 'parameters.yaml')
CONFIG_INI_PATH = pathlib.Path(CONFIG_DIR, 'config.ini')

# JSONWriter parameters
JSON_NUMBER_DECIMALS = 2
EXCEL_NUMBER_DECIMALS = 3

INPUT_NAMES = ['coordinates', 'monthly_temps', 'biogenic_factors',
               'year_profile', 'catchment_inputs', 'reservoir_inputs',
               'gasses']

FACTOR_NAMES = {
    "biome": "Biome",
    "climate": "Climate",
    "soil_type": "Soil Type",
    "treatment_factor": "Treatment Factor",
    "landuse_intensity": "Landuse Intensity",
}


@dataclass  # type: ignore[misc]
class Writer(ABC):
    """Abstract base class for all writers."""

    @staticmethod
    def round_parameter(
            number: Union[float, list],
            number_decimals: int) -> Union[float, list]:
        """
        Rounds number to the number of decimals provided in number_decimals.
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
        """Creates names of variables given as items in a variable that is
        a sequence, e.g. a list or a tuple.
        e.g. a variable var = [3,4,5] becomes: ('var_0', 3), ('var_1', 4),
            ('var_2', 5).
        """
        if isinstance(var_vector, Sequence):
            for iter_var, var in enumerate(var_vector):
                yield '_'.join([var_name, str(iter_var)]), var

    @staticmethod
    def write_par_to_dict(input_name: str, parameter: Any, par_dict: Dict,
                          reservoir_name: str, precision: int = 3) -> Dict:
        """Writes a parameter to a dictionary of par_name: par_value pairs.
        Args:
            input_name: Name of the input parameter.
            parameter: Parameter data. It can be a single value or a sequence.
            par_dict: Dictionary of parameter names and values that follows
                the structure: {reservoir_name: {par_name: par_values}}
            reservoir_name: Name of the reservoir for which tha parameter is to
                be added.
            precision: Number of decimal points for the parameter.
        """
        if isinstance(parameter, collections.abc.Sequence):
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
        """Writes outputs to the format of choice."""


@dataclass  # type: ignore[misc]
class ExcelWriter(Writer):
    """Formats and write data to an Excel file.

    Attributes:
        inputs: Inputs object with input data.
        outputs: Outputs dictionary.
        output_file_path: Path where the output file is to be saved/written to.
        output_config: Configuration file with formatting settings.
        input_config: Configuration file with formatting settings.
        parameter_config: Configuration file with formatting settings.
        config_ini: Global parameter configuration file.
        author: Author of the document.
        title: Title of the document.

        output_df: pandas.DataFrame with model outputs.
        input_df: pandas.DataFrame with model inputs.
    """

    inputs: Inputs
    outputs: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    parameter_config: Dict
    config_ini: configparser.ConfigParser
    author: str
    title: str
    output_df: pd.DataFrame = field(init=False)
    input_df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        """Initialize output dataframe which will be output as Excel file."""
        self.output_df = pd.DataFrame()
        self.input_df = pd.DataFrame()

    def add_inputs(self, reservoir_name: str) -> None:
        """Create an inputs dictionary for a given reservoir and append to
        a pandas DataFrame."""
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
                          "monthly_temps": "monthly_temps",
                          "year_profile": "year_vector",
                          "gasses": "gasses"}

        for input_name, input_name_mapped in config_to_data.items():
            if input_name in included_inputs:
                try:
                    parameter_value = input_data[input_name_mapped]
                    input_dict = self.write_par_to_dict(
                        input_name=input_name,
                        parameter=parameter_value,
                        par_dict=input_dict,
                        reservoir_name=reservoir_name,
                        precision=EXCEL_NUMBER_DECIMALS)
                except AttributeError:
                    log.error("Input name %s not found. Skipping",
                              input_name_mapped)

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
                        print(factor_name, factor_value)
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

    def add_outputs(self, reservoir_name: str) -> None:
        """Create an outputs dataframe for a given reservoir."""
        config = self.output_config['outputs']
        data = self.outputs[reservoir_name]
        output_dict: dict = {reservoir_name: {}}
        for parameter, parameter_value in data.items():
            # Add parameter if the parameter is marked for presentation
            if config[parameter]['include']:
                # If variabel is a sequence, rollout to multiple variables
                # stored in individual columns
                if isinstance(parameter_value, collections.abc.Sequence):
                    for name, value in self.rollout(
                            parameter, parameter_value):
                        param_dict = {name: self.round_parameter(
                            value, EXCEL_NUMBER_DECIMALS)}
                        output_dict[reservoir_name].update(param_dict)
                else:
                    param_dict = {parameter: self.round_parameter(
                        parameter_value, EXCEL_NUMBER_DECIMALS)}
                    output_dict[reservoir_name].update(param_dict)
        reservoir_df = pd.DataFrame.from_dict(output_dict, orient='index')
        reservoir_df.index.names = ['Name']
        if self.output_df.empty:
            self.output_df = reservoir_df
        else:
            self.output_df = pd.concat([self.output_df, reservoir_df])

    def write(self) -> None:
        """Write input/output data (all reservoirs) to an excel file."""
        if not bool(self.outputs):
            log.error("Attempting to write before generating outputs")
            return None
        for reservoir_name in self.outputs:
            self.add_inputs(reservoir_name=reservoir_name)
            self.add_outputs(reservoir_name=reservoir_name)
        with pd.ExcelWriter(self.output_file_path) as writer:
            self.input_df.to_excel(writer, sheet_name="inputs", index=True)
            self.output_df.to_excel(writer, sheet_name="outputs", index=True)
        log.info("Created a Excel file with outputs.")
        return None


@dataclass  # type: ignore[misc]
class JSONWriter(Writer):
    """Format and write data to a JSON file.

    Attributes:
        inputs: Inputs object with input data.
        outputs: Outputs dictionary.
        output_file_path: Path where the output file is to be saved/written to.
        output_config: Configuration file with formatting settings.
        input_config: Configuration file with formatting settings.
        parameter_config: Configuration file with formatting settings.
        config_ini: Global parameter configuration file.
        author: Author of the document.
        title: Title of the document.
        json_dict: Output dictionary save to the output JSON file.
    """

    inputs: Inputs
    outputs: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    parameter_config: Dict
    config_ini: configparser.ConfigParser
    author: str
    title: str
    json_dict: Dict = field(init=False)

    def __post_init__(self):
        """Initialie output_dict which will be output as JSON"""
        self.json_dict = {}

    def add_inputs(self, reservoir_name: str) -> None:
        """Create inputs dictionary for a given reservoir and append to
        json_dict."""
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

    def add_outputs(self, reservoir_name: str) -> None:
        """Create an outputs dictionary for a given reservoir and append to
        json_dict."""
        config = self.output_config['outputs']
        data = self.outputs[reservoir_name]
        output_dict: dict = {'outputs': {}}
        for parameter, parameter_value in data.items():
            # Add parameter if the parameter is marked for presentation
            if config[parameter]['include']:
                param_dict: dict = {parameter: {}}
                for item_name in ['name', 'gas_name', 'unit',
                                  'long_description']:
                    item = config[parameter].get(item_name)
                    param_dict[parameter][item_name] = item
                param_dict[parameter]['value'] = self.round_parameter(
                    parameter_value, JSON_NUMBER_DECIMALS)
                output_dict['outputs'].update(param_dict)
        try:
            self.json_dict[reservoir_name].update(output_dict)
        except KeyError:
            self.json_dict[reservoir_name] = output_dict

    def write(self) -> None:
        """Write output data (all reservoirs) to a JSON file."""
        if not bool(self.outputs):
            log.error("Attempting to write before generating outputs.")
            return None
        for reservoir_name in self.outputs:
            self.add_inputs(reservoir_name=reservoir_name)
            self.add_outputs(reservoir_name=reservoir_name)
        with open(self.output_file_path,
                  'w', encoding='utf-8') as file_pointer:
            json.dump(self.json_dict, file_pointer, indent=4)
        log.info("Created a json file with outputs.")
        return None


@dataclass  # type: ignore[misc]
class LatexWriter(Writer):
    """Format and write data to a latex file using PyLaTeX.

    Attrributes:
        inputs: Inputs object with input data.
        outputs: Outputs dictionary.
        output_file_path: Path where the output file is to be saved/written to.
        output_config: Configuration file with formatting settings.
        input_config: Configuration file with formatting settings.
        parameter_config: Configuration file with formatting settings.
        config_ini: Global parameter configuration file.
        author: Author of the document.
        title: Title of the document.

        document: pylatex Document object.
    """

    inputs: Inputs
    outputs: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    parameter_config: Dict
    config_ini: configparser.ConfigParser
    author: str
    title: str

    def __post_init__(self):
        path_no_ext = os.path.splitext(self.output_file_path)[0]
        self.document = Document(path_no_ext, geometry_options=self.geometry())

    @staticmethod
    def geometry() -> Dict:
        """Create document geometry structure."""
        document_geometry = {
            "head": "0.0in", "margin": "0.75in", "top": "0.55in",
            "bottom": "0.55in", "includeheadfoot": True}
        return document_geometry

    def plot_profile(self, axis: plt.Axes, emission: str, output_name: str,
                     annotate: bool = True) -> None:
        """Plot an emission profile using matplotlib.

        Args:
            axis: matplotlib axes object.
            emission: name of the gas/emision to plot.
            output_name: name of the output/reservoir.
            annotate: Flag setting whether emission values are added to plot.
        """
        plot_options = {'linewidth': 1.0, 'axiswidth': 2, 'tickwidth': 2,
                        'labelpad': 5, 'titlepad': 15,
                        'label_fontsize': LABEL_FONTSIZE,
                        'title_fontsize': TITLE_FONTSIZE,
                        'tick_fontsize': TICK_FONTSIZE,
                        'annotation_fontsize': ANNOTATION_FONTSIZE}

        emission_var = {"co2": "co2_profile",
                        "ch4": "ch4_profile",
                        "n2o": "n2o_profile"}
        data = self.outputs[output_name]
        # Create title and y_label
        title = ", ".join((self.output_config['outputs'][
            emission_var[emission]]['name_latex'], output_name))
        emission_unit = self.output_config['outputs'][
            emission_var[emission]]['unit_latex']
        y_label = "Emission, " + emission_unit

        # Get the x and y data
        try:
            y_data = data[emission_var[emission]]
            x_data = self.inputs.inputs[output_name].data["year_vector"]
        except KeyError:
            y_data = None
            x_data = None

        # Escape the function if data not found
        if x_data is None or y_data is None:
            return None

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
        plt.xticks(fontsize=plot_options['tick_fontsize'])
        plt.yticks(fontsize=plot_options['tick_fontsize'])
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

    def plot_emission_bars(self, axis: plt.Axes, output_name: str) -> None:
        """Visualise total emissions (unit x surface area) for the
        calculated gases.

        Args:
            axis: matplotlib axes object.
            output_name: name of the output/reservoir.
        """

        plot_options = {'linewidth': 1.0, 'axiswidth': 2, 'tickwidth': 2,
                        'labelpad': 5, 'titlepad': 15,
                        'label_fontsize': LABEL_FONTSIZE,
                        'title_fontsize': TITLE_FONTSIZE,
                        'tick_fontsize': TICK_FONTSIZE,
                        'annotation_fontsize': ANNOTATION_FONTSIZE}

        data = self.outputs[output_name]
        vars_to_plot = ('co2_net', 'ch4_net', 'n2o_mean')
        bars = [self.output_config['outputs'][var]['gas_name_latex'] for
                var in vars_to_plot if var in data]
        # Get reservoir area from inputs (convert from km2 to m2)
        area = self.inputs.inputs[output_name].data['reservoir']['area'] * \
            10**6
        values = [data[var] * area * 10 ** (-6) for var in vars_to_plot
                  if var in data]
        y_pos = np.arange(len(bars))[::-1]
        plt.barh(y_pos, values, color=(0.2, 0.4, 0.6, 0.6), edgecolor='blue')
        plt.yticks(y_pos, bars)
        plt.xticks(fontsize=plot_options['tick_fontsize'])
        plt.yticks(fontsize=plot_options['tick_fontsize'])
        axis.set_xlabel("Total annual emission, tonnes CO$_{2,eq}$ yr$^{-1}$",
                        fontsize=plot_options['label_fontsize'])
        axis.set_ylabel("Gas", fontsize=plot_options['label_fontsize'])
        axis.set_title("Total annual gas emissions, {}".format(output_name),
                       fontsize=plot_options['title_fontsize'],
                       pad=plot_options['titlepad'])
        # Make the (visible) axes thicker
        for axis_pos in ['bottom', 'left']:
            axis.spines[axis_pos].set_linewidth(plot_options['axiswidth'])
        # Increase tick width
        axis.tick_params(width=plot_options['tickwidth'])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    def add_plots(self, output_name: str, plot_fraction: float = 0.75,
                  dpi: int = 300) -> None:
        """Checks the number of plots to be produced and plots them in
        subfigures.

        Args:
            output_name: Name of the output / reservoir.
            plot_fraction: Fraction of text width the plot takes on the page.
            dpi: Resolution of the created figure within PDF.
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
                    if gas in self.inputs.inputs[output_name].data['gasses']:
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
        return None

    def add_parameters(self, precision: int = 4) -> None:
        """Adds information about model parameters such as conversion factors
        and other information that might be useful to report alongside
        calculation results.

        Args:
            precision: Number of decimal points in output parameters.
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
                            self.config_ini.getfloat('CARBON_DIOXIDE',
                                                     'co2_gwp100'))
                        desc.add_item(
                            NoEscape("GWP100 for CH$_4$: "),
                            self.config_ini.getfloat('METHANE',
                                                     'ch4_gwp100'))
                        desc.add_item(
                            NoEscape("GWP100 for N$_2$O: "),
                            self.config_ini.getfloat('NITROUS_OXIDE',
                                                     'nitrous_gwp100'))
                    except configparser.NoSectionError:
                        pass
        if self.parameter_config['parameters']['conv_factors']['include']:
            with self.document.create(Subsection("Unit conversion factors")):
                try:
                    with self.document.create(Description()) as desc:
                        conv_coeff = self.config_ini.getfloat(
                            'CARBON_DIOXIDE', 'conv_coeff')
                        desc.add_item(
                            NoEscape('Conversion from mg~CO$_2$-C~m$^{-2}$~' +
                                     'd$^{-1}$ to g~CO$_{2,eq}$~m$^{-2}$~' +
                                     'yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options),
                        )
                    with self.document.create(Description()) as desc:
                        conv_coeff = self.config_ini.getfloat(
                            'METHANE', 'conv_coeff')
                        desc.add_item(
                            NoEscape('Conversion from mg CH$_4$~m$^{-2}$~' +
                                     'd$^{-1}$ to g~CO$_{2,eq}$~m$^{-2}$~' +
                                     'yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options),
                        )
                    with self.document.create(Description()) as desc:
                        conv_coeff = self.config_ini.getfloat(
                            'NITROUS_OXIDE', 'conv_coeff')
                        desc.add_item(
                            NoEscape('Conversion from $\\mu$g~N$_2$O~' +
                                     'm$^{-2}$~d$^{-1}$ to g~CO$_{2,eq}$~' +
                                     'm$^{-2}$~yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options),
                        )
                except configparser.NoSectionError:
                    pass

    def add_outputs_table(self, output_name: str, precision: int = 4) -> None:
        """Adds outputs table to latex source code.

        Args:
            output_name: Name of the output/reservoir.
            precision: Number of decimal points in the output input values.
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
                if (
                    self.output_config['outputs']['co2_ch4']['include']
                    and self.output_config['outputs']['co2_net']['include']
                    and self.output_config['outputs']['ch4_net']['include']
                ):
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

                if (
                    self.output_config['outputs']['co2_ch4_n2o']['include']
                    and self.output_config['outputs']['co2_net']['include']
                    and self.output_config['outputs']['ch4_net']['include']
                    and self.output_config['outputs']['n2o_mean']['include']
                ):
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
            output_name: Name of the output/reservoir.
            precision: Number of decimal points in the output input values.
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

    def add_header(self, header_title: str = "Results") -> None:
        """Adds a header to latex document source code."""
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
        """Writes title to latex document source code."""
        self.document.preamble.append(Command('title', title))
        self.document.preamble.append(Command('author', author))
        self.document.preamble.append(Command('date', NoEscape(r'\today')))
        self.document.append(NoEscape(r'\maketitle'))

    def add_parameters_section(self) -> None:
        """
        Writes model parameter information to latex document source code.
        """
        with self.document.create(Section('Global parameters')):
            self.add_parameters()
        self.document.append(NoEscape(r'\pagebreak'))

    def add_inputs_subsection(self, reservoir_name: str) -> None:
        """Writes inputs information to latex document source code."""
        # Add inputs section to the document
        with self.document.create(Subsection('Inputs')):
            self.add_inputs_table(output_name=reservoir_name)

    def add_outputs_subsection(self, reservoir_name: str) -> None:
        """Writes outputs information to latex document source code."""
        plot_fraction = 0.75
        # Add inputs section to the document
        with self.document.create(Subsection('Outputs')):
            self.add_outputs_table(output_name=reservoir_name)
            self.add_plots(output_name=reservoir_name,
                           plot_fraction=plot_fraction)

    def write(self) -> None:
        """Writes output data (all reservoir) to a tex and pdf files."""
        if not bool(self.outputs):
            # If outputs are None or an empty dictionary.
            return None
        if reemission.utils.is_latex_installed():
            self.add_header()
            self.add_title_section(title=self.title, author=self.author)
            self.add_parameters_section()
            # Iterate through all reservoirs in outputs and write to tex
            for reservoir_name in self.outputs:
                with self.document.create(Section(reservoir_name)):
                    self.add_inputs_subsection(reservoir_name=reservoir_name)
                    self.add_outputs_subsection(reservoir_name=reservoir_name)
            # Generate a Tex Source file
            self.document.generate_tex()
            log.info("Created a LaTeX document with outputs.")
            # Generate a PDF (requires a LaTeX compiler present in the system)
            try:
                self.document.generate_pdf(clean_tex=False)
            except CompilerError:
                log.error(
                    "LaTeX compiler not found or returned an error. " +
                    "PDF could not be created.")
        else:
            log.error(
                "LaTeX cannot be found in your environment." +
                "'.tex' and '.pdf' files could not be created.")
        return None


@dataclass
class Presenter:
    """Reads and processes results of GHG emission calculations and outputs
    them in different formats.

    Atrributes:
        inputs: inputs.Inputs object.
        outputs: Model outputs dictionary.
        writers: a list of Writer objects.
        author: Author name.
        title: Document title.
    """

    inputs: Inputs
    outputs: Dict
    writers: Optional[List[Writer]] = field(default=None)
    author: str = field(default='Anonymous')
    title: str = field(default='HEET Results')

    @classmethod
    def fromfiles(cls, input_file: str, output_file: str, **kwargs):
        """Load outputs dictionary from json file."""
        inputs = Inputs.fromfile(input_file)
        with open(output_file, 'r', encoding='utf-8') as json_file:
            output_dict = json.load(json_file)
        return cls(inputs=inputs, outputs=output_dict, **kwargs)

    def __post_init__(self):
        """Load configuration dictionaries from provided yaml files."""
        self.input_config = reemission.utils.load_yaml(INPUT_CONFIG_PATH)
        self.output_config = reemission.utils.load_yaml(OUTPUT_CONFIG_PATH)
        self.parameter_config = reemission.utils.load_yaml(
            PARAMETER_CONFIG_PATH)
        self.config_ini = reemission.utils.read_config(CONFIG_INI_PATH)

    def add_writer(self, writer: Type[Writer], output_file: str) -> None:
        """Insantiates writer object and appends it to self.writers."""
        if self.writers is None:
            self.writers = []
        self.writers.append(
            writer(
                output_file_path=output_file,
                outputs=self.outputs,
                output_config=self.output_config,
                input_config=self.input_config,
                parameter_config=self.parameter_config,
                config_ini=self.config_ini,
                inputs=self.inputs,
                author=self.author,
                title=self.title,
            )
        )

    def output(self) -> None:
        """Present GHG emission calculation results using writers."""
        if self.writers is None:
            log.info("No writers specified. Results could not be output.")
            return None
        for writer in self.writers:
            writer.write()
        return None


if __name__ == "__main__":
    pass
