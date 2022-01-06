""" Output result presentation layer for the GHG emissions computation engine
    """
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Type
from abc import ABC, abstractmethod
import configparser
import os
import logging
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pylatex.utils import bold
from pylatex import PageStyle, Head, simple_page_number, Figure, NoEscape, \
    Tabu, Center, Quantity, Description, Document, Section, Subsection, \
    Command, MultiColumn
from .utils import read_config
from .input import Inputs

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Format pyplot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

# Plot parameters
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
TICK_FONTSIZE = 10
ANNOTATION_FONTSIZE = 10

# Config paths
CONFIG_DIR = os.path.abspath(
    os.path.join(os.getcwd(), '..', 'config', 'emissions'))
INPUT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'inputs.yaml')
OUTPUT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'outputs.yaml')
PARAMETER_CONFIG_PATH = os.path.join(CONFIG_DIR, 'parameters.yaml')
CONFIG_INI_PATH = os.path.join(CONFIG_DIR, 'config.ini')


class Writer(ABC):
    """ Abstract base class for all writers """
    @abstractmethod
    def write(self):
        """ Writes outputs to the format of choice """
        ...


@dataclass
class JSONWriter(Writer):
    """ Format and write data to a JSON file """
    inputs: Inputs
    outputs: Dict
    output_file_path: str
    output_config: Dict
    input_config: Dict
    parameter_config: Dict
    config_ini: configparser.ConfigParser
    author: str
    title: str

    def write(self) -> None:
        """ Writes output data (all reservoir) to a JSON file """
        with open(self.output_file_path, 'w') as file_pointer:
            json.dump(self.outputs, file_pointer, indent=4)


@dataclass
class LatexWriter(Writer):
    """ Format and write data to a latex file using PyLaTeX"""
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
        """ Create document geometry """
        document_geometry = {
            "head": "0.0in",
            "margin": "0.75in",
            "top": "0.55in",
            "bottom": "0.55in",
            "includeheadfoot": True
        }
        return document_geometry

    def plot_profile(self, axis: plt.Axes, emission: str,
                     output_name: str, annotate: bool = True) -> None:
        """ Plot an emission profile using matplotlib """
        emission_var = {"co2": "co2_profile", "ch4": "ch4_profile",
                        "n2o": "n2o_profile"}
        data = self.outputs[output_name]
        # Create title and y_label
        title = ", ".join((
            self.output_config[
                'outputs'][emission_var[emission]]['name_latex'], output_name))
        emission_unit = self.output_config[
            'outputs'][emission_var[emission]]['unit_latex']
        y_label = "Emission, " + emission_unit
        # Get the x and y data
        y_data = data[emission_var[emission]]
        x_data = self.inputs.inputs[output_name].data["year_vector"]
        # Format plot
        axis.plot(x_data, y_data, '-', color='k', linewidth=1.0)
        axis.plot(x_data, y_data, marker='o', color='r')
        axis.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, labelpad=5)
        axis.set_xlabel('Time, years', fontsize=LABEL_FONTSIZE, labelpad=5)
        axis.set_title(title, fontsize=TITLE_FONTSIZE, pad=15)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        # Make the (visible) axes thicker
        for axis_pos in ['bottom', 'left']:
            axis.spines[axis_pos].set_linewidth(2)
        # Increase tick width
        axis.tick_params(width=2)
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
                    fontsize=ANNOTATION_FONTSIZE)

    def plot_emission_bars(self, axis: plt.Axes, output_name: str) -> None:
        """ Visualise total emissions (unit x surface area) for the
            calculated gases """
        data = self.outputs[output_name]
        vars_to_plot = ('co2_net', 'ch4_net', 'n2o_mean')
        bars = [self.output_config['outputs'][var]['gas_name_latex'] for
                var in vars_to_plot if var in data]
        # Get reservoir area from inputs (convert from km2 to m2)
        area = self.inputs.inputs[output_name].data[
            'reservoir']['area'] * 10**6
        values = [data[var] * area * 10**(-6) for var in vars_to_plot if
                  var in data]
        y_pos = np.arange(len(bars))[::-1]
        plt.barh(y_pos, values, color=(0.2, 0.4, 0.6, 0.6), edgecolor='blue')
        plt.yticks(y_pos, bars)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        axis.set_xlabel("Total annual emission, tonnes CO$_2$ yr$^{-1}$",
                        fontsize=LABEL_FONTSIZE)
        axis.set_ylabel("Gas", fontsize=LABEL_FONTSIZE)
        axis.set_title("Total annual gas emissions, {}".format(output_name),
                       fontsize=TITLE_FONTSIZE, pad=15)
        # Make the (visible) axes thicker
        for axis_pos in ['bottom', 'left']:
            axis.spines[axis_pos].set_linewidth(2)
        # Increase tick width
        axis.tick_params(width=2)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    def add_plots(self, output_name: str, plot_fraction: float = 0.85) -> None:
        """ Checks the number of plots to be produced and plots them in
            subfigures """
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
                        self.plot_profile(axis=axis, emission=gas,
                                          output_name=output_name,
                                          annotate=False)
                        fig_index += 1
            if bar_plot:
                axis = fig.add_subplot(*subplot_dim, fig_index)
                self.plot_emission_bars(axis=axis, output_name=output_name)

            fig.tight_layout(pad=1.0)
            with self.document.create(Figure(position='htbp')) as plot:
                width = r'{}\textwidth'.format(plot_fraction)
                dpi = 300
                plot.add_plot(width=NoEscape(width), dpi=dpi)
        return None

    def add_parameters(self) -> None:
        """ Adds information about model parameters such as conversion factors
            and other information that might be useful to report alongside
            calculation results """
        number_precision = 4
        round_options = {'round-precision': number_precision,
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
                            self.config_ini.getfloat(
                                'CARBON_DIOXIDE', 'co2_gwp100'))
                        desc.add_item(
                            NoEscape("GWP100 for CH$_4$: "),
                            self.config_ini.getfloat(
                                'METHANE', 'ch4_gwp100'))
                        desc.add_item(
                            NoEscape("GWP100 for N$_2$O: "),
                            self.config_ini.getfloat(
                                'NITROUS_OXIDE', 'nitrous_gwp100'))
                    except configparser.NoSectionError:
                        pass
        if self.parameter_config['parameters']['conv_factors']['include']:
            with self.document.create(
                    Subsection("Unit conversion factors")):
                try:
                    with self.document.create(Description()) as desc:
                        conv_coeff = self.config_ini.getfloat(
                            'CARBON_DIOXIDE', 'conv_coeff')
                        desc.add_item(
                            NoEscape('Conversion from mg~CO$_2$-C~m$^{-2}$~' +
                                     'd$^{-1}$ to g~CO$_{2,eq}$~m$^{-2}$~' +
                                     'yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options))
                    with self.document.create(Description()) as desc:
                        conv_coeff = self.config_ini.getfloat(
                            'METHANE', 'conv_coeff')
                        desc.add_item(
                            NoEscape('Conversion from mg CH$_4$~m$^{-2}$~' +
                                     'd$^{-1}$ to g~CO$_{2,eq}$~m$^{-2}$~' +
                                     'yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options))
                    with self.document.create(Description()) as desc:
                        conv_coeff = self.config_ini.getfloat(
                            'NITROUS_OXIDE', 'conv_coeff')
                        desc.add_item(
                            NoEscape('Conversion from $\\mu$g~N$_2$O~' +
                                     'm$^{-2}$~d$^{-1}$ to g~CO$_{2,eq}$~' +
                                     'm$^{-2}$~yr$^{-1}$: '),
                            Quantity(conv_coeff, options=round_options))
                except configparser.NoSectionError:
                    pass

    def add_outputs_table(self, output_name: str) -> None:
        """ Adds outputs table to latex source code """
        number_precision = 4
        round_options = {'round-precision': number_precision,
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
                    if self.output_config['outputs'][parameter]['include'] and \
                            not isinstance(value, list):
                        # Lists describe profiles and should not be put in the
                        # table
                        value = Quantity(value, options=round_options)
                        row = [parameter_name, unit, value]
                        data_table.add_row(row)
                # Add summed (composite) emission values
                if self.output_config['outputs']['co2_ch4']['include'] and \
                        self.output_config['outputs']['co2_net']['include'] and \
                        self.output_config['outputs']['ch4_net']['include']:
                    try:
                        value = Quantity(data['co2_net'] + data['ch4_net'],
                                         options=round_options)
                        unit = NoEscape(
                            self.output_config[
                                'outputs']['co2_ch4']['unit_latex'])
                        row = [NoEscape('CO$_2$+CH$_4$ net emissions'),
                               unit, value]
                        data_table.add_hline()
                        data_table.add_row(row)
                    except KeyError:
                        # Do not output anything if one of the data (either)
                        # CO2 net or CH4 net are not included in the results
                        pass

                if self.output_config['outputs']['co2_ch4_n2o']['include'] and \
                        self.output_config['outputs']['co2_net']['include'] and \
                        self.output_config['outputs']['ch4_net']['include'] and \
                        self.output_config['outputs']['n2o_mean']['include']:
                    try:
                        value = Quantity(data['co2_net'] + data['ch4_net'] +
                                         data['n2o_mean'],
                                         options=round_options)
                        unit = NoEscape(
                            self.output_config[
                                'outputs']['co2_ch4_n2o']['unit_latex'])
                        row = [NoEscape('CO$_2$+CH$_4$+N$_2$O net emissions'),
                               unit, value]
                        data_table.add_hline()
                        data_table.add_row(row)
                    except KeyError:
                        # Do not output anything if one of the data (either)
                        # CO2 net or CH4 net are not included in the results
                        pass

    def add_inputs_table(self, output_name: str) -> None:
        """ Add information with model inputs (for each reservoir) """
        number_precision = 4
        round_options = {'round-precision': number_precision,
                         'round-mode': 'figures'}
        column_names = ["Input Name", "Unit", "Value(s)"]
        table_format = "X[l] X[c] X[l]"
        input_names = ['monthly_temps', 'biogenic_factors', 'year_profile',
                       'catchment_inputs', 'reservoir_inputs', 'gasses']
        # Get inputs and input config
        input_data = self.inputs.inputs[output_name].data
        included_inputs = []
        for input_name in input_names:
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
                if "monthly_temps" in included_inputs:
                    name = self.input_config['monthly_temps']['name']
                    unit = NoEscape(
                        self.input_config['monthly_temps']['unit_latex'])
                    input_value = input_data["monthly_temps"]
                    input_value = ', '.join(
                        [str(item) for item in input_value])
                    row = [name, unit, input_value]
                    data_table.add_row(row)
                    printout = True
                if "year_profile" in included_inputs:
                    name = self.input_config['year_profile']['name']
                    unit = NoEscape(
                        self.input_config['year_profile']['unit_latex'])
                    input_value = input_data["year_vector"]
                    input_value = ', '.join(
                        [str(item) for item in input_value])
                    row = [name, unit, input_value]
                    data_table.add_row(row)
                    printout = True
                if "gasses" in included_inputs:
                    gas_name_latex = {'co2': 'CO$_2$', 'ch4': 'CH$_4$',
                                      'n2o': 'N$_2$O'}
                    name = self.input_config['gasses']['name']
                    unit = NoEscape(
                        self.input_config['gasses']['unit_latex'])
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
                    factor_names = {
                        "biome": "Biome",
                        "climate": "Climate",
                        "soil_type": "Soil Type",
                        "treatment_factor": "Treatment Factor",
                        "landuse_intensity": "Landuse Intensity"}
                    row_name = self.input_config['biogenic_factors']['name']
                    data_table.add_row(
                        (MultiColumn(3, align='c', data=row_name),))
                    data_table.add_hline()
                    for input_name, input_value in input_data[
                            'catchment']['biogenic_factors'].items():
                        name = factor_names[input_name]
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
                            input_value = Quantity(input_value,
                                                   options=round_options)
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
                        if not isinstance(input_value, Iterable):
                            input_value = Quantity(input_value,
                                                   options=round_options)
                        else:
                            input_value = ', '.join(
                                [str(item) for item in input_value])
                        row = [name, unit, input_value]
                        data_table.add_row(row)
        return None

    def add_header(self) -> None:
        """ Adds a header to latex document source code """
        header = PageStyle("header")
        # Create center header
        with header.create(Head("C")):
            header.append("HEET Results")
        # Create right header
        with header.create(Head("R")):
            header.append(simple_page_number())
        self.document.preamble.append(header)
        self.document.change_document_style("header")

    def add_title_section(self, title: str, author: str) -> None:
        """ Writes title to latex document source code """
        self.document.preamble.append(Command('title', title))
        self.document.preamble.append(Command('author', author))
        self.document.preamble.append(Command('date', NoEscape(r'\today')))
        self.document.append(NoEscape(r'\maketitle'))

    def add_parameters_section(self) -> None:
        """
            Writes model parameter information to latex document source code
        """
        with self.document.create(Section('Global parameters')):
            self.add_parameters()
        self.document.append(NoEscape(r'\pagebreak'))

    def add_inputs_subsection(self, reservoir_name: str) -> None:
        """ Writes inputs information to latex document source code """
        # Add inputs section to the document
        with self.document.create(Subsection('Inputs')):
            self.add_inputs_table(output_name=reservoir_name)

    def add_outputs_subsection(self, reservoir_name: str) -> None:
        """ Writes outputs information to latex document source code """
        plot_fraction = 0.9
        # Add inputs section to the document
        with self.document.create(Subsection('Outputs')):
            self.add_outputs_table(output_name=reservoir_name)
            self.add_plots(output_name=reservoir_name,
                           plot_fraction=plot_fraction)

    def write(self) -> None:
        """ Writes output data (all reservoir) to a text file """
        if not bool(self.outputs):
            return None
        self.add_header()
        self.add_title_section(title=self.title, author=self.author)
        self.add_parameters_section()
        # Iterate through all reservoirs in outputs and write to tex
        for reservoir_name in self.outputs:
            with self.document.create(Section(reservoir_name)):
                self.add_inputs_subsection(reservoir_name=reservoir_name)
                self.add_outputs_subsection(reservoir_name=reservoir_name)
        self.document.generate_pdf(clean_tex=False)
        self.document.generate_tex()
        return None


@dataclass
class Presenter:
    """ Reads and processes results of GHG emission calculations and outputs
        them in different formats """
    inputs: Inputs
    outputs: Dict
    writers: List[Writer] = field(default=None)
    author: str = 'Anonymous'
    title: str = 'HEET Results'

    @classmethod
    def fromfiles(cls, input_file: str, output_file: str, **kwargs):
        """ Load outputs dictionary from json file """
        inputs = Inputs.fromfile(input_file)
        with open(output_file) as json_file:
            output_dict = json.load(json_file)
        return cls(inputs=inputs, outputs=output_dict, **kwargs)

    def __post_init__(self):
        """ Load configuration dictionaries from provided yaml files """
        with open(INPUT_CONFIG_PATH) as file:
            self.input_config = yaml.load(file, Loader=yaml.FullLoader)
        with open(OUTPUT_CONFIG_PATH) as file:
            self.output_config = yaml.load(file, Loader=yaml.FullLoader)
        with open(PARAMETER_CONFIG_PATH) as file:
            self.parameter_config = yaml.load(file, Loader=yaml.FullLoader)
        self.config_ini = read_config(CONFIG_INI_PATH)

    def add_writer(self, writer: Type[Writer], output_file: str) -> None:
        """ Insantiates writer object and appends it to self.writers """
        if self.writers is None:
            self.writers = []
        self.writers.append(writer(
            output_file_path=output_file,
            outputs=self.outputs,
            output_config=self.output_config,
            input_config=self.input_config,
            parameter_config=self.parameter_config,
            config_ini=self.config_ini,
            inputs=self.inputs,
            author=self.author,
            title=self.title))

    def output(self) -> None:
        """ Present GHG emission calculation results using writers """
        if self.writers is None:
            log.info("No writers specified. Results could be output.")
            return None
        for writer in self.writers:
            writer.write()
        return None
