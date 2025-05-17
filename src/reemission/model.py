"""
Simulation/calculation wrapper for GHG emission models.

This module is designed to handle a variety of inputs, configure settings for different calculation 
methods, and produce detailed emission reports.

The main components of this module include:

1. **Configurations**:
    - The module reads configurations from a file to set default calculation methods and other options.
    - Configuration options are read from `config/config.ini` and include methods for retention coefficient 
      calculation, phosphorus export calculation, and nitrous oxide emission calculation.

2. **Inputs Handling**:
    - The `Inputs` class is used to manage and validate input data required for the simulations.
    - This includes catchment data, reservoir data, monthly temperature data, and other relevant parameters.

3. **Temperature Calculations**:
    - The `MonthlyTemperature` class is used to handle monthly temperature data which is crucial for estimating 
      gas emissions.
    - Effective temperature calculations are performed for different gases such as CO2, CH4, and N2O.

4. **Emissions Calculations**:
    - Separate classes for each gas (`CarbonDioxideEmission`, `MethaneEmission`, `NitrousOxideEmission`) handle 
      the specific calculations for each type of emission.
    - These classes utilize the inputs and configurations to compute emission profiles, total emissions, 
      diffusion fluxes, and other relevant metrics.

5. **Presenter and Writer**:
    - The `Presenter` class is responsible for formatting and presenting the results of the calculations.
    - Multiple writers can be added to the presenter to output data in different formats (e.g., CSV, JSON).

6. **Utility Functions**:
    - Utility functions such as `create_exec_dictionary` help in organizing and managing the execution of various 
      emission calculations.
    - Functions for adding presenters and saving results streamline the workflow for generating and exporting reports.

This module is designed to be flexible and extensible, allowing for easy integration of new calculation methods, 
input types, and output formats. It supports detailed logging and error handling to facilitate debugging and 
ensure robust performance.

Typical usage involves:
    1. Creating an `EmissionModel` object with appropriate inputs and configurations.
    2. Running the `calculate` method to perform the emissions calculations.
    3. Adding a presenter with desired output formats.
    4. Saving the results to specified files.

**Example:**

.. code-block:: Python
    
    from reemission import EmissionModel, Inputs, Writer

    # Initialize inputs
    inputs = Inputs(input_data)

    # Create emission model
    model = EmissionModel(inputs=inputs, config='path/to/config.yaml')

    # Calculate emissions
    model.calculate()

    # Add presenters for output
    model.add_presenter(writers=[Writer], output_files=['output.csv'])

    # Save results
    model.save_results()

Note:
    Ensure the configuration file and input data are properly formatted and validated before running the calculations.
"""
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, Union, Optional, List
import pathlib
import logging
from itertools import chain
from reemission.utils import load_yaml, read_config_dict
from reemission.globals import internal
from reemission.input import Inputs
from reemission.temperature import MonthlyTemperature
from reemission.catchment import Catchment
from reemission.reservoir import Reservoir
from reemission.emissions import (
    CarbonDioxideEmission, NitrousOxideEmission, MethaneEmission)
from reemission.presenter import Presenter, Writer
from reemission import registry

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#  TODO: potential problems in Python <= 3.6 because dicts are not ordered
#       pay attention
TRUTHS = [True, 'true', 'True', 'yes', 'on']
FALSES = [False, 'false', 'False', 'no', 'off', 'null']


@dataclass
class EmissionModel:
    """
    Calculates emissions for a reservoir or a number of reservoirs.

    Attributes:
        inputs (Inputs): Inputs object with input data.
        model_config (Union[Dict, pathlib.Path, str]): Dictionary with model configuration data or path to config file.
        presenter_config (Union[Dict, pathlib.Path, str]): Dictionary with Presenter configuration data or path to config file.
        outputs (Dict): Emission calculation outputs in a dictionary structure.
        internal (Dict): Internal variables for calculations.
        author (str): Author's name.
        report_title (str): Title of output report / GHG emission estimation study.
        ret_coeff (str): Reservoir retention coefficient calculation model.
        p_model (str): P export calculation method.
        n2o_model (str): Nitrous Oxide calculation method.
        presenter (Optional[Presenter]): Presenter object for presenting input and output data in various formats.
    """
    inputs: Inputs
    model_config: Union[Dict, pathlib.Path, str] = field(default=None)
    presenter_config: Union[Dict, pathlib.Path, str] = field(default=None)
    outputs: Dict = field(default_factory=dict)
    internal: Dict = field(default_factory=dict)
    author: str = field(default="")
    report_title: str = field(default="Results")
    ret_coeff: Optional[str] = field(default=None)
    p_model: Optional[str] = field(default=None)
    n2o_model: Optional[str] = field(default=None)
    presenter: Optional[Presenter] = field(default=None)

    def __post_init__(self) -> None:
        """Initialize outputs dict and load presenter_config file if presenter_config is a path."""
        internal = {}
        if self.presenter_config is None:
            self.presenter_config = registry.presenter_config.get("report_outputs")
        elif isinstance(self.presenter_config, (pathlib.Path, str)):
            self.presenter_config: dict = load_yaml(pathlib.Path(self.presenter_config))
        if self.model_config is None:
            self.model_config = registry.main_config.get("model_config")
        elif isinstance(self.model_config, (pathlib.Path, str)):
            self.model_config: dict = read_config_dict(self.model_config)
        
        # Option to explicitly set individual parameters in the class - perhaps of removing it and
        # relying directly on configurations
        if not self.ret_coeff:
            self.ret_coeff = self.model_config["CALCULATIONS"]['ret_coeff_method']
        if not self.p_model:
           self.p_model = self.model_config["CALCULATIONS"]['p_export_cal']
        if not self.n2o_model:
            self.n2o_model = self.model_config["CALCULATIONS"]['nitrous_oxide_model']
            
    def get_outputs(self) -> Dict:
        """Getter for outputs dictionary."""
        return self.outputs
        
    def get_inputs(self) -> Inputs:
        """Getter for the Inputs object"""
        return self.inputs
        
    def get_internal_vars(self) -> Dict:
        """Getter for internal (intermediate) variables"""
        return self.internal

    @staticmethod
    def create_exec_dictionary(
        co2_em: Union[CarbonDioxideEmission, None],
        ch4_em: Union[MethaneEmission, None],
        n2o_em: Union[NitrousOxideEmission, None],
        years: Tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50, 100),
    ) -> Dict:
        """
        Create a dictionary with function references and arguments for calculating gas emissions.

        Args:
            co2_em (Union[CarbonDioxideEmission, None]): CarbonDioxideEmission object.
            ch4_em (Union[MethaneEmission, None]): MethaneEmission object.
            n2o_em (Union[NitrousOxideEmission, None]): NitrousOxideEmission object.
            years (Tuple[int, ...]): Vector of years for which the emission profiles are calculated.

        Returns:
            Dict: Dictionary with function references and arguments for gas emissions calculations.
        """
        co2_exec, ch4_exec, n2o_exec = {}, {}, {}
        if co2_em:
            co2_exec = {
                'co2_diffusion': {
                    'ref': co2_em.diffusion_flux_int, 'args': {}},
                'co2_diffusion_nonanthro': {
                    'ref': co2_em.diffusion_flux_nonanthro, 'args': {}},
                'co2_preimp': {
                    'ref': co2_em.pre_impoundment, 'args': {}},
                'co2_minus_nonanthro': {
                    'ref': co2_em.net_total, 'args': {}},
                'co2_net': {
                    'ref': co2_em.factor,
                    'args': {'number_of_years': years[-1]}},
                'co2_total_per_year': {
                    'ref': co2_em.total_emission_per_year,
                    'args': {'number_of_years': years[-1]}},
                'co2_total_lifetime': {
                    'ref': co2_em.total_lifetime_emission,
                    'args': {'number_of_years': years[-1]}},
                'co2_profile': {
                    'ref': co2_em.profile, 'args': {'years': years}},
            }
        if ch4_em:
            ch4_exec = {
                'ch4_diffusion': {
                    'ref': ch4_em.diffusion_flux_int,
                    'args': {'time_horizon': years[-1]}},
                'ch4_ebullition': {
                    'ref': ch4_em.ebullition_flux_int,
                    'args': {'time_horizon': years[-1]}},
                'ch4_degassing': {
                    'ref': ch4_em.degassing_flux_int,
                    'args': {'time_horizon': years[-1]}},
                'ch4_preimp': {
                    'ref': ch4_em.pre_impoundment, 'args': {}},
                'ch4_net': {
                    'ref': ch4_em.factor,
                    'args': {'number_of_years': years[-1]}},
                'ch4_total_per_year': {
                    'ref': ch4_em.total_emission_per_year,
                    'args': {'number_of_years': years[-1]}},
                'ch4_total_lifetime': {
                    'ref': ch4_em.total_lifetime_emission,
                    'args': {'number_of_years': years[-1]}},
                'ch4_profile': {
                    'ref': ch4_em.profile, 'args': {'years': years}},
            }
        if n2o_em:
            n2o_exec = {
                'n2o_methodA': {
                    'ref': n2o_em.factor, 'args': {'model': 'model_1'}},
                'n2o_methodB': {
                    'ref': n2o_em.factor, 'args': {'model': 'model_2'}},
                'n2o_mean': {
                    'ref': n2o_em.factor, 'args': {'mean': True}},
                'n2o_total_per_year': {
                    'ref': n2o_em.total_emission_per_year,
                    'args': {'number_of_years': years[-1]}},
                'n2o_total_lifetime': {
                    'ref': n2o_em.total_lifetime_emission,
                    'args': {'number_of_years': years[-1]}},
                'n2o_profile': {
                    'ref': n2o_em.profile, 'args': {'years': years}},
            }
        exec_dict = dict(
            chain.from_iterable(
                d.items() for d in (co2_exec, ch4_exec, n2o_exec)))
        return exec_dict

    def add_presenter(
            self, writers: List[Type[Writer]], output_files: List[str]) \
            -> None:
        """
        Instantiate a presenter class to the emission model for data output formatting.

        Args:
            writers (List[Type[Writer]]): List of presenter.Writer objects.
            output_files (List[str]): Paths to output files, one per writer.
        """
        self.presenter = Presenter(
            inputs=self.inputs, 
            outputs=self.outputs, 
            intern_vars=self.internal,
            author=self.author, 
            title=self.report_title)
        try:
            assert len(writers) == len(output_files)
        except AssertionError:
            log.error("Number of writers not equal to the number of files.")
            return None
        for writer, output_file in zip(writers, output_files):
            self.presenter.add_writer(writer=writer, output_file=output_file)
        return None

    def save_results(self) -> None:
        """Save results using presenters defined in the presenters list."""
        if not bool(self.outputs):
            log.error(
                "Output dictionary empty. Run calculations first and " +
                " try again.")
            return None
        if self.presenter is not None:
            self.presenter.output()
        else:
            log.error("Presenter not defined.")
        return None

    def calculate(self) -> None:
        """Calculate emissions for a number of variables defined in config."""
        # Check the calculation options given in input arguments.
        avail_p_calc_methods = ('g-res', 'mcdowell')
        avail_n2o_models = ('model_1', 'model_2')
        if self.p_model not in avail_p_calc_methods:
            log.warning(
                "Invalid P calculation method. Expected: %s. " +
                "Using default g-res method.",
                ', '.join(avail_p_calc_methods))
            self.p_model = 'g-res'
        if self.n2o_model not in avail_n2o_models:
            log.warning(
                "Invalid total N2O emission model. Expected: %s. " +
                "Using default model 1.",
                ', '.join(avail_n2o_models))
            self.n2o_model = 'model_1'

        # Iterate through each set of inputs and output results in a dict
        for _, model_input in self.inputs.inputs.items():
            monthly_temp = MonthlyTemperature(
                model_input.data['monthly_temps'])
            # Instantiate Catchment and Reservoir objects
            catchment_data = model_input.catchment_data
            reservoir_data = model_input.reservoir_data
            if catchment_data is not None and reservoir_data is not None:
                catchment = Catchment.from_dict(
                    parameters=catchment_data,
                    name=model_input.name)
                reservoir = Reservoir.from_dict(
                    parameters=reservoir_data,
                    temperature=monthly_temp,
                    coordinates=model_input.data["coordinates"],
                    inflow_rate=catchment.discharge,
                    name=model_input.name)
            else:
                log.warning("Catchment or Reservoir data absent.")
                return None
            
            # Run calculations of internal variables that we might want to output
            # but which are not called during GHG estimation
            #1. Trophic status
            _ = reservoir.trophic_status(
                tp_inflow_conc=catchment.inflow_p_conc(method=self.p_model))

            # Calculate gas emissions
            if "co2" in model_input.data['gasses']:
                em_co2 = CarbonDioxideEmission(
                    catchment=catchment,
                    reservoir=reservoir,
                    eff_temp=monthly_temp.eff_temp(gas='co2'),
                    p_calc_method=self.p_model)
                
                #2. Reservoir TN concentration
                _ = em_co2.reservoir_tn

            else:
                em_co2 = None
            if "ch4" in model_input.data['gasses']:
                em_ch4 = MethaneEmission(
                    catchment=catchment,
                    reservoir=reservoir,
                    monthly_temp=monthly_temp)
            else:
                em_ch4 = None

            if "n2o" in model_input.data['gasses']:
                em_n2o = NitrousOxideEmission(
                    catchment=catchment,
                    reservoir=reservoir,
                    model=self.n2o_model,
                    p_export_model=self.p_model)
            else:
                em_n2o = None

            exec_dict = self.create_exec_dictionary(
                co2_em=em_co2,
                ch4_em=em_ch4,
                n2o_em=em_n2o,
                years=model_input.data['year_vector'])

            output = {}
            # Iterate through all emission components and record those that
            # are marked for outputting
            if isinstance(self.presenter_config, dict):
                for emission, em_config in self.presenter_config['outputs'].items():
                    if emission in exec_dict.keys() and \
                            em_config['include'] in TRUTHS:
                        output[emission] = exec_dict[emission]['ref'](
                            **exec_dict[emission]['args'])
                self.outputs[model_input.name] = output
                self.internal[model_input.name] = internal.copy()
            else:
                log.warning(
                    "Output/Input configuration file not instantiated.")
        return None
