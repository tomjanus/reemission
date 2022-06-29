"""Simulation/calculation wrapper for GHG emission models."""
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, Union, Optional, List
import pathlib
import logging
from itertools import chain
import reemission.utils
from reemission.input import Inputs
from reemission.temperature import MonthlyTemperature
from reemission.catchment import Catchment
from reemission.reservoir import Reservoir
from reemission.emissions import (
    CarbonDioxideEmission, NitrousOxideEmission, MethaneEmission)
from reemission.presenter import Presenter, Writer

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#  TODO: potential problems in Python <= 3.6 because dicts are not ordered
#       pay attention
TRUTHS = [True, 'true', 'True', 'yes', 'on']
FALSES = [False, 'false', 'False', 'no', 'off', 'null']


@dataclass
class EmissionModel:
    """Calculates emissions for a reservoir or a numver of reservoirs.

    Atrributes:
        inputs: Inputs object with input data.
        outputs: outputs in a dictionary structure.
        config: dictionary with configuration data.
        presenter: Presenter object for presenting input and output data
            in various formats.
    """
    inputs: Inputs
    config: Union[Dict, pathlib.Path, str]
    outputs: Dict = field(init=False)
    author: str = field(default="")
    report_title: str = field(default="Results")
    p_model: str = field(default="g-res")
    presenter: Optional[Presenter] = field(default=None)

    def __post_init__(self) -> None:
        """Initialize outputs dict an load config file if config is a path."""
        self.outputs = {}
        if isinstance(self.config, (pathlib.Path, str)):
            self.config = reemission.utils.load_yaml(pathlib.Path(self.config))

    @staticmethod
    def create_exec_dictionary(
        co2_em: Union[CarbonDioxideEmission, None],
        ch4_em: Union[MethaneEmission, None],
        n2o_em: Union[NitrousOxideEmission, None],
        years: Tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50, 100),
    ) -> Dict:
        """
        Create a dictionary with function references and arguments for
        calculating gas emissions.

        Args:
            co2_em: emissions.CarbonDioxideEmission object.
            ch4_em: emissions.MethaneEmission object.
            n2o_em: emissions.NitrousOxideEmission object.
            years: vector of years for which the emission profiles are
                calculated.
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
                    'ref': n2o_em.factor, 'args': {'model': 'model 1'}},
                'n2o_methodB': {
                    'ref': n2o_em.factor, 'args': {'model': 'model 2'}},
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
        """Instantiates a presenter class to the emission model for data
        output formatting.

        Args:
            writers: List of presenter.Writer objects
            output_files: Paths to output files, one per writer.
        """
        self.presenter = Presenter(
            inputs=self.inputs, outputs=self.outputs,
            author=self.author, title=self.report_title)
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

    def calculate(self, n2o_model='model 1') -> None:
        """Calculate emissions for a number of variables defined in config.

        Args:
            n2o_model: total N2O emission calculation method.
                Options: 'model 1', 'model 2'
        """
        p_calc_method = self.p_model
        # Check the calculation options given in input arguments.
        avail_p_calc_methods = ('g-res', 'mcdowell')
        avail_n2o_models = ('model 1', 'model 2')
        if p_calc_method not in avail_p_calc_methods:
            log.warning(
                "Invalid P calculation method. Expected: %s. " +
                "Using default g-res method.",
                ', '.join(avail_p_calc_methods))
            p_calc_method = 'g-res'
        if n2o_model not in avail_n2o_models:
            log.warning(
                "Invalid total N2O emission model. Expected: %s. " +
                "Using default model 1.",
                ', '.join(avail_n2o_models))
            n2o_model = 'model 1'

        # Iterate through each set of inputs and output results in a dict
        for _, model_input in self.inputs.inputs.items():
            monthly_temp = MonthlyTemperature(
                model_input.data['monthly_temps'])
            # Instantiate Catchment and Reservoir objects
            catchment_data = model_input.catchment_data
            reservoir_data = model_input.reservoir_data
            if catchment_data is not None and reservoir_data is not None:
                catchment = Catchment.from_dict(
                    parameters=catchment_data)
                reservoir = Reservoir.from_dict(
                    parameters=reservoir_data,
                    temperature=monthly_temp,
                    coordinates=model_input.data["coordinates"],
                    inflow_rate=catchment.discharge)
            else:
                log.warning("Catchment or Reservoir data absent.")
                return None

            # Calculate gas emissions
            if "co2" in model_input.data['gasses']:
                em_co2 = CarbonDioxideEmission(
                    catchment=catchment,
                    reservoir=reservoir,
                    eff_temp=monthly_temp.eff_temp(gas='co2'),
                    p_calc_method=p_calc_method)
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
                    model=n2o_model)
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
            if isinstance(self.config, dict):
                for emission, em_config in self.config['outputs'].items():
                    if emission in exec_dict.keys() and \
                            em_config['include'] in TRUTHS:
                        output[emission] = exec_dict[emission]['ref'](
                            **exec_dict[emission]['args'])
                self.outputs[model_input.name] = output
            else:
                log.warning(
                    "Output/Input configuration file not instantiated.")
        return None
