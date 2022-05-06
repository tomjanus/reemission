""" Module with simulation/calculation facilities for calculating GHG
    emissions for man-made reservoirs """
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, Union, Optional, List
import logging
from itertools import chain
import yaml
from reemission.input import Inputs
from reemission.temperature import MonthlyTemperature
from reemission.catchment import Catchment
from reemission.reservoir import Reservoir
from reemission.emissions import (CarbonDioxideEmission,
                        NitrousOxideEmission,
                        MethaneEmission)
from reemission.presenter import Presenter, Writer

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# TODO: potential problems in Python <= 3.6 because dicts are not ordered
#       pay attention
TRUTHS = [True, 'true', 'True', 'yes', 'on']
FALSES = [False, 'false', 'False', 'no', 'off', 'null']


@dataclass
class EmissionModel:
    """ Calculates emissions for a set of data provided in a dictionary
        format """
    inputs: Inputs
    outputs: Dict = field(init=False)
    config: Union[Dict, str]
    presenter: Optional[Type[Presenter]] = None

    def __post_init__(self):
        self.outputs = {}
        if isinstance(self.config, str):
            with open(self.config) as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)

    @staticmethod
    def create_exec_dictionary(co2_em: Union[CarbonDioxideEmission, None],
                               ch4_em: Union[MethaneEmission, None],
                               n2o_em: Union[NitrousOxideEmission, None],
                               years: Tuple[int] =
                               (1, 5, 10, 20, 30, 40, 50, 100)) -> Dict:
        """ Create a dictionary with function references and arguments for
            calculating gas emissions """
        co2_exec, ch4_exec, n2o_exec = {}, {}, {}
        if co2_em:
            co2_exec = {
                'co2_diffusion': {'ref': co2_em.gross_total,
                                  'args': {}},
                'co2_diffusion_nonanthro': {'ref': co2_em.flux_nonanthro,
                                            'args': {}},
                'co2_preimp': {'ref': co2_em.pre_impoundment,
                               'args': {}},
                'co2_minus_nonanthro': {'ref': co2_em.net_total,
                                        'args': {}},
                'co2_net': {'ref': co2_em.factor,
                            'args': {'number_of_years': years[-1]}},
                'co2_profile': {'ref': co2_em.profile,
                                'args': {'years': years}}}
        if ch4_em:
            ch4_exec = {
                'ch4_diffusion': {'ref': ch4_em.diffusion,
                                  'args': {'number_of_years': years[-1]}},
                'ch4_ebullition': {'ref': ch4_em.ebullition,
                                   'args': {}},
                'ch4_degassing': {'ref': ch4_em.degassing,
                                  'args': {'number_of_years': years[-1]}},
                'ch4_preimp': {'ref': ch4_em.pre_impoundment,
                               'args': {}},
                'ch4_net': {'ref': ch4_em.factor,
                            'args': {'number_of_years': years[-1]}},
                'ch4_profile': {'ref': ch4_em.profile,
                                'args': {'years': years}}}
        if n2o_em:
            n2o_exec = {
                'n2o_methodA': {'ref': n2o_em.factor,
                                'args': {'model': 'model 1'}},
                'n2o_methodB': {'ref': n2o_em.factor,
                                'args': {'model': 'model 2'}},
                'n2o_mean': {'ref': n2o_em.factor,
                             'args': {'mean': True}},
                'n2o_profile': {'ref': n2o_em.profile,
                                'args': {'years': years}}}
        exec_dict = dict(chain.from_iterable(
            d.items() for d in (co2_exec, ch4_exec, n2o_exec)))
        return exec_dict

    @property
    def get_inputs(self):
        """ Returns the Inputs object """
        return self.inputs

    def add_presenter(self, writers: List[Type[Writer]],
                      output_files: List[str], author="Anonymus",
                      title="HEET Results") -> None:
        """ Instantiates a presenter class to the emission model for data
            output formatting """
        self.presenter = Presenter(
            inputs=self.inputs, outputs=self.outputs, author=author,
            title=title)
        try:
            assert len(writers) == len(output_files)
        except AssertionError:
            log.error("Number of writers not equal to the number of files.")
            return None
        for writer, output_file in zip(writers, output_files):
            self.presenter.add_writer(writer=writer, output_file=output_file)
        return None

    def save_results(self) -> None:
        """ Save results using presenters defined in the presenters list """
        if not bool(self.outputs):
            log.error("Output dictionary empty. Run calculations first and " +
                      " try again.")
            return None
        self.presenter.output()
        return None

    def calculate(self, p_calc_method='g-res',
                  n2o_model='model 1') -> None:
        """ Calculate emissions for a number of variables defined in config """
        # Iterate through each set of inputs and output results in a dict
        for _, model_input in self.inputs.inputs.items():
            monthly_temp = MonthlyTemperature(model_input.data['monthly_temps'])
            # Instantiate Catchment and Reservoir objects
            catchment = Catchment(**model_input.catchment_data)
            reservoir = Reservoir(**model_input.data['reservoir'],
                                  inflow_rate=catchment.discharge)
            # Instantiate emission objects for every gas to be computed
            em_co2, em_n2o, em_ch4 = None, None, None

            # Calculate gas emissions
            if "co2" in model_input.data['gasses']:
                em_co2 = CarbonDioxideEmission(
                    catchment=catchment, reservoir=reservoir,
                    eff_temp=monthly_temp.eff_temp(),
                    p_calc_method=p_calc_method)
            if "ch4" in model_input.data['gasses']:
                em_ch4 = MethaneEmission(
                    catchment=catchment, reservoir=reservoir,
                    monthly_temp=monthly_temp, mean_ir=4.46)
            if "n2o" in model_input.data['gasses']:
                em_n2o = NitrousOxideEmission(
                    catchment=catchment, reservoir=reservoir, model=n2o_model)

            exec_dict = self.create_exec_dictionary(
                co2_em=em_co2, ch4_em=em_ch4, n2o_em=em_n2o,
                years=model_input.data['year_vector'])

            output = {}
            # Iterate through all emission components and record those that
            # are marked for outputting
            for emission, em_config in self.config['outputs'].items():
                if emission in exec_dict.keys() and \
                        em_config['include'] in TRUTHS:
                    output[emission] = exec_dict[emission]['ref'](
                        **exec_dict[emission]['args'])
            self.outputs[model_input.name] = output
