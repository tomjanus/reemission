""" Module with simulation/calculation facilities for calculating GHG
    emissions for man-made reservoirs """
from dataclasses import dataclass, field
from typing import Type, Dict, List, Tuple, Union, Optional
from itertools import chain
import yaml
from .input import Inputs
from .temperature import MonthlyTemperature
from .catchment import Catchment
from .reservoir import Reservoir
from .emissions import (CarbonDioxideEmission,
                        NitrousOxideEmission,
                        MethaneEmission)
from .presenter import Presenter

# TODO: potential problems in Python <= 3.6 because dicts are not ordered
#       pay attention
truths = [True, 'true', 'True', 'yes', 'on']
falses = [False, 'false', 'False', 'no', 'off', 'null']


@dataclass
class EmissionModel:
    """ Calculates emissions for a set of data provided in a dictionary
        format """
    config: Union[Dict, str]
    presenters: Optional[List[Type[Presenter]]] = None
    outputs: Dict = field(init=False)

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
                                'args': {'years': years}}
            }
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
                                'args': {'years': years}}
            }
        if n2o_em:
            n2o_exec = {
                'n2o_methodA': {'ref': n2o_em.factor,
                                'args': {'model': 'model 1'}},
                'n2o_methodB': {'ref': n2o_em.factor,
                                'args': {'model': 'model 2'}},
                'n2o_mean': {'ref': n2o_em.factor,
                             'args': {'mean': True}},
                'n2o_profile': {'ref': n2o_em.profile,
                                'args': {'years': years}},
                }
        exec_dict = dict(chain.from_iterable(
            d.items() for d in (co2_exec, ch4_exec, n2o_exec)))
        return exec_dict

    def add_presenter(self, presenter: Type[Presenter]) -> None:
        """ Add presenters to the emission model for data output formatting """
        try:
            self.presenters.append(presenter)
        except AttributeError:
            self.presenters = []
            self.presenters.append(presenter)

    def save_results(self):
        """ Save results using presenters defined in the presenters list """

    def calculate(self, inputs: Type[Inputs], p_calc_method='g-res',
                  n2o_model='model 1') -> None:
        """ Calculate emissions for a number of variables defined in config """
        # Iterate through each set of inputs and output results in a dict
        for model_input in inputs.inputs:
            monthly_temp = MonthlyTemperature(model_input.monthly_temps)
            # Instantiate Catchment and Reservoir objects
            catchment = Catchment(**model_input.catchment_data)
            reservoir = Reservoir(**model_input.reservoir_data,
                                  inflow_rate=catchment.discharge)
            # Instantiate emission objects for every gas to be computed
            em_co2, em_n2o, em_ch4 = None, None, None
            if "co2" in model_input.gasses:
                em_co2 = CarbonDioxideEmission(
                    catchment=catchment, reservoir=reservoir,
                    eff_temp=monthly_temp.eff_temp(),
                    p_calc_method=p_calc_method)
            if "ch4" in model_input.gasses:
                em_ch4 = MethaneEmission(
                    catchment=catchment, reservoir=reservoir,
                    monthly_temp=monthly_temp, mean_ir=4.46)
            if "n2o" in model_input.gasses:
                em_n2o = NitrousOxideEmission(
                    catchment=catchment, reservoir=reservoir, model=n2o_model)

            exec_dict = self.create_exec_dictionary(
                co2_em=em_co2, ch4_em=em_ch4, n2o_em=em_n2o,
                years=model_input.year_vector)

            output = {}
            for gas in model_input.gasses:
                for emission, em_config in self.config['outputs'][gas].items():
                    if em_config['include'] in truths:
                        output[emission] = exec_dict[emission]['ref'](
                            **exec_dict[emission]['args'])
            self.outputs[model_input.name] = output
