""" Class containg input data for calculating GHG emissions """
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import logging
from .biogenic import BiogenicFactors

# Set up module logger
log = logging.getLogger(__name__)
# Load path to Yaml tables
module_dir = os.path.dirname(__file__)


@dataclass
class Input:
    """ Input data for a single reservoir emission calculations """
    name: str
    input_data: Dict

    @property
    def reservoir_data(self) -> Dict:
        """ Retrieve input data for reservoir-scale process calculations """
        return self.input_data.get('reservoir')

    @property
    def catchment_data(self) -> Dict:
        """ Retrieve input data for catchment-scale process calculations """
        catchment_dict = self.input_data.get('catchment')
        catchment_dict["biogenic_factors"] = \
            BiogenicFactors.fromdict(catchment_dict["biogenic_factors"])
        return catchment_dict

    @property
    def emission_factors(self) -> List[str]:
        """ Retrieve a list of emission factors to be calculated """
        return self.input_data.get('emission_factors')

    @property
    def year_vector(self) -> Tuple[float]:
        """ Retrieve a tuple of years for which emissions profiles are
            being calculated """
        return tuple(self.input_data.get('year_vector'))

    @property
    def monthly_temps(self) -> List[float]:
        """ Retrieve a vecor of monthly average temperatures """
        return self.input_data.get('monthly_temps')

    @classmethod
    def fromfile(cls, file: str, reservoir_name: str):
        """ Load inputs dictionary from json file """
        with open(file) as json_file:
            output_dict = json.load(json_file)
            input_data = output_dict.get(reservoir_name)
            if input_data is None:
                log.error("Reservoir '%s' not found. Returning empty class",
                          reservoir_name)
        return cls(name=reservoir_name, input_data=input_data)


@dataclass
class Inputs:
    """ Collection of inputs for which GHG emissions are being calculated """
    inputs: List[Input]

    def add_input(self, input_dict: Dict) -> None:
        """ Add new input dictionary into dictionary of inputs """
        reservoir_name = list(input_dict.keys())[0]
        input_data = input_dict.get(reservoir_name)
        new_input = Input(name=reservoir_name, input_data=input_data)
        self.inputs.append(new_input)

    @classmethod
    def fromfile(cls, file: str):
        """ Load inputs dictionary from json file """
        inputs = []
        with open(file) as json_file:
            output_dict = json.load(json_file)
            for reservoir_name, input_data in output_dict.items():
                new_input = Input(name=reservoir_name, input_data=input_data)
                inputs.append(new_input)
        return cls(inputs=inputs)
