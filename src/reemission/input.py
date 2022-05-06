""" Class containg input data for calculating GHG emissions """
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import logging
from reemission.biogenic import BiogenicFactors

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# Load path to Yaml tables
module_dir = os.path.dirname(__file__)


@dataclass
class Input:
    """ Input data for a single reservoir emission calculations """
    name: str
    data: Dict

    @property
    def reservoir_data(self) -> Dict:
        """ Retrieve input data for reservoir-scale process calculations """
        return self.data.get('reservoir')

    @property
    def catchment_data(self) -> Dict:
        """ Retrieve input data for catchment-scale process calculations """
        catchment_dict = self.data.get('catchment')
        catchment_dict["biogenic_factors"] = \
            BiogenicFactors.fromdict(catchment_dict["biogenic_factors"])
        return catchment_dict

    @property
    def gasses(self) -> List[str]:
        """ Retrieve a list of emission factors to be calculated """
        return self.data.get('gasses')

    @property
    def year_vector(self) -> Tuple[float]:
        """ Retrieve a tuple of years for which emissions profiles are
            being calculated """
        return tuple(self.data.get('year_vector'))

    @property
    def monthly_temps(self) -> List[float]:
        """ Retrieve a vecor of monthly average temperatures """
        return self.data.get('monthly_temps')

    @classmethod
    def fromfile(cls, file: str, reservoir_name: str):
        """ Load inputs dictionary from json file """
        with open(file) as json_file:
            output_dict = json.load(json_file)
            data = output_dict.get(reservoir_name)
            if data is None:
                log.error("Reservoir '%s' not found. Returning empty class",
                          reservoir_name)
        return cls(name=reservoir_name, data=data)


@dataclass
class Inputs:
    """ Collection of inputs for which GHG emissions are being calculated """
    inputs: Dict[str, Input]

    def add_input(self, input_dict: Dict) -> None:
        """ Add new input dictionary into dictionary of inputs """
        reservoir_name = list(input_dict.keys())[0]
        input_data = input_dict.get(reservoir_name)
        new_input = Input(name=reservoir_name, data=input_data)
        if reservoir_name not in self.inputs:
            self.inputs[reservoir_name] = new_input
        else:
            log.info("Key %s already in the inputs. Skipping", reservoir_name)

    @classmethod
    def fromfile(cls, file: str):
        """ Load inputs dictionary from json file """
        inputs = {}
        with open(file) as json_file:
            output_dict = json.load(json_file)
            for reservoir_name, input_data in output_dict.items():
                new_input = Input(name=reservoir_name, data=input_data)
                if reservoir_name not in inputs:
                    inputs[reservoir_name] = new_input
                else:
                    log.info("Key %s already in the inputs. Skipping",
                             reservoir_name)
        return cls(inputs=inputs)
