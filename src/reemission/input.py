""" Class containg input data for calculating GHG emissions """
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypeVar, Type, Optional
import json
import logging
from reemission.biogenic import BiogenicFactors

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# Load path to Yaml tables
module_dir = os.path.dirname(__file__)

InputType = TypeVar('InputType', bound='Input')
InputsType = TypeVar('InputsType', bound='Inputs')


@dataclass
class Input:
    """Input data wrapper for emission calculations in a single reservoir.

    Arguments:
        name: reservoir name.
        data: emission data dictionary.
    """
    name: str
    data: Optional[Dict]

    @property
    def reservoir_data(self) -> Optional[Dict]:
        """Retrieve input data for reservoir-scale process calculations."""
        if self.data:
            return self.data['reservoir']
        return None

    @property
    def catchment_data(self) -> Optional[Dict]:
        """Retrieve input data for catchment-scale process calculations."""
        if self.data:
            catchment_dict = self.data['catchment'].copy()
            catchment_dict["biogenic_factors"] = BiogenicFactors.fromdict(
                catchment_dict["biogenic_factors"])
            return catchment_dict
        return None

    @property
    def gasses(self) -> Optional[List[str]]:
        """Retrieve a list of emission factors/gases to be calculated."""
        if self.data:
            return self.data['gasses']
        return None

    @property
    def year_vector(self) -> Optional[Tuple[float, ...]]:
        """Retrieve a tuple of years for which emissions profiles are
        being calculated."""
        if self.data:
            return tuple(float(item) for item in self.data['year_vector'])
        return None

    @property
    def monthly_temps(self) -> Optional[List[float]]:
        """Retrieve a vecor of monthly average temperatures."""
        if self.data:
            return self.data['monthly_temps']
        return None

    @classmethod
    def fromfile(cls: Type[InputType], file: str,
                 reservoir_name: str) -> InputType:
        """Load inputs dictionary from file.

        Args:
            file: path to JSON file.
            reservoir_name: Reservoir name.
        """
        with open(file, 'r', encoding='utf-8') as json_file:
            output_dict = json.load(json_file)
            data = output_dict.get(reservoir_name, None)
            if data is None:
                log.error("Reservoir '%s' not found. Returning empty class",
                          reservoir_name)
        return cls(name=reservoir_name, data=data)


@dataclass
class Inputs:
    """Collection of inputs for which GHG emissions are being calculated.

    Arguments:
        inputs: dictionary with input data for multiple reservoirs.
    """

    inputs: Dict[str, Input]

    def add_input(self, input_dict: Dict[str, dict]) -> None:
        """Add new input to self.inputs.

        Args:
            input_dict: input dictionary with one or more reservoir names as
                keys and data for each reservoir as values.
        """
        reservoir_name = list(input_dict.keys())[0]
        input_data = input_dict[reservoir_name]
        new_input = Input(name=reservoir_name, data=input_data)
        if reservoir_name not in self.inputs:
            self.inputs[reservoir_name] = new_input
        else:
            log.info("Key %s already in the inputs. Skipping", reservoir_name)

    @classmethod
    def fromfile(cls: Type[InputsType], file: str) -> InputsType:
        """Load inputs dictionary from json file.

        Args:
            file: path to the input JSON file.
        """
        inputs = {}
        with open(file, encoding='utf-8') as json_file:
            output_dict = json.load(json_file)
            for reservoir_name, input_data in output_dict.items():
                new_input = Input(name=reservoir_name, data=input_data)
                if reservoir_name not in inputs:
                    inputs[reservoir_name] = new_input
                else:
                    log.info("Key %s already in the inputs. Skipping",
                             reservoir_name)
        return cls(inputs=inputs)
