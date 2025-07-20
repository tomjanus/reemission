"""
This module contains classes and methods for managing input data used in the calculation of greenhouse gas (GHG) emissions 
from reservoirs. The module supports the loading and processing of input data from JSON files, providing a structured way 
to handle emission data for different reservoirs.

Classes:
    * **Input**: Represents input data for emission calculations for a single reservoir.
    * **Inputs**: Manages a collection of Input instances, representing multiple reservoirs.

The module supports:
    - Loading emission data from JSON files.
    - Providing default values for optional fields.
    - Retrieving specific data subsets (e.g., reservoir data, catchment data, emission gases, year vectors).
    - Adding new input data to the collection.

**Typical usage example:**

.. code-block:: Python
    
    input_file = 'path/to/input.json'
    reservoir_name = 'example_reservoir'
    
    # Load input data for a specific reservoir from a JSON file
    input_data = Input.fromfile(file=input_file, reservoir_name=reservoir_name)
    
    # Access specific data
    reservoir_data = input_data.reservoir_data
    catchment_data = input_data.catchment_data
    gasses = input_data.gasses
    year_vector = input_data.year_vector
    monthly_temps = input_data.monthly_temps
    
    # Manage multiple inputs
    inputs_collection = Inputs.fromfile(file=input_file)
    new_input_dict = {'new_reservoir': {'data_key': 'data_value'}}
    inputs_collection.add_input(new_input_dict)
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypeVar, Type, Optional, Literal, Any
import json
import logging
from reemission.biogenic import BiogenicFactors

# Set up module logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

InputType = TypeVar('InputType', bound='Input')
InputsType = TypeVar('InputsType', bound='Inputs')
EnumLoadMethods = Literal["name", "value"]


@dataclass
class Input:
    """Input data wrapper for emission calculations in a single reservoir.

    Attributes:
        name (str): Reservoir name.
        data (Dict): Emission data dictionary.
        enum_load_method (EnumLoadMethods): Method to load enum values, default is "value".
    """
    name: str
    data: Dict
    enum_load_method: EnumLoadMethods = "value"

    def __post_init__(self) -> None:
        """Instantiate optional input fields with default values.

        E.g., reservoir type, as it is not required in the model, the input
        data may not include the 'type' field.
        """
        # TODO: Move defaults to a configuration file
        defaults: Dict[str, Any] = {
            'type': 'unknown',
            'coordinates': [0.0, 0.0],
            'gasses': ["co2", "ch4", "n2o"],
            'year_vector': [1, 5, 10, 20, 30, 40, 50, 65, 80, 100]
        }
        for data_key, data_value in defaults.items():
            if data_key not in self.data.keys():
                self.data[data_key] = data_value

    @property
    def reservoir_data(self) -> Optional[Dict]:
        """Retrieve input data for reservoir-scale process calculations.

        Returns:
            Optional[Dict]: Data dictionary for reservoir calculations.
        """
        return self.data.get('reservoir')

    @property
    def catchment_data(self) -> Optional[Dict]:
        """Retrieve input data for catchment-scale process calculations.

        Returns:
            Optional[Dict]: Data dictionary for catchment calculations.
        """
        if self.data:
            catchment_dict = self.data['catchment'].copy()
            catchment_dict["biogenic_factors"] = BiogenicFactors.fromdict(
                catchment_dict["biogenic_factors"],
                method=self.enum_load_method)
            return catchment_dict
        return None

    @property
    def gasses(self) -> Optional[List[str]]:
        """Retrieve a list of emission factors/gases to be calculated.

        Returns:
            Optional[List[str]]: List of gases.
        """
        return self.data.get('gasses')

    @property
    def year_vector(self) -> Optional[Tuple[float, ...]]:
        """Retrieve a tuple of years for which emissions profiles are being calculated.

        Returns:
            Optional[Tuple[float, ...]]: Tuple of years.
        """
        if self.data:
            return tuple(float(item) for item in self.data['year_vector'])
        return None

    @property
    def monthly_temps(self) -> Optional[List[float]]:
        """Retrieve a vector of monthly average temperatures.

        Returns:
            Optional[List[float]]: List of monthly temperatures.
        """
        return self.data.get('monthly_temps')

    @classmethod
    def fromfile(cls: Type[InputType], file: str,
                 reservoir_name: str) -> InputType:
        """Load inputs dictionary from file.

        Args:
            cls (Type[InputType]): The class type.
            file (str): Path to JSON file.
            reservoir_name (str): Reservoir name.

        Returns:
            InputType: An instance of the Input class.
        """
        with open(file, 'r', encoding='utf-8') as json_file:
            output_dict = json.load(json_file)
            data = output_dict.get(reservoir_name, None)
            if data is None:
                log.error("Reservoir '%s' not found. Returning empty class",
                          reservoir_name)
                return cls(name=reservoir_name, data={})
        return cls(name=reservoir_name, data=data)


@dataclass
class Inputs:
    """Collection of inputs for which GHG emissions are being calculated.

    Attributes:
        inputs (Dict[str, Input]): Dictionary with input data for multiple reservoirs.
    """

    inputs: Dict[str, Input]

    def add_input(self, input_dict: Dict[str, dict]) -> None:
        """Add new input to self.inputs.

        Args:
            input_dict (Dict[str, dict]): Input dictionary with one or more reservoir names as keys 
                and data for each reservoir as values.
        """
        reservoir_name = list(input_dict.keys())[0]
        input_data = input_dict[reservoir_name]
        new_input = Input(name=reservoir_name, data=input_data)
        if reservoir_name not in self.inputs:
            self.inputs[reservoir_name] = new_input
        else:
            log.info("Key %s already in the inputs. Skipping", reservoir_name)
            
    def get_input(self, name: str) -> Input | None:
        return self.inputs.get(name)

    @classmethod
    def fromfile(cls: Type[InputsType], file: str) -> InputsType:
        """Load inputs dictionary from JSON file.

        Args:
            cls (Type[InputsType]): The class type.
            file (str): Path to the input JSON file.

        Returns:
            InputsType: An instance of the Inputs class.
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
