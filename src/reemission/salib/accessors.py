""" Collection of classes and functions for accessing and modifying parameters and inputs in
ReEmission with SALib.
This module provides protocols and dataclasses for accessing model parameters and inputs,
required for allowing Re-Emission to interact with the `SALib` library for global sensitivity analysis.
It includes:
- `get_input_value`: Function to retrieve input values by group and name.
- `AccessProtocol`: Protocol for accessing model parameters.
- `ConfigParameterAccess`: Dataclass for accessing and modifying emission parameters in configurations.
- `InputAccess`: Dataclass for accessing and modifying inputs.
- `ReadOnlyParameterError`: Exception raised when attempting to modify a read-only parameter.
- `ReadOnlyInputError`: Exception raised when attempting to modify a read-only input.
"""
from typing import Literal, Any, Protocol, Tuple
from functools import reduce
import operator
from dataclasses import dataclass, field
from reemission.input import Input # type: ignore
from reemission.registry import config # type: ignore
from reemission.config_loader import ConfigLoader # type: ignore

USE_RICH = False  # Set to True to use rich for printing

if USE_RICH:
    from rich import print as rprint
else:
    def rprint(*args, **kwargs):
        """Fallback print function if rich is not used."""
        print(*args, **kwargs)

class ReadOnlyParameterError(Exception):
    """Raised when attempting to modify a read-only parameter."""


class ReadOnlyInputError(Exception):
    """Raised when attempting to modify a read-only input."""


def get_input_value(
        input_data: Input,
        group: Literal['reservoir', 'catchment'],
        input_name: str) -> Any:
    """Get inputs by group.
    Args:
        input_data: Input object containing the data.
        group: Group of the input, either 'reservoir' or 'catchment'.
        input_name: Name of the input variable to retrieve."""
    # We need to find the input value in order to calculate the bounds using relative differences
    return input_data.data[group][input_name]


class AccessProtocol(Protocol):
    """Protocol for accessing model parameters."""

    def get_value(self) -> Any | None:
        """Get the value of a parameter by its name."""
        return NotImplemented

    def set_value(self, value: Any) -> None:
        """Set the value of a parameter by its name."""
        return NotImplemented


@dataclass
class ConfigParameterAccess(AccessProtocol):
    """Protocol for accessing emission parameters.
    Attributes:
        config_name: Name of the configuration to access.
        key_path: Path to the parameter in the configuration.
        config: ConfigLoader instance to access configurations.
        read_only: If True, prevents modification of parameters.
        verbose: If True, enables verbose output.
    Raises:
        ValueError: If key_path is empty or not a tuple.
    """
    config_name: str
    key_path: str | Tuple[str, ...]
    config: ConfigLoader = field(default=config)
    read_only: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Initialize the ConfigParameterAccess."""
        if not self.key_path:
            raise ValueError("key_path must have at least one element")
        if isinstance(self.key_path, str):
            self.key_path = (self.key_path,)

    @property
    def param_name(self) -> str:
        """Get parameter name from key path."""
        return self.key_path[-1]

    @property
    def root_path(self) -> Tuple[str, ...]:
        """Get the root path of the input."""
        return self.key_path[:-1] if len(self.key_path) > 1 else tuple()

    def get_value(self) -> Any | None:
        """Get the value of an emission parameter by its name."""
        config_dict = self.config.get(name=self.config_name)
        if not config_dict:
            raise RuntimeError(f"Configuration '{self.config_name}' not found.")
        param_data = reduce(operator.getitem, self.key_path, config_dict)
        if not isinstance(param_data, dict):
            return param_data
        if self.verbose:
            rprint(f"Getting config parameter {self.param_name} from {self.config_name} at {self.key_path}")
        return param_data.get(self.param_name, None)

    def set_value(self, value: Any) -> None:
        """
        Set the value of an emission parameter by its name.

        Raises:
            ReadOnlyParameterError: If the parameter is read-only.
        """
        if self.read_only:
            raise ReadOnlyParameterError(f"Configuration '{self.config_name}' is read-only.")
        config_dict = self.config.get(name=self.config_name)
        if not config_dict:
            raise RuntimeError(f"Configuration '{self.config_name}' not found.")
        updated_param = {self.root_path: {self.param_name: value}}
        if self.verbose:
            rprint(f"Setting config parameter {self.param_name} to {value} in {self.config_name} at {self.key_path}")
        self.config.update(self.config_name, updated_param)


@dataclass 
class InputAccess(AccessProtocol):
    """Protocol for accessing inputs.
    Attributes:
        input: Input object used for model parameter access.
        key_path: Path to the input data.
        read_only: If True, prevents modification of parameters.
        verbose: If True, enables verbose output.
    Raises:
        ValueError: If key_path is empty or not a tuple.
    """
    input: Input
    key_path: str | Tuple[str, ...]
    read_only: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Initialize the InputAccess."""
        if not self.key_path:
            raise ValueError("key_path must have at least one element")
        if isinstance(self.key_path, str):
            self.key_path = (self.key_path,)

    @property
    def param_name(self) -> str:
        """Get parameter name from key path."""
        return self.key_path[-1]
    
    @property
    def root_path(self) -> Tuple[str, ...]:
        """Get the root path of the input."""
        return self.key_path[:-1] if len(self.key_path) > 1 else tuple()

    def get_value(self) -> Any | None:
        """Get the value of an input by its name."""
        if self.verbose:
            rprint(f"Getting input {self.param_name} from {self.key_path}")
        return reduce(operator.getitem, self.key_path, self.input.data)

    def set_value(self, value: Any) -> None:
        """Set the value of an input by its name."""
        if self.read_only:
            raise ReadOnlyInputError(f"Input '{self.key_path}' is read-only.")
        parent = reduce(
            operator.getitem,
            self.root_path,
            self.input.data) if self.root_path else self.input.data
        if self.verbose:
            rprint(f"Setting input {self.param_name} to {value} in {self.key_path}")
        parent[self.param_name] = value


if __name__ == "__main__":
    ...