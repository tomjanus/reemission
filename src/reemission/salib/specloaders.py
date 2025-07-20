""" """
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, TypeVar, TypeAlias
from functools import cached_property
from types import MappingProxyType
from copy import deepcopy
import pathlib
from rich import print as rprint
from reemission.input import Input, Inputs # type: ignore
from reemission.salib.common import load_yaml
from reemission.salib.distributions import Variable, Distribution, DISTRIBUTION_CLASS_MAP
from reemission.salib.accessors import AccessProtocol, ConfigParameterAccess, InputAccess


Numerical = TypeVar('Numerical', float, int)
MissingInputHandler: TypeAlias = Callable[[Numerical], Distribution]
ParameterName: TypeAlias = str


def fix_distribution(
        distribution: Distribution,
        inplace: bool = False) -> Distribution:
    """Fix a distribution to a nominal value.
    Args:
        distribution: The distribution to fix.
    Returns:
        Distribution: A fixed distribution with bounds set to the nominal value.
    """
    return distribution.collapse(inplace=inplace)


def freeze(obj: Any) -> Any:
    """Recursively freeze a data structure to make it immutable."""
    if isinstance(obj, dict):
        return MappingProxyType({k: freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(freeze(i) for i in obj)
    if isinstance(obj, set):
        return frozenset(freeze(i) for i in obj)
    if isinstance(obj, tuple):
        return tuple(freeze(i) for i in obj)
    return obj  # primitive type or already immutable


def set_unit_input_distribution_using_rel_diffrence(
        value: Numerical,
        rel_difference: float) -> Distribution:
    """Create a distribution for a proportional unit input.
    Args:
        value: The base value for the distribution.
        rel_difference: The relative difference to apply to the value.
    Returns:
        Distribution: A distribution object with bounds set to the value ± relative difference.
    Raises:
        ValueError: If the value is not positive or if the relative difference is not in (0, 1).
    """
    if value <= 0:
        raise ValueError("Value must be positive for proportional unit distribution.")
    lower_bound = value * (1 - rel_difference)
    upper_bound = value * (1 + rel_difference)
    return DISTRIBUTION_CLASS_MAP['real-uniform'](
        bounds=(lower_bound, upper_bound)
    )


@dataclass(frozen=True)
class ReEmissionSALibSpecLoader:
    """
    Parses a YAML specification file for ReEmission sensitivity analysis with SALib.
    
    Attributes:
        spec_file: Path to YAML file.
        input: Input object used for model parameter access.
        data: Immutable parsed specification.
        missing_input_dist_handler: Function to handle missing input distributions.
        set_access_to_read_only: If True, prevents modification of parameters in accessors.
    Raises:
        FileNotFoundError: If the spec_file does not exist.
        ValueError: If the spec_file is not a YAML file or if required fields are missing
    """
    spec_file: str | pathlib.Path # yaml file with ReEmission config for SALib
    input: Input
    data: Dict[str, Any] = field(init=False)
    missing_input_dist_handler: MissingInputHandler | None = field(repr=False, default=None)
    set_access_to_read_only: bool = field(default=False)

    def __post_init__(self) -> None:
        # Check if file has yaml yml extension
        object.__setattr__(self, "spec_file", pathlib.Path(self.spec_file))
        if not self.spec_file.exists():
            raise FileNotFoundError(f"Spec file '{self.spec_file}' does not exist.")
        if self.spec_file.suffix.lower() not in ('.yaml', '.yml'):
            raise ValueError("spec_file must be a YAML file with .yaml or .yml extension")
        raw_data = load_yaml(self.spec_file)
        # Convert key_path lists to tuples
        processed_data = {
            param_name: {
                **config_data,
                'key_path': tuple(config_data['key_path'])
            }
            for param_name, config_data in raw_data.items()
        }
        # Deepcopy + freeze to ensure full immutability
        frozen_data = freeze(deepcopy(processed_data))
        object.__setattr__(self, "data", frozen_data)

    @cached_property
    def accessors(self) -> Dict[ParameterName, AccessProtocol]:
        """Create a map of accessors for the parameters defined in the config file.
        Returns:
            Dict[str, AccessProtocol]: A dictionary mapping parameter names to their accessors.
        Raises:
            ValueError: If 'key_path' or 'par_type' is missing in the variable definition.
        """
        accessors: Dict[ParameterName, AccessProtocol] = {}
        name_accessor_map = {
            'config': lambda config_data: ConfigParameterAccess(
                key_path=config_data['key_path'],
                config_name=config_data['config_name'],
                read_only=self.set_access_to_read_only),
            'input':  lambda config_data: InputAccess(
                input=self.input,
                key_path=config_data['key_path'],
                read_only=self.set_access_to_read_only),
        }
        for var_name, config_data in self.data.items():
            if config_data.get('include') is False:
                continue
            key_path = config_data.get('key_path')
            if key_path is None:
                raise ValueError(f"'key_path' is missing in variable '{var_name}'")
            par_type = config_data.get('par_type')
            if par_type is None:
                raise ValueError(f"'par_type' is missing in variable '{var_name}'")
            accessor_factory = name_accessor_map[config_data['par_type']]
            accessors[var_name] = accessor_factory(config_data)
        return accessors
    
    @cached_property
    def var_name_map(self) -> Dict[str, str]:
        """ A mapping between variable 'symbol' name and its full name for presentation and reporting """
        var_map: Dict[str, str] = {}
        for var_name, config_data in self.data.items():
            if config_data.get("name") is not None:
                var_map[var_name] = config_data['name']
        return var_map
                
    @cached_property
    def list_of_variables(self) -> List[Variable]:
        """Create a list of variables from the spec_file.
        Returns:
            List[Variable]: A list of Variable objects created from the spec_file.
        Raises:
            ValueError: If a variable does not have a distribution defined and is not an input type
        """
        _vars: List[Variable] = []
        for var_name, config_data in self.data.items():
            if config_data.get('include') is False:
                continue
            dist_dict = config_data.get('distribution')
            #dist_type = dist_dict.pop('type', None) - this now fails because dictionary is frozen.
            dist_type = dist_dict.get('type', None)

            # Get the nominal value from distribution and set a uniform distribution with
            if dist_type is None:
                if config_data.get('par_type') == 'input':
                    _accessor = self.accessors[var_name]
                    input_value = _accessor.get_value()
                    distribution = self.missing_input_dist_handler(input_value)
                    if config_data.get('fixed', False):
                        # If the input is fixed, use a uniform distribution with the nominal value
                        distribution = fix_distribution(distribution)
                else:
                    # If no distribution is defined, and the parameter type is not input, raise an error
                    raise ValueError(f"Parameter '{var_name}' does not have a distribution defined.")
            else:
                # If a distribution type is defined, create the distribution
                try:
                    dist_class = DISTRIBUTION_CLASS_MAP[dist_type]
                except KeyError as exc:
                    raise ValueError(f"Unknown distribution type: {dist_type} for parameter {var_name}") from exc
                dist_args = {k: v for k, v in dist_dict.items() if k != 'type'}
                try:
                    distribution = dist_class(**dist_args)
                except TypeError as exc:
                    raise ValueError(f"Error initializing distribution for '{var_name}' with type '{dist_type}'") from exc
                if config_data.get('fixed', False):
                    # If the parameter is fixed, use a uniform distribution with the nominal value
                    distribution = fix_distribution(distribution)
            _vars.append(
                Variable(
                    name=var_name,
                    distribution=distribution,
                    group=config_data.get('group'),
                    metadata=config_data.get('metadata', {})
                ))
        return _vars


@dataclass(frozen=True)
class TestModelSALibSpecLoader:
    """ Parses a YAML specification file for a test model with SALib.
    This loader is used to create a list of variables from the spec_file.
    Attributes:
        spec_file: Path to YAML file.
        data: Immutable parsed specification.
    Raises:
        FileNotFoundError: If the spec_file does not exist.
        ValueError: If the spec_file is not a YAML file or if required fields are missing
    """
    spec_file: str | pathlib.Path # yaml file with ReEmission config for SALib
    data: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        # Check if file has yaml yml extension
        object.__setattr__(self, "spec_file", pathlib.Path(self.spec_file))
        if not self.spec_file.exists():
            raise FileNotFoundError(f"Spec file '{self.spec_file}' does not exist.")
        if self.spec_file.suffix.lower() not in ('.yaml', '.yml'):
            raise ValueError("spec_file must be a YAML file with .yaml or .yml extension")
        raw_data = load_yaml(self.spec_file)
        # Deepcopy + freeze to ensure full immutability
        frozen_data = freeze(deepcopy(raw_data))
        object.__setattr__(self, "data", frozen_data)
        
    @cached_property
    def list_of_variables(self) -> List[Variable]:
        """Create a list of variables from the spec_file.
        Returns:
            List[Variable]: A list of Variable objects created from the spec_file.
        Raises:
            ValueError: If a variable dasoes not have a distribution defined and is not an input type
        """
        _vars: List[Variable] = []
        for var_name, config_data in self.data.items():
            if config_data.get('include') is False:
                continue
            dist_dict = config_data.get('distribution')
            #dist_type = dist_dict.pop('type', None) - this now fails because dictionary is frozen.
            dist_type = dist_dict.get('type', None)

            # Get the nominal value from distribution and set a uniform distribution with
            if dist_type is None:
                raise ValueError(f"Parameter '{var_name}' does not have a distribution defined.")
            # If a distribution type is defined, create the distribution
            try:
                dist_class = DISTRIBUTION_CLASS_MAP[dist_type]
            except KeyError as exc:
                raise ValueError(f"Unknown distribution type: {dist_type} for parameter {var_name}") from exc
            dist_args = {k: v for k, v in dist_dict.items() if k != 'type'}
            try:
                distribution = dist_class(**dist_args)
            except TypeError as exc:
                raise ValueError(f"Error initializing distribution for '{var_name}' with type '{dist_type}'") from exc
            if config_data.get('fixed', False):
                # If the parameter is fixed, use a uniform distribution with the nominal value
                distribution = fix_distribution(distribution)
            _vars.append(
                Variable(
                    name=var_name,
                    distribution=distribution,
                    group=config_data.get('group'),
                    metadata=config_data.get('metadata', {})
                ))
        return _vars


if __name__ == "__main__":
    from functools import partial
    # 1. Read the inputs and select an input for one selected reservoir
    source_file_directory = pathlib.Path(__file__).parent.resolve()
    uk_input_file = (source_file_directory / "../../data/uk_inputs.json").resolve()
    inputs = Inputs.fromfile(uk_input_file)
    selected_input = inputs.get_input('Katrine') # Use a single reservoir
    # 2. Define a function to set an input distribution dynamically, if it is missing in the spec file
    set_unit_input_distribution_10pct = partial(
        set_unit_input_distribution_using_rel_diffrence,
        rel_difference=0.1
    )
    # 3. Read the spec file
    spec_file = (source_file_directory / "../params_reemission_short.yaml").resolve()
    loader = ReEmissionSALibSpecLoader(
        spec_file=spec_file,
        input=selected_input,
        missing_input_dist_handler = set_unit_input_distribution_10pct
    )
    # Create accessors for each parameter
    accessors_map = loader.accessors
    for param_name, accessor in accessors_map.items():
        #accessor.set_value(666) # - should not be allowed externally but has to made available by SALib
        rprint(f"{param_name}: {accessor.get_value()}")
    # Create a list of variables from the spec file
    variables = loader.list_of_variables
    for variable in variables:
        rprint(f"Variable: {variable.name}, Distribution: {variable.distribution}")
    rprint("Input object in the loader...")
    rprint(loader.input)
    rprint("Accessing test model spec file....")
    loader_test = TestModelSALibSpecLoader(
        spec_file=(source_file_directory / "../params_test.yaml").resolve()
    )
    test_variables = loader_test.list_of_variables
    for variable in test_variables:
        rprint(f"Test Variable: {variable.name}, Distribution: {variable.distribution}")
