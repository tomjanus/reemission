""" """

from __future__ import annotations
from typing import (
    TypeVar,
    Callable,
    Iterable,
    Sequence,
    Dict,
    Protocol,
    TypeAlias,
    List,
    Set,
    ClassVar,
)
from collections.abc import Iterable as IterableType
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from reemission.input import Input, Inputs  # type: ignore
from reemission.model import EmissionModel
from reemission.salib.distributions import Variable
from reemission.salib.accessors import AccessProtocol


NumpyValue = TypeVar("NumpyValue", np.floating, np.integer)
Numerical = TypeVar("Numerical", float, int)
SALibModelInputs: TypeAlias = NDArray[NumpyValue] | Iterable[Numerical]
SALibModelOutput: TypeAlias = NumpyValue | Numerical
MCModelInputs: TypeAlias = SALibModelInputs
MCModelOutput: TypeAlias = SALibModelOutput
SALibModel = Callable[[SALibModelInputs], SALibModelOutput]
ParameterName: TypeAlias = str

# Define a type variable for the parameter values
ModelInputs = TypeVar("ModelInputs", bound=Sequence[Variable])


class SALibModelWrapperProtocol(Protocol):
    """Protocol defining the interface for model execution in the SALib package
    for global sensitivity analysis"""

    def run(self, inputs: SALibModelInputs) -> SALibModelOutput:
        """Run the model with inputs provided as a NDArray or an Iterable"""
        raise NotImplementedError()

    def get_index(self, var_name: str, raise_error: bool = False) -> int | None:
        """Gets variable index by its name"""


class MCModelWrapperProtocol(Protocol):
    """ """

    def run(self, inputs: MCModelInputs) -> MCModelOutput:
        """ """
        raise NotImplementedError()

    def get_index(self, var_name: str, raise_error: bool = False) -> int | None:
        """Gets variable index by its name"""


@dataclass
class TestModelSALibWrapper:
    """A simple test model wrapper for SALib analysis.

    This model takes a numpy array of parameters and returns a numpy array of results.

    TODO: Check if SALib also runs models with multidimensional dataset. If so, modify the code.
    """

    ix_to_par: Dict[int, str]  # Relates index in the model vector to variable
    _par_to_ix: Dict[str, int] = field(init=False, repr=False)
    required_vars: ClassVar[List[str]] = [
        "a",
        "b",
        "c",
        "d0",
        "d1",
        "d2",
        "e",
        "cont1",
        "cont2",
        "cat1",
        "cat2",
    ]

    @classmethod
    def from_variables(cls, variables: List[Variable]) -> TestModelSALibWrapper:
        """ """
        variable_names = [var.name for var in variables]
        missing_vars: Set[str] = set(cls.required_vars) - set(variable_names)
        if missing_vars:
            raise ValueError(
                "Cannot instantiate from the Variables list. ",
                f"The following variables are missing: {', '.join(missing_vars)}",
            )
        # Assume the order of variables is the input to run_salib is the same
        # as the order of variables in the variables list
        ix_to_par = {}
        supported_vars = [var for var in variables if var.name in cls.required_vars]
        for ix, variable in enumerate(supported_vars):
            ix_to_par[ix] = variable.name
        return cls(ix_to_par=ix_to_par)

    def __post_init__(self):
        """Initialize the wrapper with a mapping from indices to parameter names."""
        if not self.ix_to_par:
            raise ValueError("Parameter index to name mapping cannot be empty.")
        if len(self.ix_to_par) != len(self.required_vars):
            raise ValueError(
                f"Expected {len(self.required_vars)} parameters, got {len(self.ix_to_par)}"
            )
        if not set(self.required_vars).issubset(self.ix_to_par.values()):
            raise ValueError(
                f"Expected parameters {self.required_vars}, got {set(self.ix_to_par.values())}"
            )

    @property
    def variable_order(self) -> List[str]:
        """ """
        return [self.ix_to_par[key] for key in sorted(self.ix_to_par)]

    def get_index(self, var_name: str, raise_error: bool = False) -> int | None:
        """Get the index of a variable by its name, with lazy reverse mapping."""
        if not hasattr(self, "_par_to_ix"):
            # Build the reverse mapping only once
            self._par_to_ix = {name: ix for ix, name in self.ix_to_par.items()}
        if var_name in self._par_to_ix:
            return self._par_to_ix[var_name]
        if raise_error:
            raise ValueError(
                f"Variable '{var_name}' not found in the model parameters."
            )
        return None

    @staticmethod
    def simple_test_model(par: Dict[str, Numerical]) -> SALibModelOutput:
        """Simple input output model for testing SA with Sobol Indices and SALib"""
        # par = tuple(par)
        # a,b,c,d0,d1,d2,e,cont1,cont2,cat1,cat2 = par
        cat1_idx = min(int(par["cat1"]), 1)
        cat2_idx = min(int(par["cat2"]), 2)
        gross = (
            par["a"] * par["cont1"] ** 2
            + par["b"] * np.sqrt(par["cont2"] + 1)
            + par["c"] * cat1_idx
        )
        if cat2_idx == 0:
            pre = par["d0"] + par["e"] * par["cont1"]
        elif cat2_idx == 1:
            pre = par["d1"] + par["e"] * par["cont1"]
        else:
            pre = par["d2"] + par["e"] * par["cont1"]
        return gross - pre

    def run(self, inputs: SALibModelInputs) -> SALibModelOutput:
        """Run the model with inputs provided as a NDArray or an Iterable.
        Note: inputs can be a single row of parameters or a 2D array with multiple rows.
        """
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2:
            if inputs.shape[1] != 11:
                raise ValueError(f"Expected 11 parameters, got {inputs.shape[1]}")
            # Run the model for all rows
            outputs = []
            for single_input in inputs:
                par = {self.ix_to_par[ix]: val for ix, val in enumerate(single_input)}
                outputs.append(self.simple_test_model(par))
            return np.array(outputs)
        if (isinstance(inputs, np.ndarray) and inputs.ndim == 1) or isinstance(
            inputs, Sequence
        ):
            if len(inputs) != 11:
                raise ValueError(f"Expected 11 parameters, got {len(inputs)}")
            par = {self.ix_to_par[ix]: val for ix, val in enumerate(inputs)}
            return np.array(self.simple_test_model(par))


@dataclass
class ReEmissionSALibWrapper:
    """A simple wrapper for ReEmission model to be used with SALib analysis.
    This wrapper takes a model function and a list of variables, and evaluates the model
    for given parameters.
    Attributes:
        input: Input object used for model parameter access.
        ix_name_map: Mapping between variable index and variable name.
        output_variable: The output variable to be computed ('co2_net', 'ch4_net', 'total_net').
        accessors: A dictionary of accessors for model parameters.
    Raises:
        ValueError: If the number of inputs does not match the number of parameters.
    """

    _supported_emissions: ClassVar = ("co2_net", "ch4_net", "total_net")
    input: Input
    ix_name_map: Dict[
        int, ParameterName
    ]  # mapping between variable index and variable name
    variables: List[Variable]
    emission: str
    p_model: str = "g-res"  #'mcdowell' #'g-res'
    accessors: Dict[ParameterName, AccessProtocol] = field(default_factory=dict)
    _name_ix_map: Dict[ParameterName, int] = field(init=False, repr=False)

    @classmethod
    def from_variables(
        cls,
        variables: List[Variable],
        input: Input,
        emission: str,
        accessors: Dict[ParameterName, int],
    ) -> ReEmissionSALibWrapper:
        """Create a ReEmissionSALibWrapper from a list of variables and an Input object.
        Args:
            variables: List of Variable objects representing model parameters.
            input: Input object used for model parameter access.
            emission: The emission type to be computed ('co2_net', 'ch4_net', 'total_net').
            accessors: A dictionary of accessors for model parameters.
        Returns:
            ReEmissionSALibWrapper: An instance of the wrapper with the specified parameters.
        """
        # Mapping between input index and variable / parameter name
        ix_name_map = {ix: var.name for ix, var in enumerate(variables)}
        return cls(
            input=input,
            ix_name_map=ix_name_map,
            variables=variables,
            emission=emission,
            accessors=accessors,
        )

    @staticmethod
    def get_input_dimension(inputs: SALibModelInputs) -> int:
        """
        Returns the number of parameters (columns if 2D, length if 1D) in SALibModelInputs.
        """
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 2:
                # 2D array: shape (n_samples, n_parameters)
                return inputs.shape[1]
            elif inputs.ndim == 1:
                # 1D array: shape (n_parameters,)
                return inputs.shape[0]
            else:
                raise ValueError("Inputs of dimensions greater than 2 not supported.")
        # For other sequences (list, tuple, etc.)
        if isinstance(inputs, Sequence):
            return len(inputs)
        raise TypeError("Unsupported input type for SALibModelInputs")

    @staticmethod
    def get_input_ndim(inputs: SALibModelInputs) -> int:
        """
        Returns the number of dimensions of SALibModelInputs.
        Raises ValueError if dimensions are greater than 2.
        """
        if isinstance(inputs, np.ndarray):
            ndim = inputs.ndim
            if ndim > 2:
                raise ValueError(
                    f"Inputs with ndim > 2 are not supported (got ndim={ndim})."
                )
            return ndim
        # For other sequences (list, tuple, etc.), treat as 1D
        if isinstance(inputs, Sequence):
            return 1
        raise TypeError("Unsupported input type for SALibModelInputs")

    def __post_init__(self) -> None:
        # Assert that each parameter name in ix_name_map has an accessor
        missing_keys = set(self.ix_name_map.values()) - self.accessors.keys()
        if missing_keys:
            raise ValueError(f"Missing accessors for: {missing_keys}")
        # Calculate _name_ix_map from ix_name_map
        self._name_ix_map = {name: ix for ix, name in self.ix_name_map.items()}
        if self.emission not in self._supported_emissions:
            raise ValueError(
                "Emission %s not in the list of supported emissions.", self.emission
            )

    def run(self, inputs: SALibModelInputs) -> SALibModelOutput:
        """Run the ReEmission model with inputs provided as a NDArray or an Iterable."""

        def run_single(single_input_vector) -> Numerical:
            """ """
            for par_index, par in enumerate(single_input_vector):
                # par_name = self.ix_name_map[par_index]
                par_value = (
                    self.variables[par_index].num_to_cat(par)
                    if self.variables[par_index].is_categorical
                    else par
                )
                self.accessors[self.ix_name_map[par_index]].set_value(value=par_value)

            # Wrap the input inside the Inputs class for batch processing
            input_data = Inputs(inputs={self.input.name: self.input})
            model = EmissionModel(inputs=input_data, p_model=self.p_model)
            model.calculate()
            if self.emission == "total_net":
                return (
                    model.outputs[self.input.name]["co2_net"]
                    + model.outputs[self.input.name]["ch4_net"]
                )
            return model.outputs[self.input.name][self.emission]

        # Check input dimensions
        if self.get_input_dimension(inputs) != len(self.ix_name_map):
            raise ValueError(
                f"Expected {len(self.ix_name_map)} parameters, got {len(inputs)}"
            )

        if self.get_input_ndim(inputs) == 2:
            # Process each individual input in a loop
            out = []
            for single_input in inputs:
                out.append(run_single(single_input))
        else:
            out = [run_single(inputs)]
        out_array = np.array(out)
        if len(out_array) == 0:
            out_array = out_array[0]
        return out_array

    def get_name(self, ix: int) -> str:
        """Return the name of the parameter based on index"""
        name = self.ix_name_map.get(ix)
        if name is None:
            raise IndexError(f"Index {ix} not found in the index map.")
        return name

    def get_index(self, var_name: str, raise_error: bool = False) -> int | None:
        """Return index of the paremeter in the sequence of input values from the
        parameter name.
        """
        ix = self._name_ix_map.get(var_name)
        if ix is None and raise_error:
            raise IndexError(f"Variable {var_name} not in the index map.")
        return ix


if __name__ == "__main__":
    # Example usage
    pass
