""" """
from __future__ import annotations
from typing import (
    List, Dict, Tuple, Any, Optional, Iterable, Callable, TypeVar, Sequence,
    Literal, TypeAlias)
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from SALib.analyze import sobol
from SALib.util.results import ResultDict
from SALib.sample.sobol import sample as sobol_sample
from reemission.salib.distributions import (
    Variable, DistributionType, NormalDistribution, DiscreteNormalDistribution,
    DiscreteUniformDistribution, UniformDistribution)
from reemission.salib.wrappers import SALibModelWrapperProtocol


NumpyValue = TypeVar('NumpyValue', np.floating, np.integer)
Numerical = TypeVar('Numerical', float, int)
SALibModel = Callable[[NDArray[NumpyValue] | Iterable[Numerical]], NumpyValue | Numerical]
ModelInputsType = TypeVar('ModelInputsType', bound=Sequence[Variable])
FixedVarName: TypeAlias = str
ScenarioName: TypeAlias = str
SobolIndexName: TypeAlias = Literal['S1', 'S2', 'ST']
GroupName: TypeAlias = str


@dataclass
class VarianceContributions:
    """ A dataclass to hold variances for each scenario storing contributions of variable groups
    to total variance.
    
    Attributes:
        sobol_index: The Sobol index type (e.g., 'S1', 'S2', 'ST') used for estimating variance
            contributions to total variance from each variable group.
        total_variance: The total variance of the output for this scenario.
        contributions_by_group: A dictionary mapping group names to their contributions to the total variance.
    """
    sobol_index: SobolIndexName
    total_variance: Numerical
    contributions_by_group: Dict[GroupName, Numerical]

    def __post_init__(self) -> None:
        """ Post-initialization to validate the Sobol index.
        Raises:
            ValueError: If the Sobol index is not one of the supported indices ('S1', 'S2', 'ST').
        """
        supported_sobol_indices = ('S1', 'S2', 'ST')
        if self.sobol_index not in supported_sobol_indices:
            raise ValueError(
                f"Sobol index must be: {'.'.join(supported_sobol_indices)} - {self.sobol_index} given.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the `VarianceContributions` instance to a dictionary."""
        return {
            'Sobol Index': self.sobol_index,
            'total_variance': self.total_variance,
            'contributions': self.contributions_by_group
        }


@dataclass(frozen=True)
class SobolScenarioResults:
    """ A class to hold the results of Sobol sensitivity analysis for multiple scenarios.
    Attributes:
        results: A list of SobolResults for each scenario.
        sc_names: A list of scenario names.
        ix_to_name: A mapping from scenario index to scenario name.
        
    NOTE: This class is used to hold the results of Sobol sensitivity analysis for multiple
    scenarios, where each scenario can fix some variables to specific values.
    It contains a list of SobolResults for each scenario and a list of VarianceContributions
    for each scenario, which holds the total variance and contributions of each group to
    the total variance.
    """

    results: List[SobolResults] = field(repr=False)
    sc_names: List[str]
    ix_to_name: Dict[int, str] = field(init=False)
    
    @property
    def var_names(self) -> List[str]:
        """ Returns the variable names for the scenarios. """
        return self.results[0].var_names
    
    @property
    def variances(self) -> List[VarianceContributions]:
        """Contributions of variable groups to total variance"""
        return [result.variance_contributions for result in self.results]
    
    @property
    def scenario_names(self) -> List[str]:
        """ Returns the scenario names for the Sobol results."""
        return self.sc_names
    
    @property
    def group_names(self) -> List[str]:
        """ Returns the group names for the Sobol results."""
        return list(self.results[0].grouped_indices['S1'].keys())
    
    @property
    def nominal_outputs(self) -> List[Numerical]:
        """ Returns the nominal outputs for each scenario. """
        return [result.nominal_output for result in self.results]
    
    def __post_init__(self) -> None:
        """ """
        object.__setattr__(
            self, 
            "ix_to_name", 
            {ix: scenario_name for ix, scenario_name in enumerate(self.sc_names)})


@dataclass(frozen=True)
class SobolResults:
    """ A class to hold the results of Sobol sensitivity analysis.
    Attributes:
        outputs: The model outputs for the Sobol samples.
        nominal_output: The model output for the nominal input.
        indices: A dictionary containing Sobol indices (first-order, second-order, total effect).
        var_names: A list of variable names corresponding to the Sobol indices.
        grouped_indices: A dictionary containing grouped Sobol indices by variable groups.
        variance_contributions: Optional VarianceContributions for the total variance and contributions.
    Raises:
        ValueError: If the indices do not contain the expected keys.
    Usage:
        >>> from utils.salib import SobolResults
        >>> from utils.distributions import Variable, NormalDistribution
        >>> outputs = np.array([1.0, 2.0, 3.0])
        >>> nominal_output = 2.0
        >>> indices = {
        ...     'S1': [0.1, 0.2, 0.3],
        ...     'S2': [0.05, 0.1, 0.15],
        ...     'ST': [0.2, 0.3, 0.4]
        ... }
        >>> var_names = ['a', 'b', 'c']
        >>> sobol_results = SobolResults(
        ...     outputs=outputs,
        ...     nominal_output=nominal_output,
        ...     indices=indices,
        ...     var_names=var_names
        ... )
        >>> print(sobol_results.first_order)  # Output: [0.1, 0.2, 0.3]
    """
    outputs: NDArray
    nominal_output: np.float | np.int
    indices: ResultDict
    var_names: List[str]
    fixed_var_names: Optional[List[str]] = None
    grouped_indices: Dict[str, ResultDict] = field(
        default_factory=lambda: defaultdict(dict))
    variance_contributions: Optional[VarianceContributions] = None
    
    @property
    def first_order(self) -> List[Numerical]:
        """ Returns the first-order Sobol indices. """
        return self.indices['S1']
    
    @property
    def second_order(self) -> List[Numerical]:
        """ Returns the second-order Sobol indices. """
        if 'S2' in self.indices:
            return self.indices['S2']
        return []
    
    @property
    def fixed_variables(self) -> List[str]:
        """ Returns the names of fixed variables if any are defined."""
        return self.fixed_var_names if self.fixed_var_names else []
    
    @property
    def total_effect(self) -> List[Numerical]:
        """ Returns the total effect Sobol indices. """
        return self.indices['ST']
    
    @property
    def groups(self) -> List[str]:
        """
        Return the group names from the first entry in grouped_indices,
        or an empty list if grouped_indices is empty.
        """
        if not self.grouped_indices:
            return []
        # Get the first key in grouped_indices
        first_key = next(iter(self.grouped_indices))
        # Return the keys of the dictionary belonging to the first key
        return list(self.grouped_indices[first_key].keys())


@dataclass
class SobolAnalyser:
    """ A class to perform Sobol sensitivity analysis using SALib.
    
    Attributes:
        problem: A SALibProblem instance defining the problem.
        variables: A list of Variable objects defining the model parameters.
        model: A callable model function that takes a list of variables.
        num_sampes: Number of samples to generate with Sobol sampling.
        calc_second_order: If True, calculates second-order Sobol indices.
        group_indices: If True, groups Sobol indices by variable groups.
    Raises:
        ValueError: If the number of variables in the problem does not match the number of variables
            provided, or if the variable names do not match the names in the problem definition.
    Usage:
        >>> from utils.salib import SobolAnalyser, SALibProblem
        >>> from utils.distributions import Variable, NormalDistribution
        >>> problem = SALibProblem.from_variables(
        ...     variables=[
        ...         Variable(name='a', distribution=NormalDistribution(mean=0.0, std_dev=1.0)),
        ...         Variable(name='b', distribution=NormalDistribution(mean=0.0, std_dev=1.0)),
        ...         Variable(name='c', distribution=DiscreteNormalDistribution(mean=0.0, std_dev=1.0))
        ...     ]
        ... )
        >>> model = TestModelSALibWrapper.from_variables(problem.variables)
        >>> analyser = SobolAnalyser(
        ...     problem=problem,    
        ...     variables=problem.variables,
        ...     model=model,
        ...     num_samples=1000,
        ...     calc_second_order=True,
        ...     group_indices=True
        ... )
        >>> sobol_results = analyser.run_sobol_sensitivity()
        >>> print(sobol_results)
    """
    problem: SALibProblem
    variables: List[Variable]
    model: SALibModelWrapperProtocol
    num_samples: int
    calc_second_order: bool = True
    group_indices: bool = True
    _variable_cache: dict[str, Variable] = field(init=False, repr=False, default_factory=dict)
    
    def __post_init__(self) -> None:
        """ """
        if self.problem.num_vars != len(self.variables):
            raise ValueError(
                "Number of variables in the problem does not match the number of variables provided.")
        if set(self.problem.names) != set(var.name for var in self.variables):
            raise ValueError(
                "Variable names do not match the names in the problem definition.")
        if self.problem.names != [var.name for var in self.variables]:
            raise ValueError(
                "Order of variables in the problem specification is different from that in the variable list.")
        object.__setattr__(self, "_variable_cache", {})
        
    def get_variable(self, var_name: str) -> Optional[Variable]:
        """Get a variable by its name, with memoization."""
        cache = self._variable_cache
        if var_name in cache:
            return cache[var_name]
        for var in self.variables:
            if var.name == var_name:
                cache[var_name] = var
                return var
        cache[var_name] = None
        return None
    
    def run_sobol_sampling(self, **kwargs) -> NDArray[NumpyValue]:
        """Run Sobol sampling to generate input samples."""
        # Step 1 - Generate Sobol samples
        sampled_inputs = sobol_sample(
            problem=self.problem.to_dict(),
            N=self.num_samples,  # The number of samples will be N * (2D + 2) where D is the number of variables
            calc_second_order=self.calc_second_order,
            **kwargs
        )
        # Step 2. Transform samples so that they follow tue parameter distributions
        if self.problem.using_unit_bounds:
            # If the bounds have been normalized, we need to transform them back to original
            sampled_inputs = transform_uniform(sampled_inputs, self.variables)
        return sampled_inputs
    
    def calculate_grouped_indices(
            self,
            sobol_indices: ResultDict
        ) -> Dict[SobolIndexName, Dict[GroupName, Numerical]]:
        """ Aggregate Sobol indices by variable groups.
        If no groups are defined, return the Sobol indices as is.
        Args:
            sobol_indices: A dictionary containing Sobol indices.
        Returns:
            A dictionary with aggregated Sobol indices by groups.
        Raises:
            ValueError: If the sobol_indices do not contain the expected keys.
        """
        grouped_sobol_indices = {}
        if not self.problem.groups:
            # If there are no groups defined, return an empty dict
            return grouped_sobol_indices
        grouped_sobol_indices['S1'] = \
            {group: sobol_indices['S1'][idxs].sum() for group, idxs in self.problem.groups.items()}
        if 'S2' in sobol_indices:
            # If second-order indices are calculated, aggregate them as well
            grouped_sobol_indices['S2'] = \
                {group: sobol_indices['S2'][idxs].sum() for group, idxs in self.problem.groups.items()}
        grouped_sobol_indices['ST'] = \
            {group: sobol_indices['ST'][idxs].sum() for group, idxs in self.problem.groups.items()}
        return grouped_sobol_indices
    
    def run_sobol_scenarios(
            self,
            scenarios: Dict[ScenarioName, Dict[FixedVarName, Numerical]],
            pi_bounds: Tuple[float, float] = (0.25, 0.975),
            sobol_index: SobolIndexName = 'ST',
            **kwargs) -> SobolScenarioResults:
        """
        Run Sobol analysis for multiple scenarios, where each scenario fixes some variables.
        Returns a dictionary mapping scenario names to SobolResults.
        Args:
            scenarios: A dictionary where keys are scenario names and values are dictionaries
                mapping variable names to fixed values for that scenario.
            pi_bounds: A tuple of two floats representing the lower and upper bounds for the
                percentile interval to calculate the nominal output.
            sobol_index: The Sobol index type to use for estimating variance contributions.
            **kwargs: Additional keyword arguments to pass to the Sobol sampling function.
        """
        # NOTE: Variable has a `fixed` field but it is not used here currently.
        scenario_ix_map: Dict[ScenarioName, Dict[int, Numerical]] = {
            scenario_name: {
                self.model.get_index(var_name, raise_error=True): fixed_value
                for var_name, fixed_value in fixed_var_names.items()
            }
            for scenario_name, fixed_var_names in scenarios.items()
        }
        nominal_outputs = []
        sobol_results = []
        scenario_names = []
        # Step 1 - Generate Sobol samples
        sampled_inputs = self.run_sobol_sampling(**kwargs)
        
        # Fix variables in the sampled inputs according to the scenario specifications
        for scenario_name, fixed_vars in scenario_ix_map.items():
            scenario_names.append(scenario_name)
            fixed_var_names = {
                self.variables[ix].name: value for ix, value in fixed_vars.items()}
            # Create a copy of the sampled inputs to modify
            fixed_inputs = sampled_inputs.copy()
            for ix, fixed_value in fixed_vars.items():
                if ix < len(fixed_inputs):
                    fixed_inputs[:, ix] = fixed_value
                else:
                    raise ValueError(f"Index {ix} is out of bounds for the input array.")
            # Step 2. Evaluate the model for the sampled inputs
            outputs = self.model.run(fixed_inputs)
            # Step 3. Evaluate the model for the nominal input
            ci_lower = np.percentile(outputs, pi_bounds[0])
            ci_upper = np.percentile(outputs, pi_bounds[1])
            nominal_values = [
                fixed_vars[ix] if ix in fixed_vars else var.distribution.nominal_value for ix, var in enumerate(self.variables)]
            # NOTE: This should work as long as the index map is generated from variables!!! 
            #       Double check later!!
            nominal_output = self.model.run(nominal_values)
            nominal_outputs.append(nominal_output)
            # Calculate sobol indices
            sobol_indices = sobol.analyze(
                self.problem.to_dict(), outputs, print_to_console=False)
            ST_local = sobol_indices['ST']
            # Group indices by variable groups (defined in each `variable` object)
            if not self.calc_second_order:
                sobol_indices.pop("S2")
            # Step 5 - 
            if self.group_indices:
                grouped_sobol_indices = self.calculate_grouped_indices(sobol_indices)
            else:
                grouped_sobol_indices = {}
            
            var_Y = np.var(outputs, ddof=1)
            sc_var = VarianceContributions(
                sobol_index = sobol_index,
                total_variance=var_Y,
                contributions_by_group={
                    group: grouped_sobol_indices[sobol_index][group] * var_Y
                    for group in grouped_sobol_indices[sobol_index]
                })
            
            sobol_result = SobolResults(
                outputs = outputs,
                nominal_output = nominal_output,
                indices = sobol_indices,
                var_names = [var.name for var in self.variables],
                fixed_var_names = fixed_var_names,
                grouped_indices=grouped_sobol_indices,
                variance_contributions = sc_var
            )
            sobol_results.append(sobol_result)

        return SobolScenarioResults(
            sc_names = scenario_names,
            results = sobol_results,
        )
    
    def run_sobol(
            self,
            fix_variables: bool = False,
            sobol_index: SobolIndexName = 'ST',
            **kwargs) -> SobolResults:
        """Run Sobol sensitivity analysis.
        Args:
            fix_variables: If True, fix the variables to their nominal values.
            sobol_index: The Sobol index type to use for estimating variance contributions.
            **kwargs: Additional keyword arguments to pass to the Sobol sampling function.
        """
        if fix_variables:
            raise NotImplementedError(
                "Fixing variables is not implemented in this method. Use `run_sobol_scenarios` instead.")
        # Step 1 - Generate Sobol samples
        sampled_inputs = self.run_sobol_sampling(**kwargs)
        # Step 2. Evaluate the model for the sampled inputs
        outputs = self.model.run(sampled_inputs) # pylint: disable=assignment-from-no-return
        # Step 3. Evaluate the model for the nominal input
        nominal_values = [var.distribution.nominal_value for var in self.variables]
        # NOTE: This should work as long as the index map is generated from variables!!! Double check later!!
        nominal_output = self.model.run(nominal_values)
        # Step 4 - Calculate sobol indices
        sobol_indices = sobol.analyze(self.problem.to_dict(), outputs, print_to_console=False)
        if not self.calc_second_order:
            sobol_indices.pop("S2")
        # Step 5 - Group indices by variable groups (defined in each `variable` object)
        if self.group_indices:
            grouped_sobol_indices = self.calculate_grouped_indices(sobol_indices)
            var_Y = np.var(outputs, ddof=1)
            sc_var = VarianceContributions(
                sobol_index = sobol_index,
                total_variance=var_Y,
                contributions_by_group={
                    group: grouped_sobol_indices[sobol_index][group] * var_Y
                    for group in grouped_sobol_indices[sobol_index]
                })            
        else:
            grouped_sobol_indices = {}
            sc_var = None  

        return SobolResults(
            outputs=outputs,
            nominal_output = nominal_output,
            indices=sobol_indices,
            var_names = [var.name for var in self.variables],
            fixed_var_names = None,
            grouped_indices=grouped_sobol_indices,
            variance_contributions = sc_var)


@dataclass
class SobolBatchAnalyser:
    """ A class to perform Sobol sensitivity analysis for a batch of inputs.
    Attributes:
        problem: A SALibProblem instance defining the problem.
        batch_variables: A dictionary mapping scenario names to lists of Variable objects defining the model parameters.
        models: A dictionary mapping scenario names to SALibModelWrapperProtocol instances.
        num_samples: Number of samples to generate with Sobol sampling.
        calc_second_order: If True, calculates second-order Sobol indices.
        group_indices: If True, groups Sobol indices by variable groups.
    Raises:
        ValueError: If the number of batch variables does not match the number of models,
            or if the variable names do not match the model names."""
    problem: SALibProblem
    batch_variables: Dict[ScenarioName, List[Variable]]
    models: Dict[ScenarioName, SALibModelWrapperProtocol]
    num_samples: int
    calc_second_order: bool = True
    group_indices: bool = True
    
    def __post_init__(self) -> None:
        """Post-initialization to validate the problem and batch variables."""
        if len(self.batch_variables) != len(self.models):
            raise ValueError("Number of batch variables must match the number of models.")
        if not set(self.batch_variables.keys()).issubset(set(self.models.keys())):
            raise ValueError("All batch variable names must match the model names.")
    
    def run_sobol_batch(
            self, 
            fix_variables: bool = False, 
            **kwargs) -> SobolScenarioResults:
        """Run Sobol sensitivity analysis for a batch of inputs."""
        # Similar to run_sobol_scenarios, but for a batch of inputs.
        sobol_batch_results: List[SobolResults] = []
        # TODO: Scenario Variances need to be evaluated in this method
        for scenario_name, variables in self.batch_variables.items():
            analyser = SobolAnalyser(
                problem=self.problem,
                variables=variables,
                model=self.models[scenario_name],
                num_samples=self.num_samples,
                calc_second_order=self.calc_second_order,
                group_indices=self.group_indices
            )
            results = analyser.run_sobol(fix_variables=fix_variables, **kwargs)
            sobol_batch_results.append(results)
            
        return SobolScenarioResults(
            sc_names = list(self.batch_variables.keys()),
            results = sobol_batch_results,
        )


@dataclass(frozen=True)
class SALibProblem:
    """ A dataclass to represent a SALib problem definition and construct
    SALib `problem` dictionary.
    
    Attributes:
        num_vars: Number of variables in the problem.
        names: List of variable names.
        bounds: List of bounds for each variable.
        groups: Optional dictionary mapping group names to lists of variable indices.
        use_groups_in_sobol: If True, use groups in Sobol analysis.
        using_unit_bounds: If True, all bounds are (0.0, 1.0).
    """
    num_vars: int = field(init=False)
    names: List[str]
    bounds: List[Tuple[float, float]] = field(default_factory=list)
    groups: Optional[Dict[str, List[int]]] = None
    use_groups_in_sobol: bool = False
    using_unit_bounds: bool = field(init=False)

    def __post_init__(self):
        """Post-initialization to set the number of variables."""
        object.__setattr__(self, "num_vars", len(self.names))
        if not self.bounds:
            object.__setattr__(self, "bounds", [(0.0, 1.0)] * self.num_vars)
            object.__setattr__(self, "using_unit_bounds", True)
        if len(self.bounds) != self.num_vars:
            raise ValueError("Length of bounds must match the number of variables.")
        object.__setattr__(self, "using_unit_bounds", all(bound == (0.0, 1.0) for bound in self.bounds))

    @classmethod
    def from_variables(
            cls,
            variables: List[Variable],
            use_groups_in_sobol: bool = False) -> SALibProblem:
        """Create a SALibProblem instance from a list of Variable objects.
        Args:
            variables: A list of Variable objects defining the model parameters.
            use_groups_in_sobol: If True, use groups in Sobol analysis.
        Returns:
            A SALibProblem instance.
        Raises:
            ValueError: If there are duplicate variable names or if the bounds are not defined.
        """
        var_names = [var.name for var in variables]
        duplicates = [name for name, count in Counter(var_names).items() if count > 1]
        groups  = defaultdict(list)
        for ix, var in enumerate(variables):
            if var.group:
                groups[var.group].append(ix)
            else:
                groups['unknown'].append(ix)
        if duplicates:
            raise ValueError(f"Duplicate variable names found: {duplicates}")
        distributions = [var.distribution for var in variables]
        if not all(dist.has_bounds() for dist in distributions):
            bounds = [(0.0, 1.0)] * len(variables)
        else:
            # Extract bounds from uniform and discrete-uniform distributions
            bounds = [dist.bounds for dist in distributions]
        return cls(
            names=var_names,
            bounds=bounds,
            groups=groups,
            use_groups_in_sobol=use_groups_in_sobol
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the SALibProblem instance to a dictionary.
        Returns:
            A dictionary representation of the SALib problem.
        """
        bounds_as_lists = [list(bound) for bound in self.bounds]
        problem_dict = {
            'num_vars': self.num_vars,
            'names': self.names,
            'bounds': bounds_as_lists,
        }
        if self.groups is not None and self.use_groups_in_sobol:
            # Convert a dict mapping keys to positions (int or iterable of ints)
            # into a list with keys placed at specified positions.
            # This function is used to convert the 'groups' dict into a list
            # where each index corresponds to a parameter, and the value is the group name.
            idx2key = {
                pos: key
                for key, val in self.groups.items()
                for pos in (val
                            if isinstance(val, Iterable) and not isinstance(val, (str, bytes))
                            else [val])
            }
            max_idx = max(idx2key, default=-1)
            # fill from 0..max_idx, missing slots will be None
            problem_dict['sobol_groups'] = [idx2key.get(i) for i in range(max_idx + 1)]
        return problem_dict


def transform_uniform(
        uniform_samples: NDArray[NumpyValue],
        variables: List[Variable], 
        eps: float = 1e-6) -> NDArray[NumpyValue]:
    """ Transforms uniform samples U to the distributions defined by the variables.
    Args:
        uniform_samples: NDArray of uniform samples, shape (num_samples, num_variables)
        variables: List of Variable objects defining the distributions.
        eps: Small value to prevent numerical issues when transforming uniform samples.
    Returns:
        NDArray of transformed samples, shape (num_samples, num_variables).
    Note:
        Variables in the `variables` list must be in the same order as the columns in `uniform_samples`.
    Raises:
        TypeError: If `uniform_samples` is not a numpy ndarray.
        ValueError: If `uniform_samples` is not a 2D array or if the number of variables does not match the shape of `uniform_samples`.
        ValueError: If `variables` list is empty or contains non-Variable objects.
        ValueError: If a variable's distribution type is not supported.
    """
    if not isinstance(uniform_samples, np.ndarray):
        raise TypeError("U must be a numpy ndarray")
    if uniform_samples.ndim != 2:
        raise ValueError("U must be a 2D array with shape (num_samples, num_variables)")
    if not variables:
        raise ValueError("variables list cannot be empty")
    if not all(isinstance(var, Variable) for var in variables):
        raise TypeError("All elements in variables must be instances of Variable class")
    num_samples, num_variables = uniform_samples.shape
    if num_variables != len(variables):
        raise ValueError("Number of variables does not match the shape of U")
    X = np.zeros_like(uniform_samples)
    for j, variable in enumerate(variables):
        uniform_samples_transformed = np.clip(uniform_samples[:, j], eps, 1-eps) # prevents numerical issues when transforming uniform samples to other distributions
        dist_type = variable.distribution.type
        if dist_type == DistributionType.NORMAL:
            mu = variable.distribution.mean
            sigma = variable.distribution.std_dev
            X[:, j] = norm.ppf(uniform_samples_transformed, loc=mu, scale=sigma)
        elif dist_type == DistributionType.UNIFORM:
            low, high = variable.distribution.bounds
            X[:, j] = low + uniform_samples_transformed * (high - low)
        elif dist_type == DistributionType.DISCRETEUNIFORM:
            low, high = variable.distribution.bound
            vals = np.arange(low, high + 1) # Add 1 so that we can use floor
            idx = np.floor(uniform_samples_transformed * len(vals)).astype(int) # Convert to discrete values
            idx = np.clip(idx, 0, len(vals) - 1)
            X[:, j] = vals[idx]
        elif dist_type == DistributionType.DISCRETENORMAL:
            mu = variable.distribution.mean
            sigma = variable.distribution.std_dev
            low, high = variable.distribution.bound
            vals = np.arange(low, high + 1)
            # Map uniform to normal, then to nearest integer in bounds
            norm_vals = norm.ppf(uniform_samples_transformed, loc=mu, scale=sigma)
            int_vals = np.round(norm_vals).astype(int)
            int_vals = np.clip(int_vals, low, high)
            X[:, j] = int_vals
        elif dist_type == DistributionType.DISCRETEHISTOGRAM:
            categories = np.array(variable.distribution.categories)
            probabilities = np.array(variable.distribution.probabilities)
            cum_probs = np.cumsum(probabilities)
            idx = np.searchsorted(cum_probs, uniform_samples_transformed, side="right")
            X[:, j] = categories[idx]
        elif dist_type == DistributionType.CATEGORICALUNIFORM:
            categories = np.array(variable.distribution.categories)
            idx = np.floor(uniform_samples_transformed * len(categories)).astype(int)
            idx = np.clip(idx, 0, len(categories) - 1)
            X[:, j] = variable.cat_to_num(categories[idx])
        elif dist_type == DistributionType.CATEGORICALNORMAL:
            categories = np.array(variable.distribution.categories)
            mu = variable.distribution.mean
            sigma = variable.distribution.std_dev
            idx = np.round(norm.ppf(uniform_samples_transformed, loc=mu, scale=sigma)).astype(int)
            idx = np.clip(idx, 0, len(categories) - 1)
            X[:, j] = variable.cat_to_num(categories[idx])
        elif dist_type == DistributionType.CATEGORICALHISTOGRAM:
            categories = np.array(variable.distribution.categories)
            probabilities = np.array(variable.distribution.probabilities)
            cum_probs = np.cumsum(probabilities)
            idx = np.searchsorted(cum_probs, uniform_samples_transformed, side="right")
            X[:, j] = variable.cat_to_num(categories[idx])
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return X


if __name__ == "__main__":
    # Example usage
    salib_problem_1 = SALibProblem.from_variables(
        variables=[
            Variable(name='a', distribution=NormalDistribution(mean=0.0, std_dev=1.0)),
            Variable(name='b', distribution=NormalDistribution(mean=0.0, std_dev=1.0)),
            Variable(name='c', distribution=DiscreteNormalDistribution(mean=0.0, std_dev=1.0))
        ]
    )
    salib_problem_2 = SALibProblem.from_variables(
        variables=[
            Variable(name='a', distribution=UniformDistribution(bounds=(0.5, 1.634))),
            Variable(name='b', distribution=DiscreteUniformDistribution(bounds=(1, 4.0))),
        ]
    )
    print("SALib problem 1", salib_problem_1.to_dict())
    print("SALib problem 2", salib_problem_2.to_dict())
