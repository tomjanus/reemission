""" """
from __future__ import annotations
import collections.abc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, ClassVar, Dict, Sequence
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from rich import print as rprint


# TODO: Add a distribution defined by an arbitrary probability density function

# TODO: Check that discrete distributions work properly -- like categorical ones.
#       Add support for nominal values for discrete distributions in spec - just like with categorical distributions

class DistributionType(Enum):
    """ Enum for distribution types """
    # These are used to identify the type of distribution for a variable
    NORMAL = "real-normal"
    UNIFORM = "real-uniform"
    DISCRETENORMAL = "discrete-normal" # discrete variables with normal distribution
    DISCRETEUNIFORM = "discrete-uniform" # discrete variables with equal probabilities can usually be converted to discrete uniform distribution
    DISCRETEHISTOGRAM = "discrete-histogram" # discrete variables with non-equal probabilities
    CATEGORICALNORMAL = "categorical-normal" # categorical variables with normal distribution
    CATEGORICALUNIFORM = "categorical-uniform" # categorical variables with equal probabilities
    CATEGORICALHISTOGRAM = "categorical-histogram" # categorical variables with non-equal probabilities


class Distribution(ABC):
    """Abstract base for parameterizations (std-dev or min/max)."""

    @property
    @abstractmethod
    def type(self) -> str:  # pylint: disable=missing-function-docstring
        """
        Returns the type of the parameter as a string.

        Returns:
            str: The type of the parameter.
        """
        
    def has_bounds(self) -> bool:
        """ Check if the distribution has bounds """
        return self.type in (
            DistributionType.UNIFORM,
            DistributionType.DISCRETEUNIFORM
        )
        
    def is_discrete(self) -> bool:
        """ Check if the distribution is discrete """
        return self.type in (
            DistributionType.DISCRETENORMAL,
            DistributionType.DISCRETEUNIFORM,
            DistributionType.DISCRETEHISTOGRAM,
            DistributionType.CATEGORICALNORMAL,
            DistributionType.CATEGORICALUNIFORM,
            DistributionType.CATEGORICALHISTOGRAM
        )
        
    def is_normal(self) -> bool:
        """ Check if the distribution is normal """
        return self.type in (
            DistributionType.NORMAL,
            DistributionType.DISCRETENORMAL,
            DistributionType.CATEGORICALNORMAL
        )

    def is_uniform(self) -> bool:
        """ Check if the distribution is uniform or discrete uniform """
        return self.type in (
            DistributionType.UNIFORM,
            DistributionType.DISCRETEUNIFORM,
            DistributionType.CATEGORICALUNIFORM
        )

    def is_categorical(self) -> bool:
        """ Check if the distribution is categorical """
        return self.type in (
            DistributionType.CATEGORICALUNIFORM,
            DistributionType.CATEGORICALHISTOGRAM,
            DistributionType.CATEGORICALNORMAL
        )

    @property
    @abstractmethod
    def nominal_value(self) -> int | float | None: # pylint: disable=missing-function-docstring
        """ Returns the nominal value of the distribution"
        
        Returns:
            int | float | List[int]: The nominal value or range of values for the distribution.
        """
        
    @abstractmethod
    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value, if applicable.
        
        Returns:
            Distribution: A fixed distribution with bounds set to the nominal value.
        """
        
    @abstractmethod
    def sample(self, n_samples: int) -> NDArray: # pylint: disable=missing-function-docstring
        """ Generate N samples from the distribution 
        
        Returns:
            NDArray: An array of samples drawn from the distribution.
        """

    # Added here to support Pylint
    def cat_to_num(self, cat_values: Sequence[str]) -> Sequence[int]:
        """ Default implementation for converting categories to numbers """
        raise NotImplementedError(f"{self.__class__.__name__} does not support categorical conversion.")

    # Added here to support Pylint
    def num_to_cat(self, num_values: Sequence[int]) -> Sequence[str]:
        """ Default implementation for converting numbers to categories """
        raise NotImplementedError(f"{self.__class__.__name__} does not support numerical-to-categorical conversion.")
    
    @abstractmethod
    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """


@dataclass
class CategoricalConversionMixin:
    """ Mixin for categorical conversion methods. """
    _cat_to_num: Dict[str, int] = field(init=False, repr=False)
    _num_to_cat: Dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_cat_to_num", {cat: i for i, cat in enumerate(self.categories)})
        object.__setattr__(self, "_num_to_cat", dict(enumerate(self.categories)))
    
    @staticmethod
    def is_iterable(value) -> bool:
        """ Check if the value is an iterable.
        Args:
            value: The value to check.
        Returns: bool: True if the value is an iterable and not a scalar, False otherwise.
        """
        # Exclude numpy scalars and Python scalars
        if np.isscalar(value):
            return False
        # Exclude strings and bytes if needed
        if isinstance(value, (str, bytes)):
            return False
        return isinstance(value, collections.abc.Iterable)

    def cat_to_num(self, cat_values: Sequence[str] | str) -> Sequence[int] | int:
        """ Convert categorical values to numerical indices """
        if not self.is_iterable(cat_values):
            return self._cat_to_num.get(cat_values, None)
        return [self._cat_to_num[cat] for cat in cat_values if cat in self._cat_to_num]

    def num_to_cat(self, num_values: Sequence[int] | int) -> Sequence[str] | str:
        """ Convert numerical indices to categorical values """
        if not self.is_iterable(num_values):
            return self._num_to_cat.get(int(num_values), None)
        return [self._num_to_cat[int(num)] for num in num_values if int(num) in self._num_to_cat]


@dataclass
class NormalDistribution(Distribution):
    """ Standard (Gaussian) distribution 
    Attributes:
        mean (int | float): Mean of the normal distribution.
        std_dev (int | float): Standard deviation of the normal distribution.
    """
    _type: ClassVar[DistributionType] = DistributionType.NORMAL
    mean: int | float
    std_dev: int | float

    def __post_init__(self) -> None:
        """ Ensure mean and std_dev are valid for normal distribution """
        if self.std_dev <= 0:
            raise ValueError("Standard deviation must be positive for normal distribution")

    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> int | float:
        return self.mean
    
    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        # Return a new instance with std_dev set to 0
        return NormalDistribution(mean=self.nominal_value, std_dev=0.0)

    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the normal distribution """
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=n_samples)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.mean,
            "std_dev": self.std_dev
        }


@dataclass
class UniformDistribution(Distribution):
    """ Uniform distribution supported by SALib and Sobol sampling
    Attributes:
        bounds (Tuple[float, float]): Lower and upper bounds of the uniform distribution.
    """
    _type: ClassVar[DistributionType] = DistributionType.UNIFORM
    bounds: Tuple[float, float]
    
    def __post_init__(self) -> None:
        """ Ensure bounds are valid for uniform distribution """
        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("Bound must be a tuple of two elements (lower, upper)")
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError("Lower bound must be less than upper bound for uniform distribution")
    
    @property
    def type(self) -> str:
        return self._type
    
    @property
    def nominal_value(self) -> int | float:
        return (self.bounds[0] + self.bounds[1]) / 2
    
    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        return UniformDistribution(
            bounds=(self.nominal_value, self.nominal_value))
    
    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the uniform distribution """
        return np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=n_samples)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "bounds": self.bounds}


@dataclass
class DiscreteNormalDistribution(Distribution):
    """ Discrete normal distribution for numerical variables
    Attributes:
        mean (int): Mean of the discrete normal distribution (must be an integer).
        std_dev (float): Standard deviation of the discrete normal distribution (must be positive).
    """
    _type: ClassVar[DistributionType] = DistributionType.DISCRETENORMAL
    mean: int
    std_dev: float
    
    def __post_init__(self) -> None:
        """ Ensure mean is an integer for discrete normal distribution """
        if isinstance(self.mean, float) and not self.mean.is_integer(): # pylint: disable=no-member 
            raise ValueError(f"Mean must be an integer or a float equivalent to an integer, got {self.mean}")
        object.__setattr__(self, "mean", int(self.mean))
        if self.std_dev <= 0:
            raise ValueError("Standard deviation must be positive for discrete normal distribution")

    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> int:
        return int(self.mean)
    
    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        # Return a new instance with std_dev set to 0
        return DiscreteNormalDistribution(
            mean=self.nominal_value, std_dev=0.0)

    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the discrete normal distribution """
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=n_samples).round().astype(int)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "mean": self.mean,
            "std_dev": self.std_dev}


@dataclass
class DiscreteUniformDistribution(Distribution):
    """ Discrete uniform distribution for numerical variables
    Attributes:
        bounds (Tuple[int, int]): Lower and upper bounds of the discrete uniform distribution (must be integers).
    """
    _type: ClassVar[DistributionType] = DistributionType.DISCRETEUNIFORM
    bounds: Tuple[int, int]
    
    def __post_init__(self) -> None:
        """ Ensure bounds are integers for discrete uniform distribution """
        if not all(isinstance(b, (int, float)) and (isinstance(b, int) or b.is_integer()) for b in self.bounds):
            raise ValueError("Bounds must be integers or floats equivalent to integers for discrete uniform distribution")
        # Convert all to int
        converted_bounds = tuple(int(b) for b in self.bounds)
        object.__setattr__(self, "bounds", converted_bounds)
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError("Lower bound must be less than upper bound for discrete uniform distribution")

    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> int:
        return (int(self.bounds[0]) + int(self.bounds[1])) // 2

    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        return DiscreteUniformDistribution(
            bounds=(self.nominal_value, self.nominal_value))
        
    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the discrete uniform distribution """
        return np.random.choice(self.nominal_value, size=n_samples)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "bounds": self.bounds}


@dataclass
class DiscreteHistogramDistribution(Distribution):
    """ Discrete histogram distribution for numerical variables.
    Attributes:
        values (Tuple[int, ...]): Values for the discrete histogram distribution.
        probabilities (Tuple[float, ...]): Probabilities for each value in the discrete histogram distribution.
    """
    _type: ClassVar[DistributionType] = DistributionType.DISCRETEHISTOGRAM
    values: Tuple[int, ...] = field(default_factory=tuple)
    probabilities: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """ Ensure that values and probabilities are valid for discrete histogram distribution """
        if not self.values:
            raise ValueError("Values must not be empty for discrete histogram distribution")
        if len(self.values) != len(self.probabilities):
            raise ValueError("Values and probabilities must have the same length")
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1 for discrete histogram distribution")

    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> int:
        """Returns the category with the highest probability, or the middle category if all probabilities are equal."""
        # Check if all probabilities are (almost) equal
        if np.allclose(self.probabilities, self.probabilities[0]):
            # Return the middle category
            mid_idx = len(self.values) // 2
            return self.values[mid_idx]
        # Return the category with the highest probability
        max_index = np.argmax(self.probabilities)
        return self.values[max_index]

    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        return DiscreteHistogramDistribution(
            values=(self.nominal_value, ),
            probabilities = (1, ))

    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the discrete histogram distribution """
        return np.random.choice(self.values, size=n_samples, p=self.probabilities)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "values": self.values,
            "probabilities": self.probabilities}


@dataclass
class CategoricalNormalDistribution(CategoricalConversionMixin, Distribution):
    """ Normal distribution for categorical variables 
    This distribution is defined by a mean and standard deviation, and samples are drawn from a 
    normal distribution with respect to the categories.
    Attributes:
        categories (Tuple[str, ...]): Categories for the categorical variable.
        mean (float): Mean of the normal distribution.
        std_dev (float): Standard deviation of the normal distribution.
        nominal_as_index (bool): If True, nominal_value is the middle category index, else it's the middle category itself.
        nominal_category (Optional[str]): If provided, overrides nominal_as_index to use this category as the nominal value.
    """
    _type: ClassVar[DistributionType] = DistributionType.CATEGORICALNORMAL
    categories: Tuple[str, ...] = field(default_factory=tuple)
    mean: float = 0.0
    std_dev: float = 1.0
    nominal_as_index: bool = True  # If True, nominal_value is the middle category index, else it's the middle category itself
    nominal_category: Optional[str] = None  # If provided, overrides nominal_as_index
    
    def __post_init__(self) -> None:
        """ Ensure categories are valid for categorical normal distribution """
        if not self.categories:
            raise ValueError("Categories must not be empty for categorical normal distribution")
        if self.std_dev <= 0:
            raise ValueError("Standard deviation must be positive for categorical normal distribution")
        if self.nominal_category and self.nominal_category not in self.categories:
            raise ValueError(f"Nominal category '{self.nominal_category}' must be one of the categories: {self.categories}")
        # Call the mixin's __post_init__ to initialize _cat_to_num and _num_to_cat
        CategoricalConversionMixin.__post_init__(self)

    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> str:
        """Return the category whose index is closest to the mean or return the nominal catergory
        if provided as attribute."""
        # Find index of nominal category if <nominal_category is not None
        if self.nominal_category:
            index = self._cat_to_num.get(self.nominal_category, None)
            if index is None:
                raise ValueError(f"Nominal category '{self.nominal_category}' not found in categories: {self.categories}")
        else:
            index = round(self.mean)
            index = max(0, min(index, len(self.categories) - 1))  # Clamp to valid index range
        if self.nominal_as_index:
            return index
        return self.categories[index]
    
    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        return CategoricalNormalDistribution(
            categories = (self.nominal_value, ),
            mean = 0.0,  # Nominal value is the only category, so mean is set to 0)
            std_dev=0.0)  # Standard deviation is set to 0 for a fixed value

    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the normal distribution for categorical variables """
        indices = np.arange(len(self.categories))
        probs = np.exp(-0.5 * ((indices - self.mean) / self.std_dev) ** 2)
        probs /= probs.sum()  # normalize to get valid probabilities
        return np.random.choice(self.categories, size=n_samples, p=probs)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "categories": self.categories,
            "mean": self.mean,
            "std_dev": self.std_dev}


@dataclass
class CategoricalUniformDistribution(CategoricalConversionMixin, Distribution):
    """ Uniform distribution for categorical variables 
    This distribution is defined by a set of categories and samples are drawn uniformly from these 
    categories.
    Attributes:
        categories (Tuple[str, ...]): Categories for the categorical variable.
        nominal_as_index (bool): If True, nominal_value is the middle category index, else it's the middle category itself.
        nominal_category (Optional[str]): If provided, overrides nominal_as_index to use this category as the nominal value.
    """
    _type: ClassVar[DistributionType] = DistributionType.CATEGORICALUNIFORM
    categories: Tuple[str, ...] = field(default_factory=tuple)
    nominal_as_index: bool = True  # If True, nominal_value is the middle category index, else it's the middle category itself
    nominal_category: Optional[str] = None  # If provided, overrides nominal_as_index
    
    def __post_init__(self) -> None:
        if not self.categories:
            raise ValueError(
                "Categories must not be empty for categorical uniform distribution")
        if self.nominal_category and self.nominal_category not in self.categories:
            raise ValueError(f"Nominal category '{self.nominal_category}' must be one of the categories: {self.categories}")
        # Call the mixin's __post_init__ to initialize _cat_to_num and _num_to_cat
        CategoricalConversionMixin.__post_init__(self)
        
    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> str:
        """ Return the middle category as nominal value """
        if self.nominal_category:
            index = self._cat_to_num.get(self.nominal_category, None)
            if index is None:
                raise ValueError(f"Nominal category '{self.nominal_category}' not found in categories: {self.categories}")
        else:
            index = len(self.categories) // 2
        if self.nominal_as_index:
            return index
        return self.categories[index]

    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        return CategoricalUniformDistribution(
            categories = (self.nominal_value, )
        )  # Standard deviation is set to 0 for a fixed value

    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the discrete uniform distribution """
        sample = np.random.choice(self.categories, size=n_samples)
        print(sample)
        return sample

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "categories": self.categories}


@dataclass
class CategoricalHistogramDistribution(CategoricalConversionMixin, Distribution):
    """ Categorical histogram distribution for categorical variables 
    """
    _type: ClassVar[DistributionType] = DistributionType.CATEGORICALHISTOGRAM
    categories: Tuple[str, ...] = field(default_factory=tuple)
    probabilities: Tuple[float, ...] = field(default_factory=tuple)
    nominal_as_index: bool = True  # If True, nominal_value is the middle category index, else it's the middle category itself
    nominal_category: Optional[str] = None  # If provided, overrides nominal_as_index
    
    def __post_init__(self) -> None:
        """ Ensure categories and probabilities are valid for categorical histogram distribution """
        if not self.categories:
            raise ValueError("Categories must not be empty for categorical histogram distribution")
        if len(self.categories) != len(self.probabilities):
            raise ValueError("Categories and probabilities must have the same length")
        if self.nominal_category and self.nominal_category not in self.categories:
            raise ValueError(f"Nominal category '{self.nominal_category}' must be one of the categories: {self.categories}")
        prob_sum = sum(self.probabilities)
        if prob_sum <= 0:
            raise ValueError("Probabilities must sum to a positive value")
        if not np.isclose(sum(self.probabilities), 1.0):
            self.probabilities = tuple(p / prob_sum for p in self.probabilities)
        self.categories = tuple(self.categories)
        self.probabilities = tuple(self.probabilities)
        # Call the mixin's __post_init__ to initialize _cat_to_num and _num_to_cat
        CategoricalConversionMixin.__post_init__(self)

    @property
    def type(self) -> str:
        return self._type

    @property
    def nominal_value(self) -> str:
        """ Returns the category with the highest probability """
        if self.nominal_category:
            index = self._cat_to_num.get(self.nominal_category, None)
            if index is None:
                raise ValueError(f"Nominal category '{self.nominal_category}' not found in categories: {self.categories}")
        else:
            index = np.argmax(self.probabilities)
        if self.nominal_as_index:
            return index
        return self.categories[index]

    def collapse(self) -> Distribution:
        """ Collapse the distribution to a fixed value """
        return CategoricalHistogramDistribution(
            categories = (self.nominal_value, ),
            probabilities = (1, )
        )  # Standard deviation is set to 0 for a fixed value

    def sample(self, n_samples: int) -> NDArray:
        """ Generate N samples from the categorical histogram distribution """
        return np.random.choice(
            self.categories,
            size=n_samples,
            p=self.probabilities)

    def to_dict(self) -> dict:
        """ Convert the distribution to a dictionary representation """
        return {
            "type": self.type.value,
            "nominal_value": self.nominal_value,
            "categories": self.categories,
            "probabilities": self.probabilities}


DISTRIBUTION_CLASS_MAP = {
    "real-normal": NormalDistribution,
    "real-uniform": UniformDistribution,
    "discrete-normal": DiscreteNormalDistribution,
    "discrete-uniform": DiscreteUniformDistribution,
    "discrete-histogram": DiscreteHistogramDistribution,
    "categorical-normal": CategoricalNormalDistribution,
    "categorical-uniform": CategoricalUniformDistribution,
    "categorical-histogram": CategoricalHistogramDistribution,
}


@dataclass
class Variable:
    """ Represents a variable with a distribution, name, and optional metadata.
    This class is used to define model parameters and input parameters used in
    sensitivity analysis.
    
    Attributes:
        name (str): Unique name of the variable.
        distribution (Distribution): Probability distribution associated with the variable.
        group (Optional[str]): Optional logical group (e.g. "inputs", "parameters").
        metadata (dict[str, Any]): Arbitrary extra metadata for downstream use.
    """
    name: str
    distribution: Distribution
    fixed: bool = False  # Indicates if the variable is fixed (not sampled)
    group: Optional[str] = None # Used for grouping parameters into categories (e.g. inputs, parameters, etc.)
    metadata: dict[str, Any] = field(default_factory=dict) # Additional metadata for the variable

    def __post_init__(self) -> None:
        assert isinstance(self.distribution, Distribution), (
            f"`distribution` must be an instance of Distribution or its subclass, got {type(self.distribution)}"
        )

    def is_categorical(self) -> bool:
        """ Check if the variable is categorical based on its distribution """
        return self.distribution.is_categorical()
    
    def cat_to_num(self, cat_values: Sequence[str]) -> Sequence[int]:
        """Convert categories to numbers if supported by the distribution, else return input.
        Catches NotImplementedError if the distribution does not support this conversion.
        """
        try:
            return self.distribution.cat_to_num(cat_values)
        except (AttributeError, NotImplementedError):
            return cat_values

    def num_to_cat(self, num_values: Sequence[int]) -> Sequence[str]:
        """Convert numbers to categories if supported by the distribution, else return input.
        Catches NotImplementedError if the distribution does not support this conversion.
        """
        try:
            return self.distribution.num_to_cat(num_values)
        except (AttributeError, NotImplementedError):
            return num_values
    
    def is_discrete(self) -> bool:
        """ Check if the variable's distribution is discrete """
        return self.distribution.is_discrete()

    def is_uniform(self) -> bool:
        """ Check if the variable's distribution is uniform or discrete uniform """
        return self.distribution.is_uniform()
    
    def is_fixed(self) -> bool:
        """ Check if the variable is fixed (not sampled) """
        return self.fixed

    def to_dict(self) -> dict:
        """ Present the variable to a dictionary representation."""
        base = {
            "name": self.name,
            "is_categorical": self.is_categorical(),
            "is_discrete": self.is_discrete(),
            "is_uniform": self.is_uniform(),
            "is_fixed": self.is_fixed(),
            "distribution": self.distribution.to_dict(),
            "group": self.group,
            "metadata": self.metadata
        }
        #base.update(self.distribution.to_dict())
        return base


if __name__ == "__main__":

    # a “bounds” input variable:
    iv_bounds = Variable(
        name="temperature",
        distribution=UniformDistribution(bounds=(273.15, 313.15))
    )

    rprint(iv_bounds.to_dict())
    # {'name': 'temperature', 'is_categorical': False, 'is_uniform': True,
    #  'group': None, 'metadata': None, 'type': 'real-uniform',
    #  'nominal_value': 293.15, 'bounds': (273.15, 313.15)}
