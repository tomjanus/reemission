"""
Collection of classes and functions for managing and calculating emission profiles of
emissions from multiple reservoirs constructed at different dates.
"""
from __future__ import annotations
from functools import reduce
import warnings
from collections.abc import Iterator, Iterable
from typing import List, Callable, Optional
from dataclasses import dataclass, field, InitVar
import math
import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pint
from reemission.constants import AssetConstructionStage


TIME_FREQ = "YS" # Default month attached to each year value in annual time-series data
EMISSION_UNIT = "tonne_CO2e_per_year" # Default unit of emissions in the emission profile


# Set up a unit registry
ureg = pint.UnitRegistry()
ureg.define("g_CO2e = g")
ureg.define("kg_CO2e = kg")
ureg.define("tonne_CO2e = tonne")
ureg.define("kt_CO2e = 1000 * tonne_CO2e")
ureg.define("Mt_CO2e = 1000 * kt_CO2e")
ureg.define("tonne_CO2e_per_year = tonne_CO2e / year")
ureg.define("kg_CO2e_per_year = kg_CO2e / year")
ureg.define("kt_CO2e_per_year = kt_CO2e / year")
ureg.define("Mt_CO2e_per_year = Mt_CO2e / year")
ureg.define("g_CO2e_per_metre2_per_year = g_CO2e / metre**2 / year")


def pretty_unit(unit: str) -> str:
    """ Converts a unit string to a more human-readable format.
    Args:
        unit (str): The unit string to convert.
    Returns:
        str: The human-readable unit string.
    """
    if not isinstance(unit, str):
        raise TypeError("Unit must be a string.")
    if unit not in ureg:
        raise ValueError(f"Unit '{unit}' is not defined in the unit registry.")
    # Replace common unit patterns with more readable formats
    return (
        unit.replace("kg_CO2e_per_year", "kg_CO2e / year")
            .replace("tonne_CO2e_per_year", "tonne_CO2e / year")
            .replace("kt_CO2e_per_year", "kt_CO2e / year")
            .replace("_per_", " / ")
            .replace("_CO2e", " CO2e")
            .replace("_", " ")
    )


def calculate_age(construction_year: int, year: int) -> int:
    """Calculates the asset age in years."""
    return year - construction_year


def calculate_status(construction_year: int, year: int) -> AssetConstructionStage:
    """Determines the asset status (EXISTING/FUTURE) based on its age."""
    age = calculate_age(construction_year, year)
    if age >= 0:
        return AssetConstructionStage.EXISTING
    return AssetConstructionStage.FUTURE


class YearsNotEquallySpacedError(Exception):
    """Custom exception raised if years in the emission profile are not equally spaced."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"YearsNotEquallySpacedError: {self.message}"


@dataclass
class Emission:
    """Represents emission as emission value + unit.
    Note:
        The emission quantity is defined by a value and a unit.
        The unit can be changed using the `set_unit()` method.
        The class supports addition and subtraction of emission quantities,
        converting units if necessary.

    Attributes:
        value (float): The emission value.
        unit (str): The unit of the emission (read-only; use `set_unit()` to modify).

    Methods:
        set_unit(new_unit): Converts the emission quantity to a different unit.
        __add__(emission2): Adds two emission quantities.
        __sub__(emission2): Subtracts two emission quantities.
    """
    value: float
    unit: InitVar[str] = EMISSION_UNIT
    _unit: str = field(init=False)

    def __post_init__(self, unit):
        if unit not in ureg:
            raise ValueError(f"Unit '{unit}' is not defined in the unit registry.")
        self._unit = unit
        
    def copy(self) -> Emission:
        """Return a copy of this Emission object."""
        return Emission(self.value, self.unit)

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, value) -> None:
        raise AttributeError("Use `set_unit()` to change the unit.")

    def set_unit(self, new_unit: str) -> None:
        """Converts the emission quantity to a different unit.
        Args:
            new_unit (str): The new unit to convert to.
        Raises:
            ValueError: If the new unit is not recognized.
        """
        try:
            q = self.value * ureg(self._unit)
            q_converted = q.to(new_unit)
            self.value = q_converted.magnitude
            self._unit = new_unit
        except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
            warnings.warn(
                f"Unit '{new_unit}' not recognized or incompatible. No change made. ({e})",
                UserWarning)
            
    def __mul__(self, other):
        raise TypeError("Multiplication of Emission objects is not supported.")

    def __rmul__(self, other):
        raise TypeError("Multiplication of Emission objects is not supported.")

    def __truediv__(self, other):
        raise TypeError("Division of Emission objects is not supported.")

    def __rtruediv__(self, other):
        raise TypeError("Division of Emission objects is not supported.")

    def __add__(self, emission2: Emission) -> Emission:
        """Adds two emission quantities, converting units if necessary."""
        if self.unit == emission2.unit:
            return Emission(value=self.value + emission2.value, unit=self.unit)
        try:
            # Try to convert emission2 to self.unit
            q2 = emission2.value * ureg(emission2.unit)
            q2_converted = q2.to(self.unit)
            return Emission(value=self.value + q2_converted.magnitude, unit=self.unit)
        except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
            raise ValueError(
                f"Units do not match and cannot be converted: '{self.unit}' and '{emission2.unit}'. ({e})"
            ) from e

    def __sub__(self, emission2: Emission) -> Emission:
        """Subtracts two emission quantities, converting units if necessary."""
        if self.unit == emission2.unit:
            return Emission(value=self.value - emission2.value, unit=self.unit)
        else:
            try:
                # Try to convert emission2 to self.unit
                q2 = emission2.value * ureg(emission2.unit)
                q2_converted = q2.to(self.unit)
                return Emission(value=self.value - q2_converted.magnitude, unit=self.unit)
            except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
                raise ValueError(
                    f"Units do not match and cannot be converted: '{self.unit}' and '{emission2.unit}'. ({e})"
                ) from e


@dataclass
class EmissionProfile(Iterator):
    """Stores and manipulates emission profiles.
    Inherits from Iterator to allow iteration over emission quantities.
    
    Note:
        The emission profile is defined by a list of emission quantities and their corresponding years.
        The emission quantities must have the same unit.
        The years must be in ascending order.
    
    Raises:
        ValueError: If the lengths of the emission quantities and years do not match.
        ValueError: If the emission quantities do not have the same unit.

    Attributes:
        values (Iterable[EmissionQuantity]): List of emission quantities.
        years (Iterable[int]): List of corresponding years.

    Methods:
        __next__(): Returns the next emission quantity.
        convert_unit(new_unit): Converts all emission quantities to a new unit.
        unit(check_consistency): Returns the unit of the emission profile.
        _is_equally_spaced(spacing): Checks if the years are equally spaced.
        interpolate(spacing): Interpolates the emission profile to have equally spaced years.
        plot(): Plots the emission profile.
        to_series(construction_year, interpolate): Converts the emission profile to a Pandas Series.
    """
    values: Iterable[Emission]
    years: Iterable[int]
    _index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.values, Iterable):
            raise TypeError("Emission values must be an iterable.")
        if not isinstance(self.years, Iterable):
            raise TypeError("Years must be an iterable.")
        if not all(isinstance(em, Emission) for em in self.values):
            raise TypeError("All emission values must be of type Emission.")
        if not all(isinstance(year, (int, float)) for year in self.years):
            raise TypeError("All years must be integers or floats.")
        if len(self.values) != len(self.years):
            raise ValueError(
                "Emission and Year vectors have different lengths.")
        self._check_units()
        
    @classmethod
    def from_profile_function(
            cls,
            profile_fun: Callable[[int], Emission], 
            years: Iterable[int]) -> EmissionProfile:
        """Alternative constructor: Creates an EmissionProfile from a callable and a range of years.

        Args:
            profile_fun (Callable[[int], Emission]): The function to calculate emissions.
            years (Iterable[int]): The range of years.

        Returns:
            EmissionProfile: The calculated emission profile.
        """
        years = list(years)
        years.sort()
        emissions = [profile_fun(year) for year in years]
        return cls(emissions, years)
        
    def __iter__(self) -> EmissionProfile:
        self._index = 0  # Reset iteration
        return self
    
    def __next__(self):
        """ Returns the next emission quantity in the profile.
        Raises:
            StopIteration: If there are no more emission quantities to return.
        """
        if self._index >= len(self.values):
            raise StopIteration
        result = self.values[self._index]
        self._index += 1
        return result

    def _check_units(self) -> None:
        """Checks if all emission quantities have the same unit.
        If not, attempts to convert them to the first emission's unit.
       
        Raises:
            ValueError: If emissions in the EmissionProfile do not have same units
            and cannot be converted.
            
        Attempts:
            Converts mismatched units to the first emission's unit before raising error.
        """
        base_unit = self.values[0].unit
        for emission in self.values:
            if emission.unit != base_unit:
                try:
                    emission.set_unit(base_unit)
                except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
                    raise ValueError(
                        f"Emissions in the EmissionProfile have mismatched units "
                        f"and cannot be converted: '{emission.unit}' to '{base_unit}'. ({e})"
                    ) from e
                    
    def _calculate_spacing(self) -> int:
        """Calculates the spacing between years.
        
        Returns:
            int: The spacing between consecutive years.
        """
        year_differences = np.diff(self.years)
        if not all(year_differences == year_differences[0]):
            raise YearsNotEquallySpacedError("Years are not equally spaced.")
        return year_differences[0]

    def _is_equally_spaced(self) -> bool:
        """Checks if the years are equally spaced.
        
        Returns:
            bool: True if the years are equally spaced, False otherwise.
        """
        try:
            self._calculate_spacing()
            return True
        except YearsNotEquallySpacedError:
            return False

    def set_unit(self, new_unit: str) -> None:
        """Sets all emission quantities in the profile to a different unit."""
        for emission in self.values:
            emission.set_unit(new_unit)

    @property
    def unit(self) -> str:
        """Returns the unit of the emission profile."""
        self._check_units()
        return self.values[0].unit

    def interpolate(self, spacing: int = 1) -> EmissionProfile:
        """
        Creates a new emission profile with equally spaced years.

        Args:
            spacing (int): The interval between consecutive years in the new profile.
                           Default is 1.

        Returns:
            EmissionProfile: A new emission profile with equally spaced years.

        Notes:
            - If the current profile already has the specified spacing, it is returned as-is.
            - The interpolation is performed using linear interpolation.

        Raises:
            YearsNotEquallySpacedError:
                If the years in the current profile are not equally spaced.
        """
        try:    
            self._calculate_spacing()
        except YearsNotEquallySpacedError:
            pass
        else:
            return self
        first_year, last_year = self.years[0], self.years[-1]
        interp_years = list(range(first_year, last_year + 1, spacing))
        emission_values = [em.value for em in self.values]
        emission_unit = self.unit
        # Perform linear interpolation
        interp_emissions = np.interp(interp_years, self.years, emission_values)
        interp_emission_objects = [
            Emission(value, emission_unit) for value in interp_emissions
        ]
        return EmissionProfile(values=interp_emission_objects, years=interp_years)
    
    def integrate(
            self, 
            start_year: Optional[int] = None, 
            end_year: Optional[int] = None) -> Emission:
        """Integrates the emission profile over a range of years.
    
        Note:
            Uses scipy's trapezoidal rule.

        Args:
            start_year (Optional[int]): The start year for the integration.
            end_year (Optional[int]): The end year for the integration.

        Returns:
            EmissionQuantity: The integrated emission quantity.
        """
        if not end_year:
            end_year = self.years[-1]
        if not start_year:
            start_year = self.years[0]
        if start_year not in self.years:
            raise ValueError(f"Start year {start_year} not in emission profile years.")
        if end_year not in self.years:
            raise ValueError(f"End year {end_year} not in emission profile years.")
        if end_year <= start_year:
            raise ValueError("Start year needs to be smaller than end year")
        trimmed_years = [year for year in self.years if start_year <= year <= end_year]
        trimmed_values = [
            self.values[self.years.index(year)] for year in trimmed_years
        ]
        new_profile = EmissionProfile(values=trimmed_values, years=trimmed_years)
        integral_value = sc.integrate.trapezoid([emission.value for emission in new_profile])
        return Emission(integral_value, new_profile.values[0].unit)

    def plot(
            self, 
            title: str = "Emission Profile") -> None:
        """Plots the emission profile.

        Args:
            title (str): The title of the plot.

        Notes:
            - The x-axis represents the years.
            - The y-axis represents the emission values with appropriate units.
        """
        # Extract emission values and years
        years = self.years
        emission_values = [emission.value for emission in self.values]
        emission_unit = pretty_unit(self.unit)
        sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.3)
        line_color = sns.color_palette("colorblind")[0]  # Professional blue
        edge_color = "k" 
        marker_size = 100  # Size in points^2 for scatter markers
        title_fontsize = 20
        label_fontsize = 18
        tick_fontsize = 14
        plt.figure(figsize=(10, 6))
        plt.plot(
            years, emission_values,
            linestyle='--',
            linewidth=2.5,
            color=line_color,
            zorder=1  # Make sure line is behind markers
        )
        plt.scatter(
            years, emission_values,
            s=marker_size,
            color=line_color,
            edgecolor=edge_color,
            linewidth=1.2,
            zorder=2  # Markers on top
        )
        plt.title(title, fontsize=title_fontsize, fontweight="bold", pad=20)
        plt.xlabel("Year", fontsize=label_fontsize, labelpad=10)
        plt.ylabel(f"Emission ({emission_unit})", fontsize=label_fontsize, labelpad=10)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        plt.tight_layout()
        plt.show()

    def to_series(
            self,
            construction_year: int, 
            interpolate: bool = True) -> EmissionProfileSeries:
        """Converts the emission profile (a list of emission quantities) to a Pandas Series
        with DateTimeIndex.
        
        Hint:
            Allows subsequent addition of multiple profiles with different lengths
            and starting/ending in different years.

        Args:
            construction_year (int): The construction year of the reservoir.
            interpolate (bool): Whether to interpolate the profile if years are not equally 
                spaced. Default is True.
            
        Raises:
            YearsNotEquallySpacedError: If emissions in are not equally spaced.

        Returns:
            EmissionProfileSeries: The emission profile as a Pandas Series.
        """
        if not self._is_equally_spaced():
            if interpolate:
                emission_profile = self.interpolate(spacing=1)
            else:
                raise YearsNotEquallySpacedError(
                    "Years in the emission profile are not equally spaced." +
                    "Fix the spacing manually or using EmissionProfile.interpolate()."
                )
        else:
            emission_profile = self
        years_list = list(
            map(str, [year + construction_year for year in emission_profile.years]))
        return EmissionProfileSeries(
            values=pd.Series(
                emission_profile.values, index=pd.DatetimeIndex(data=years_list)),
            unit=emission_profile.unit)


@dataclass
class EmissionProfileSeries:
    """Represents an emission profile as a Pandas Series.

    Attributes:
        values (pd.Series[Emission]): The emission profile values.
        unit (str): The unit of the emission profile.

    Methods:
        plot(title, marker, linestyle, linewidth): Plots the emission profile.
        __add__: Adds two emission profiles.
        combine: Combines multiple emission profiles.
    """
    values: pd.Series  # type: pd.Series[Emission]
    unit: str

    def __post_init__(self):
        if self.unit not in ureg:
            raise ValueError(f"Unit '{self.unit}' is not defined in the unit registry.")
        if not isinstance(self.values.index, pd.DatetimeIndex):
            raise TypeError("Profile index must be a pd.DatetimeIndex")

    def plot(
            self, 
            title: str = "Combined Profile",
            linewidth: float = 1.0) -> None:
        """Plots the emission profile as a Pandas time-series.

        Args:
            title (str): The title of the plot.
            linewidth (float): The line width.
        """
        # Reindex and add zero values in the missing fields
        idx = pd.date_range(
            self.values.index[0], self.values.index[-1], freq=TIME_FREQ)
        em_reindexed: pd.Series[Emission] = self.values.reindex(
            idx, fill_value=Emission(0.0, self.unit))
        em_values = em_reindexed.apply(lambda x: x.value)
        sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.3)
        line_color = sns.color_palette("colorblind")[0]  # Professional blue
        edge_color = "k"
        marker_size = 100  # Size in points^2 for scatter markers
        title_fontsize = 20
        label_fontsize = 18
        tick_fontsize = 14
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.values.index, em_values,
            linestyle="--",
            linewidth=linewidth,
            color=line_color,
            zorder=1  # Make sure line is behind markers
        )
        plt.scatter(
            self.values.index, em_values,
            s=marker_size,
            color=line_color,
            edgecolor=edge_color,
            linewidth=1.2,
            zorder=2  # Markers on top
        )
        plt.title(title, fontsize=title_fontsize, fontweight="bold", pad=20)
        plt.xlabel("Date", fontsize=label_fontsize, labelpad=10)
        plt.ylabel(f"GHG emission ({pretty_unit(self.unit)})", fontsize=label_fontsize, labelpad=10)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        plt.tight_layout()
        plt.show()

    def __add__(self, other: EmissionProfileSeries) -> EmissionProfileSeries:
        """Adds two emission profiles, aligning them by index and checking unit consistency."""
        if self.unit != other.unit:
            try:
                other.values = other.values.apply(lambda x: x.set_unit(self.unit) or x)
            except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
                raise ValueError(
                    f"Cannot add profiles with different units: "
                    f"{self.unit} vs {other.unit}. ({e})") from e
        new_index = self.values.index.union(other.values.index)
        re_self = self.values.reindex(new_index, fill_value=Emission(0.0, self.unit))
        re_other = other.values.reindex(new_index, fill_value=Emission(0.0, self.unit))
        combined = re_self.add(re_other)
        return EmissionProfileSeries(values=combined, unit=self.unit)

    @classmethod
    def combine(cls, profiles: Iterable[EmissionProfileSeries]) -> EmissionProfileSeries:
        """Combines multiple emission profiles."""   
        em_series: List[pd.Series[Emission]] = [
            emission.values for emission in profiles]
        inferred_unit: str = profiles[0].unit
        combined_profile = reduce(lambda l, r: l.add(
            r, fill_value=Emission(0.0, inferred_unit)), em_series)
        return EmissionProfileSeries(
            values=combined_profile, unit=inferred_unit)

if __name__ == "__main__":

    def test_one() -> None:
        """TEST 1: Test Asset Date Container functionality"""
        assert calculate_age(construction_year=2000, year=2010) == 10
        assert calculate_age(construction_year=2020, year=2010) == -10

    def test_two() -> None:
        # Create two Emission objects with different units
        em1 = Emission(1.0, "tonne_CO2e_per_year")
        em1b = em1.copy()
        em1b.set_unit("kg_CO2e_per_year")  # 1 tonne = 1000 kg
        
        print(f"em1: {em1.value} {em1.unit}")  # Should print 1 tonne_CO2e_per_year
        print(f"em1b: {em1b.value} {em1b.unit}")
        
        em2 = Emission(500.0, "kg_CO2e_per_year")  # 500 kg = 0.5 tonne

        # Add emissions (em2 will be converted to em1's unit)
        em_sum = em1 + em2
        print(f"Sum: {em_sum.value} {em_sum.unit}")  # Should print 1.5 tonne_CO2_per_year

        # Subtract emissions (em2 will be converted to em1's unit)
        em_diff = em1 - em2
        print(f"Difference: {em_diff.value} {em_diff.unit}")  # Should print 0.5 tonne_CO2_per_year

        # Try with incompatible units (should raise ValueError)
        try:
            em3 = Emission(1.0, "tonne_CO2e_per_year")
            em4 = Emission(1.0, "kg_CO2e")  # No per year
            em_bad = em3 + em4
        except ValueError as e:
            print(f"Error: {e}")

    def test_three() -> None:
        """TEST 2: Test Emission profile calculation, conversion to Pandas series
        profile addition and plotting"""
        # Profile 1
        def uniform_profile(year: int) -> Emission:
            return Emission(6.0, 'tonne_CO2e_per_year')
        # Profile 2
        def exp_profile(year: int) -> Emission:
            out = 10 + 100 * math.exp(-0.04*year)
            return Emission(out, 'tonne_CO2e_per_year')
        years_equal = range(1, 100, 1)
        years_unequal: List[int] = [1, 5, 20, 50, 100]
        em_uniform = EmissionProfile.from_profile_function(
            uniform_profile, years_equal)
        em_exp = EmissionProfile.from_profile_function(exp_profile, years_unequal)
        em_exp.plot()
        # Convert to pandas series
        pandas_prof_uniform = em_uniform.to_series(2000, interpolate=True)
        pandas_prof_exp = em_exp.to_series(2050, interpolate=True)
        comb_profile = EmissionProfileSeries.combine([pandas_prof_uniform, pandas_prof_exp])
        comb_profile.plot()

    # Run tests
    test_one()
    test_two()
    test_three()

    # TODO 1: Move tests to pytest
