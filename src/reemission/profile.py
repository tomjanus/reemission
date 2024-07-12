"""
Collection of classes and functions for managing and calculating emission profiles of
emissions from multiple reservoirs constructed at different dates.
"""
from __future__ import annotations
from enum import Enum, auto
from functools import reduce
from collections.abc import Iterator
from typing import List, Callable, Optional, ClassVar, Iterable
from dataclasses import dataclass, field, InitVar
from datetime import date
from datetime import datetime
import math
import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reemission.constants import AssetConstructionStage


# Default month attached to each year value in the annual time-series data
TIME_FREQ = "AS"
# Default emission unit. Implement unit conversion later and allow addition
# / plotting of emissions with different units
EMISSION_UNIT = "tonneCO2/year"


DEFAULT_CONSTRUCTION_DATE = date.today()
DEFAULT_SIM_START_DATE = date.today()


class YearsNotEquallySpacedError(Exception):
    """Custom exception raised if years in the emission profile are not equally spaced."""
    pass


@dataclass
class AssetSimDateManager:
    """Manages construction and simulation dates for asset simulations.

    Attributes:
        construction_date (datetime): The construction date of the asset.
        sim_start_date (datetime): The start date for the simulation.

    Methods:
        asset_status(): Returns the construction stage of the asset.
        asset_age(): Returns the age of the asset.
    """
    construction_date: datetime
    sim_start_date: datetime

    def __init__(self, construction_year: Optional[int] = None, 
                 sim_start_year: Optional[int] = None) -> None:
        if not construction_year:
            self.construction_date = DEFAULT_CONSTRUCTION_DATE
        else:
            self.construction_date = self._year_to_date(construction_year)
        if not sim_start_year:
            self.sim_start_date = DEFAULT_SIM_START_DATE
        else:
            self.sim_start_date = self._year_to_date(sim_start_year)

    @staticmethod
    def _year_to_date(year: int) -> datetime:
        """Converts a year (integer) to a datetime object."""
        return datetime.strptime(str(year), '%Y')

    def asset_status(self) -> AssetConstructionStage:
        """Determines the asset status (EXISTING/FUTURE) based on its age."""
        if self.asset_age >= 0:
            return AssetConstructionStage.EXISTING
        return AssetConstructionStage.FUTURE

    @property
    def asset_age(self) -> int:
        """Calculates the asset age in years."""
        return self.sim_start_date.year - self.construction_date.year


@dataclass
class EmissionQuantity:
    """Represents emission magnitude and unit.

    Attributes:
        value (float): The emission value.
        unit (str): The unit of the emission.

    Methods:
        convert_unit(new_unit): Converts the emission quantity to a different unit.
        __add__(emission2): Adds two emission quantities.
        __sub__(emission2): Subtracts two emission quantities.
    """
    value: float
    unit: str
    _unit: str = field(init=False, repr=False, default='default')

    @property
    def unit(self) -> str:
        """Returns the unit as a string."""
        return self._unit

    @unit.setter
    def unit(self, unit: str) -> None:
        """Sets the unit and handles unit conversion.
        
        Todo: 
            Enable automatic conversion of values on unit change
        """
        self._unit = unit

    def convert_unit(self, new_unit: str) -> None:
        """Converts the emission quantity to a different unit.
        
        Model: self.value = self.value * CONV_VALUE
            self.unit = new_unit
        Attention:
            Not implemented yet.
        """
        raise NotImplementedError("Unit conversion not yet implemented.")

    def __add__(self, emission2: EmissionQuantity) -> EmissionQuantity:
        """Adds two emission quantities.
        
        Note:
          Currently supports single units but it is meant to expand to adding
          emissions with different units using unit conversion tools like `pint`.
        """
        return EmissionQuantity(
            value=self.value + emission2.value, unit=self.unit)

    def __sub__(self, emission2: EmissionQuantity) -> EmissionQuantity:
        """Subtracts two emission quantities.
        
        Note:
          Currently supports single units but it is meant to expand to adding
          emissions with different units using unit conversion tools like `pint`.
        """
        return EmissionQuantity(
            value=self.value - emission2.value, unit=self.unit)


@dataclass
class EmissionProfile(Iterator):
    """Stores and manipulates emission profiles.

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
    values: Iterable[EmissionQuantity]
    years: Iterable[int]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.years):
            raise ValueError(
                "Emission and Year vectors have different lengths.")
        self._check_units()

    def _check_units(self) -> None:
        """Checks if all emission quantities have the same unit.
       
        Raises:
            ValueError: If emissions in the EmissionProfile do not have same units.
            
        Todo:
           In later implementations, attempt unit conversion before throwing error
        """
        for _ix in range(len(self.values) - 1):
            if self.values[_ix].unit != self.values[_ix + 1].unit:
                raise ValueError(
                    "Emissions in the EmissionProfile do not have same units.")

    def __next__(self):
        if not self.values:
            raise StopIteration
        return self.values.pop()

    def convert_unit(self, new_unit: str) -> None:
        """Converts all emission quantities in the profile to a different unit."""
        for emission in self.values:
            emission.convert_unit(new_unit)

    def unit(self, check_consistency: bool = False) -> str:
        """Returns the unit of the emission profile."""
        if check_consistency:
            self._check_units()
        return self.values[0].unit

    def _is_equally_spaced(self, spacing: int = 1) -> bool:
        """Checks if the years are equally spaced.
        
        Note:
          Required for conversion into EmissionProfileSeries that require an equally spaced grid.
          By default, spacing is at 1 year.
        """
        return all(np.diff(self.years) == spacing)

    def interpolate(self, spacing: int = 1) -> EmissionProfile:        
        """Construct emission profile with equal year spacing given in argument
        `spacing` (default = 1) spanning from the first to the last year in
        self.years
        """
        if self._is_equally_spaced(spacing):
            return self
        first_year, last_year = self.years[0], self.years[-1]
        interp_years = range(first_year, last_year+1, spacing)
        emission_values = [em.value for em in self.values]
        emission_unit = self.unit()
        interp_emissions = np.interp(interp_years, self.years, emission_values)
        interp_emission_quants = [EmissionQuantity(value, emission_unit) for
                                  value in interp_emissions]
        return EmissionProfile(
            values=interp_emission_quants, years=list(interp_years))

    def plot(self) -> None:
        """Plots the emission profile.
        
        Attention:
          Not implemented yet.
        """
        raise NotImplementedError(
            "Plotting EmissionProfile objects not yet implemented")

    def to_series(
            self, construction_year: int, 
            interpolate: bool = False) -> EmissionProfileSeries:
        """Converts the emission profile (a list of emission quantities) to a Pandas Series
        with DateTimeIndex.
        
        Hint:
            Allows subsequent addition of multiple profiles with different lengths
            and starting/ending in different years.

        Args:
            construction_year (int): The construction year of the reservoir.
            interpolate (bool): Whether to interpolate the profile if years are not equally spaced.
            
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
            unit=emission_profile.unit())


@dataclass
class EmissionProfileSeries:
    """Represents an emission profile as a Pandas Series.

    Attributes:
        values (pd.Series[EmissionQuantity]): The emission profile values.
        unit (str): The unit of the emission profile.

    Methods:
        plot(title, marker, linestyle, linewidth): Plots the emission profile.
    """
    values: pd.Series[EmissionQuantity]
    unit: str

    def plot(self, title: str, marker: str = "o", linestyle: str = "-",
             linewidth: float = 1.0) -> None:
        """Plots the emission profile as a Pandas time-series.

        Args:
            title (str): The title of the plot.
            marker (str): The marker style.
            linestyle (str): The line style.
            linewidth (float): The line width.
        """
        plot_font = {'family': 'serif', 'color':  'darkred', 'size': 16}
        # Reindex and add zero values in the missing fields
        idx = pd.date_range(
            self.values.index[0], self.values.index[-1], freq=TIME_FREQ)
        em_reindexed: pd.Series[EmissionQuantity] = self.values.reindex(
            idx, fill_value=EmissionQuantity(0.0, self.unit))
        em_values = em_reindexed.apply(lambda x: x.value)
        with plt.style.context('Solarize_Light2'):
            plt.title(title, fontdict=plot_font)
            plt.xlabel("Date", fontdict=plot_font)
            plt.ylabel(f"GHG emission, {self.unit}", fontdict=plot_font)
            plt.plot(
                em_values, marker=marker, linestyle=linestyle,
                linewidth=linewidth, color='#61370a')
            plt.show()


@dataclass
class CombinedEmissionProfile:
    """Represents a combined emission profile from multiple sources.
    
    Note:
        Enables addition of several profiles from different reservoirs constructed
        in different years. Enforces that each of the added emission profiles
        has to have the same length and unit.
    
    Attributes:
        emission_profiles (Iterable[EmissionProfileSeries]): The list of emission profiles to combine.
        profile (EmissionProfileSeries): The combined emission profile.

    Methods:
        unit(): Returns the unit of the combined emission profile.
        plot(title, marker, linestyle, linewidth): Plots the combined emission profile.
    """
    emission_profiles: InitVar[Iterable[EmissionProfileSeries]]
    profile: EmissionProfileSeries = field(init=False)

    def __post_init__(
            self,
            emission_profiles: Iterable[EmissionProfileSeries]) -> None:
        """ """
        # Check that all units in the emission profile list are consistent
        self._check_units(emission_profiles)
        # Combine emisison profiles in emission profile list
        self.profile = self._combine_profiles(emission_profiles)

    @property
    def unit(self) -> str:
        """Returns the unit of the combined emission profile."""
        return self.profile.unit

    @staticmethod
    def _check_units(
            emission_profiles: Iterable[EmissionProfileSeries]) -> None:
        """Checks if the units of all emission profiles are the same. 
        
        Raises:
            ValueError: If the units of all emission profiles are not the same.
        Todo:
            In later implementations, attempt unit conversion before throwing error
        """
        for _ix in range(len(emission_profiles) - 1):
            unit_curr = emission_profiles[_ix].unit
            unit_next = emission_profiles[_ix + 1].unit
            if unit_curr != unit_next:
                raise ValueError(
                    "Emission profiles have different units units: " +
                    f"{unit_curr}, {unit_next}.")

    @staticmethod
    def _combine_profiles(
            profiles: Iterable[EmissionProfileSeries]) -> EmissionProfileSeries:
        """Combines multiple emission profiles into one.
        
        Note:
            Profiles can start and end at different times (years).

        Args:
            profiles (Iterable[EmissionProfileSeries]): The list of emission profiles to combine.

        Returns:
            EmissionProfileSeries: The combined emission profile.
        """
        em_series: List[pd.Series[EmissionQuantity]] = [
            emission.values for emission in profiles]
        inferred_unit: str = profiles[0].unit
        combined_profile = reduce(lambda l, r: l.add(
            r, fill_value=EmissionQuantity(0.0, inferred_unit)), em_series)
        return EmissionProfileSeries(
            values=combined_profile, unit=inferred_unit)

    def plot(self, title: str = "Combined emission profile", marker: str = "o",
             linestyle: str = "-", linewidth: float = 1.0) -> None:
        """Plots the combined emission profile.

        Args:
            title (str): The title of the plot.
            marker (str): The marker style.
            linestyle (str): The line style.
            linewidth (float): The line width.
        """
        self.profile.plot(title, marker, linestyle, linewidth)


@dataclass
class EmissionProfileCalculator:
    """Calculates emission profiles using a provided profile function.
    
    Note:
        It is assumed that emission functions for years above certain time horizon 
        plateau and becomes constant.

    Attributes:
        time_horizon (ClassVar[int]): The time horizon for the emission profile.
        profile_fun (Callable[[int], EmissionQuantity]): The function to calculate emissions.

    Methods:
        emission(no_years): Calculates the emission for a given number of years.
        calculate(years): Calculates the emission profile for a range of years.
        integrate(start_year, end_year): Integrates the emission profile over a range of years.
    """
    time_horizon: ClassVar[int] = 100
    profile_fun: Callable[[int], EmissionQuantity]

    def emission(self, no_years: int) -> EmissionQuantity:
        """Calculates the emission for a given number of years elapsed from impoundment.

        Args:
            no_years (int): The number of years elapsed from impoundment.

        Returns:
            EmissionQuantity: The calculated emission.
        """
        if no_years > self.time_horizon:
            return self.profile_fun(self.time_horizon)
        return self.profile_fun(no_years)

    def calculate(self, years: Iterable[int]) -> EmissionProfile:
        """Calculates the emission profile for a range of years.

        Args:
            years (Iterable[int]): The range of years.

        Returns:
            EmissionProfile: The calculated emission profile.
        """
        years.sort()
        emissions = [self.emission(year) for year in years]
        return EmissionProfile(emissions, years)

    def integrate(
            self, start_year: int = 0, 
            end_year: Optional[int] = None) -> EmissionQuantity:
        """Integrates the emission profile over a range of years.
    
        Note:
            Uses scipy's trapezoidal rule.

        Args:
            start_year (int): The start year for the integration.
            end_year (Optional[int]): The end year for the integration.

        Returns:
            EmissionQuantity: The integrated emission quantity.
        """
        if not end_year:
            end_year = start_year + self.time_horizon
        if end_year <= start_year:
            raise ValueError("Start year needs to be smaller than end year")
        input_years = list(range(start_year, end_year+1))
        profile = self.calculate(input_years)
        integral_value = sc.integrate.trapezoid(
            [emission.value for emission in profile])
        return EmissionQuantity(integral_value, profile.values[0].unit)


@dataclass
class TotalEmissionCalculator:
    """Calculates total emissions using an analytical integral function.

    Attributes:
        time_horizon (ClassVar[int]): The time horizon for the emission profile.
        integral_fun (Callable[[int], EmissionQuantity]): The function to calculate total emissions.

    Methods:
        calculate(start_year, end_year): Calculates total emissions over a range of years.
    """
    time_horizon: ClassVar[int] = 100
    integral_fun: Callable[[int], EmissionQuantity]

    def calculate(
            self, start_year: int = 0, 
            end_year: Optional[int] = None) -> EmissionQuantity:
        """Calculates total emissions over a range of years.
        
        Note:
            If integral function *integral_fun* not given, use numerical integration.

        Args:
            start_year (int): The start year for the calculation.
            end_year (Optional[int]): The end year for the calculation.

        Returns:
            EmissionQuantity: The calculated total emission.
        """
        if not end_year:
            end_year = start_year + self.time_horizon
        if end_year <= start_year:
            raise ValueError("Start year needs to be smaller than end year")
        total_emission = self.integral_fun(end_year) - \
            self.integral_fun(start_year)
        return total_emission


if __name__ == "__main__":

    def test_one() -> None:
        """TEST 1: Test Asset Date Container functionality"""
        dates = AssetSimDateManager(2008, 2023)
        assert dates.asset_age > 0
        dates2 = AssetSimDateManager(2050, 2023)
        assert dates2.asset_age < 0

    def test_two() -> None:
        """TEST 2: Test Emission profile calculation, conversion to Pandas series
        profile addition and plotting"""
        years: List[int] = [1, 5, 20, 50, 100]

        # Profile 1
        def uniform_profile(year: int) -> EmissionQuantity:
            return EmissionQuantity(6.0, 'tonneCO2/m2/year')

        # Profile 2
        def exp_profile(year: int) -> EmissionQuantity:
            out = 10 + 100 * math.exp(-0.04*year)
            return EmissionQuantity(out, 'tonneCO2/m2/year')

        # Define profile calculator objects
        em_uniform_calculator = EmissionProfileCalculator(uniform_profile)
        em_exp_calculator = EmissionProfileCalculator(exp_profile)
        # Calculate profiles
        em_uniform = em_uniform_calculator.calculate(years)
        em_exp = em_exp_calculator.calculate(years)
        # Convert to pandas series
        pandas_prof_uniform = em_uniform.to_series(2000, interpolate=True)
        pandas_prof_exp = em_exp.to_series(2150, interpolate=True)
        # Combine emission profiles
        comb_profile = CombinedEmissionProfile(
            [pandas_prof_uniform, pandas_prof_exp])
        comb_profile.plot()

    # Run tests
    test_one()
    test_two()

    # TODO 1: Test unit conversion, unit checking, total emission calculation
    # TODO 2: Move tests to pytest
