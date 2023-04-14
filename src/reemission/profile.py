""" """
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


# Default month attached to each year value in the annual time-series data
TIME_FREQ = "AS"
# Default emission unit. Implement unit conversion later and allow addition
# / plotting of emissions with different units
EMISSION_UNIT = "tonneCO2/year"


DEFAULT_CONSTRUCTION_DATE = date.today()
DEFAULT_SIM_START_DATE = date.today()


class YearsNotEquallySpacedError(Exception):
    """Custom exception raised if years in the emission profile are not equally
    spaced."""


class AssetConstructionStage(Enum):
    """ """
    EXISTING = auto()
    FUTURE = auto()


@dataclass
class AssetSimDateManager:
    """Stores construction date and simulation date for asset simulation purposes
    e.g. for determining asset construction stage during simulation and asset
    age.
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
        """Conversion from year (integer) to datetime.date object"""
        return datetime.strptime(str(year), '%Y')

    def asset_status(self) -> AssetConstructionStage:
        """Return asset construction stage based on its age relative to 
        simulation start date"""
        if self.asset_age >= 0:
            return AssetConstructionStage.EXISTING
        return AssetConstructionStage.FUTURE

    @property
    def asset_age(self) -> int:
        """Return asset age in years"""
        return self.sim_start_date.year - self.construction_date.year


@dataclass
class EmissionQuantity:
    """Class for coupling information about emission magnitude and unit"""
    value: float
    unit: str
    _unit: str = field(init=False, repr=False, default='default')

    @property
    def unit(self) -> str:
        """ """
        return self._unit

    @unit.setter
    def unit(self, unit: str) -> None:
        """TODO: Enable automatic conversion of values on unit change"""
        self._unit = unit

    def convert_unit(self, new_unit: str) -> None:
        """Converts the emission quantity to a different unit
        Model: self.value = self.value * CONV_VALUE
               self.unit = new_unit
        """
        raise NotImplementedError("Unit conversion not yet implemented.")

    def __add__(self, emission2: EmissionQuantity) -> EmissionQuantity:
        """Method for adding two emission quantities.
        Currently supports single units but it is meant to expand to adding
        emissions with different units using unit conversion tools like pint
        """
        return EmissionQuantity(
            value=self.value + emission2.value, unit=self.unit)

    def __sub__(self, emission2: EmissionQuantity) -> EmissionQuantity:
        """Method for subtracting two emission quantities."""
        return EmissionQuantity(
            value=self.value - emission2.value, unit=self.unit)


@dataclass
class EmissionProfile(Iterator):
    """Class for storing emission profiles and converting them to Pandas series
    Attributes
        values: List of EmissionQuantity objects at time intervals (years) from
            initial state (year).
        years: List of integers representing years from initial year 0
    Methods
        __next__(): Returns the next element of the values list, or raises a 
            StopIteration error if the list is empty
        to_series(construction_year): Converts an emission profile (a list of 
            emission quantities) into a pandas Series with DateTimeIndex. Allows 
            subsequent addition of multiple profiles with different lengths 
            and starting/ending in different years.
    """
    values: Iterable[EmissionQuantity]
    years: Iterable[int]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.years):
            raise ValueError(
                "Emission and Year vectors have different lengths.")
        self._check_units()

    def _check_units(self) -> None:
        """Checks if the units in the emission list are the same for all 
        emissions. If not, raise value error.
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
        """Converts all emission quantities in the profile to a different unit"""
        for emission in self.values:
            emission.convert_unit(new_unit)

    def unit(self, check_consistency: bool = False) -> str:
        """Infer unit from the list of values"""
        if check_consistency:
            self._check_units()
        return self.values[0].unit

    def _is_equally_spaced(self, spacing: int = 1) -> bool:
        """Check if the years are equally spaced - required for conversion into
        EmissionProfileSeries that require an equally spaced grid - by default
        at spacing = 1 year."""
        return all(np.diff(self.years) == spacing)

    def interpolate(self, spacing: int = 1) -> EmissionProfile:
        """Construct emission profile with equal year spacing given in argument
        `spacing` (default = 1) spanning from the first to the last year in
        self.years"""
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
        """Plot the emission profile"""
        raise NotImplementedError(
            "Plotting EmissionProfile objects not yet implemented")

    def to_series(
            self, construction_year: int, 
            interpolate: bool = False) -> EmissionProfileSeries:
        """
        Conversion of emission profile (a list of emission quantities) into a
        pandas Series with DateTimeIndex.

        Allows subsequent addition of multiple profiles with different lengths
        and starting/ending in different years.

        Parameters
            construction_year : An integer representing the year in which the
                reservoir was created
        Return
            A pandas Series with the values and an index of datetimes.
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
    """Emission profile represented with Pandas series"""
    values: pd.Series[EmissionQuantity]
    unit: str

    def plot(self, title: str, marker: str = "o", linestyle: str = "-",
             linewidth: float = 1.0) -> None:
        """Plot emission data in the form of pandas time-series. """
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
    """Emission profile from multiple reservoirs.

    Enables addition of several profiles from different reservoirs constructed
    in different years. Enforces that each of the added emission profiles
    has to have the same length and unit.
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
        """Retrieve unit of the emission profile (combined)"""
        return self.profile.unit

    @staticmethod
    def _check_units(
            emission_profiles: Iterable[EmissionProfileSeries]) -> None:
        """Checks if the units of all emission profiles are the same. If not,
        raise value error.
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
        """Comnbine multiple emission profiles. These can start and end at
        different time"""
        em_series: List[pd.Series[EmissionQuantity]] = [
            emission.values for emission in profiles]
        inferred_unit: str = profiles[0].unit
        combined_profile = reduce(lambda l, r: l.add(
            r, fill_value=EmissionQuantity(0.0, inferred_unit)), em_series)
        return EmissionProfileSeries(
            values=combined_profile, unit=inferred_unit)

    def plot(self, title: str = "Combined emission profile", marker: str = "o",
             linestyle: str = "-", linewidth: float = 1.0) -> None:
        """Function for plotting combined (total) emission profiles"""
        self.profile.plot(title, marker, linestyle, linewidth)


@dataclass
class EmissionProfileCalculator:
    """Calculation of emission profiles using a profile function (a Callable)
    It is assumed that emission function for the years larger than time horizon
    plateus and becomes constant.
    """
    time_horizon: ClassVar[int] = 100
    profile_fun: Callable[[int], EmissionQuantity]

    def emission(self, no_years: int) -> EmissionQuantity:
        """Calculate emission for a given number of years elapsed from
        impoundment using the profile function in attributes.

        Parameters
            no_years: Number of years elapsed from impoundment
        """
        if no_years > self.time_horizon:
            return self.profile_fun(self.time_horizon)
        return self.profile_fun(no_years)

    def calculate(self, years: Iterable[int]) -> EmissionProfile:
        """Calculate emission profile for a number of years after impoundment.
        Parameters
            years(Itrerable[int]): An iterable collection of years.
        Return Value
            EmissionProfile: An emission profile object
        """
        years.sort()
        emissions = [self.emission(year) for year in years]
        return EmissionProfile(emissions, years)

    def integrate(
            self, start_year: int = 0, 
            end_year: Optional[int] = None) -> EmissionQuantity:
        """Performs numerical integration of emission profile using scipy's
        trapezoidal rule"""
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
    """Calculate total (integrated) emission using analytical integral function
    """
    time_horizon: ClassVar[int] = 100
    integral_fun: Callable[[int], EmissionQuantity]

    def calculate(
            self, start_year: int = 0, 
            end_year: Optional[int] = None) -> EmissionQuantity:
        """Calculate total emission analytically using integral_fun and (if
        integral_fun not present) - use numerical integration"""
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