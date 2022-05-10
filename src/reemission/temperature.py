""" Monthly temperature class describing temperature profile in the catchment
"""
import math
from dataclasses import dataclass
from typing import List
from numpy import mean


@dataclass
class MonthlyTemperature:
    """DataClass storing a monthly temperature profile"""

    temp_profile: List[float]

    def __post_init__(self):
        assert len(self.temp_profile) == 12

    def eff_temp(self, coeff: float = 0.052) -> float:
        """Calculate effective annual Temperature CO2 in deg C for the
        estimation of CO2 and CH4 diffusion)
        - Currently the coefficient used in CO2 calcs is 0.05 whilst for
          CH4 it is 0.052
        """
        return math.log10(mean([10 ** (temp * coeff) for temp in self.temp_profile])) / coeff

    @property
    def coldest(self):
        """Finds coldest monthly temperature within the monthly profile"""
        return min(self.temp_profile)

    def mean_warmest(self, number_of_months: int = 4) -> float:
        """Finds the mean temperature of the warmest n months
        By default the function calculates the mean of 4 warmest months"""
        sorted_temp_profile = self.temp_profile.copy()
        sorted_temp_profile.sort(reverse=True)
        return sum(sorted_temp_profile[:number_of_months]) / len(sorted_temp_profile[:number_of_months])
