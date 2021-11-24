""" Monthly temperature class describing temperature profile in the catchment
"""
import math
from dataclasses import dataclass
from typing import List
from numpy import mean


@dataclass
class MonthlyTemperature:
    """ DataClass storing a monthly temperature profile """
    temp_profile: List[float]

    def __post_init__(self):
        assert len(self.temp_profile) == 12

    def eff_temp(self, coeff: float = 0.052) -> float:
        """ Calculate effective annual Temperature CO2 in deg C for the
            estimation of CO2 and CH4 diffusion)
            - Currently the coefficient used in CO2 calcs is 0.05 whilst for CH4
              it is 0.052
        """
        return math.log10(
            mean([10**(temp * coeff) for temp in self.temp_profile]))/coeff
