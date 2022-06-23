"""Monthly temperature class describing temperature profile in a catchment.
"""
import logging
import math
from dataclasses import dataclass
from typing import List, ClassVar, Dict
from numpy import mean
from reemission.exceptions import GasNotRecognizedException

# Set up module logger
log = logging.getLogger(__name__)


@dataclass
class MonthlyTemperature:
    """Wrapper for monthly temperature profile in a catchment.

    Attributes:
        temp_profile: 12x1 list of mean monthly temperatures for a
            representative year, [deg C].
        eff_temp_coeffs: values of effective temperature coefficients per gas.
    """
    temp_profile: List[float]
    eff_temp_coeffs: ClassVar[Dict] = {
        'co2': 0.05,
        'ch4': 0.052}

    def __post_init__(self):
        try:
            temp_profile_length = len(self.temp_profile)
            assert temp_profile_length == 12
        except AssertionError:
            log.warning("Temperature vector is of length %d not equal 12.",
                        temp_profile_length)

    def _temp_corr_coeff_co2(self) -> List[float]:
        """Calculate temperature correction coefficient for CO2 for each
        month. Eq. A.13. in Praire2021.

        Returns:
            A list of temperature correction coefficients in deg C.

        Note:
            Temperatures lower than 4 degC are treated as 4 degC.
        """
        temp_coeff = self.eff_temp_coeffs['co2']
        return [10**(temp_coeff*temp) if temp >= 4 else
                10**(temp_coeff*4.0) for temp in self.temp_profile]

    def _temp_corr_coeff_ch4(self) -> List[float]:
        """Calculate temperature correction coefficient for CH4 for each
        month. Eq. A.12. in Praire2021.

        Returns:
            A list of temperature correction coefficients in deg C.

        Note:
            Temperatures lower than 4 degC are treated as 4 degC.
        """
        temp_coeff = self.eff_temp_coeffs['ch4']
        return [10**(temp_coeff*temp) if temp >= 4 else
                10**(temp_coeff*4.0) for temp in self.temp_profile]

    def eff_temp(self, gas: str) -> float:
        """Calculates effective annual temperature in deg C.
        Eqs. A.14. an A.15. in Praire2021.

        Args:
            gas: type of gas; currently available gasses are 'co2' and 'ch4'

        Raises:
            GasNotRecognizedException: An error raised when gas is not listed
                in recognized gas types.

        Note:
            Used for estimating CO2 and CH4 diffusion.
        """
        if gas == 'co2':
            corr_coeff = self._temp_corr_coeff_co2()
        elif gas == 'ch4':
            corr_coeff = self._temp_corr_coeff_ch4()
        else:
            raise GasNotRecognizedException(
                permitted_gases=('co2', 'ch4'))
        return math.log10(mean(corr_coeff)) / self.eff_temp_coeffs[gas]

    @property
    def coldest(self) -> float:
        """Finds coldest monthly temperature in the temperature profile."""
        return min(self.temp_profile)

    def mean_warmest(self, number_of_months: int = 4) -> float:
        """Returns the mean temperature of the warmest n months in a yearly
        12x1 temperature profile.

        By default the function calculates the mean of 4 warmest months.

        Args:
            number_of_months: number of warmest monthsin the profile.
        """
        sorted_temp_profile = self.temp_profile.copy()
        sorted_temp_profile.sort(reverse=True)
        return sum(sorted_temp_profile[:number_of_months]) / \
            len(sorted_temp_profile[:number_of_months])
