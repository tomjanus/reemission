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
        temp_profile (List[float]): 12x1 list of mean monthly temperatures for a representative year, deg C.
        eff_temp_coeffs (ClassVar[Dict[str, float]]): Values of effective temperature coefficients per gas.
    """
    temp_profile: List[float]
    eff_temp_coeffs: ClassVar[Dict] = {
        'co2': 0.05,
        'ch4': 0.052}

    def __post_init__(self) -> None:
        """Post-initialization to check if the temperature profile has 12 values."""
        try:
            temp_profile_length = len(self.temp_profile)
            assert temp_profile_length == 12
        except AssertionError:
            log.warning("Temperature vector is of length %d not equal 12.",
                        temp_profile_length)

    def _temp_corr_coeff_co2(self) -> List[float]:
        """Calculate temperature correction coefficient for CO$_2$ for each month (Eq. A.13 in Praire2021).

        Returns:
            List[float]: A list of temperature correction coefficients in deg C.

        Note:
            Temperatures lower than 4 deg C are treated as 4 deg C.
        """
        temp_coeff = self.eff_temp_coeffs['co2']
        corr = [10**(temp_coeff*temp) if temp >= 4 else
                10**(temp_coeff*4.0) for temp in self.temp_profile]
        return corr

    def _temp_corr_coeff_ch4(self) -> List[float]:
        """Calculate temperature correction coefficient for CH$_4$ for each month (Eq. A.12 in Praire2021).

        Returns:
            List[float]: A list of temperature correction coefficients in deg C.

        Note:
            Temperatures lower than 4 degC are treated as 4 degC.
        """
        temp_coeff = self.eff_temp_coeffs['ch4']
        corr = [10**(temp_coeff*temp) if temp >= 4 else
                10**(temp_coeff*4.0) for temp in self.temp_profile]
        return corr

    def eff_temp(self, gas: str) -> float:
        """Calculates effective annual temperature in deg C (Eqs. A.14 and A.15 in Praire2021).

        Args:
            gas (str): Type of gas; currently available gasses are 'co2' and 'ch4'.

        Raises:
            GasNotRecognizedException: If gas is not listed in recognized gas types.

        Note:
            Used for estimating CO$_2$ and CH$_4$ diffusion.

        Returns:
            float: Effective annual temperature in deg C.
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
        """Finds coldest monthly temperature in the temperature profile.

        Returns:
            float: The coldest monthly temperature.
        """
        return min(self.temp_profile)

    def mean_warmest(self, number_of_months: int = 4) -> float:
        """Returns the mean temperature of the warmest n months in a yearly 12x1 temperature profile.

        By default, the function calculates the mean of the 4 warmest months.

        Args:
            number_of_months (int, optional): Number of warmest months in the profile. Defaults to 4.

        Returns:
            float: Mean temperature of the warmest n months.
        """
        sorted_temp_profile = self.temp_profile.copy()
        sorted_temp_profile.sort(reverse=True)
        return sum(sorted_temp_profile[:number_of_months]) / \
            len(sorted_temp_profile[:number_of_months])

    def number_months_above(self, threshold: float) -> int:
        """Returns the number of months for which the temperature is above the threshold temperature.

        Args:
            threshold (float): Threshold temperature.

        Returns:
            int: Number of months with temperature above the threshold.
        """
        return len([i for i in self.temp_profile if i > threshold])
