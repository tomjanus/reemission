"""Reservoir calculations."""
import logging
import math
import inspect
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, TypeVar, Type
from reemission.temperature import MonthlyTemperature
from reemission.auxiliary import (
    water_density, cd_factor, scale_windspeed, air_density)
from reemission.constants import Landuse, TrophicStatus
from reemission.exceptions import (
    WrongAreaFractionsException,
    WrongSumOfAreasException)

# Set up module logger
log = logging.getLogger(__name__)
# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = 0.01

ReservoirType = TypeVar('ReservoirType', bound='Reservoir')


@dataclass
class Reservoir:
    """Model of a generic reservoir.

    Atttributes:
        coordnates: Reservoir's (latitude, longitude) in deg
        temperature: MonthlyTemperature object with 12x1 monthly average
            temperature vector
        volume: Reservoir volume, m3
        max_depth: Maximum reservoir depth, m
        mean_depth: Mean reservoir depth, m
        inflow_rate: flow of water entering the reservoir, m3/year
        area: Inundated area of the reservoir, km2
        soil_carbon: Mass of C in inundated area, kg/m2
        area_fractions: List of fractions of inundated area allocated
            to specific landuse types given in Landue Enum type, -
        mean_radiance: Mean monthly horizontal radiance in a year, kWh/m2/d
        mean_radiance_may_sept: Mean monthly horizontal radiance between May
            and September, kWh/m2/d
        mean_radiance_nov_mar: Mean monthly horizontal radiance between
            November and March, kWh/m2/d
        mean_monthly_windspeed: Mean monthly wind speed in a year, m/s
        water_intake_depth: Water intake depth below surface, m

    Notes:
        Sum of area fractions should be equal 1 +/- EPS
        reservoir volume in G-Res is given in km3
        Water intake depth is used for estimating CH4 emissions via degassing.
    """
    coordinates: Tuple[float, float]
    temperature: MonthlyTemperature
    volume: float
    max_depth: float
    mean_depth: float
    inflow_rate: float
    area: float
    soil_carbon: float
    area_fractions: List[float]
    mean_radiance: float
    mean_radiance_may_sept: float
    mean_radiance_nov_mar: float
    mean_monthly_windspeed: float
    water_intake_depth: float

    def __post_init__(self):
        """Check if the provided list of landuse fractions has the same
        length as the list of landuses.

        Raises:
            WrongAreaFractionsException if number of area fractions in the list
                not equal to the number of land uses.
            WrongSumAreasException if area fractions do not sum to 1 +/-
                acurracy coefficient EPS.
        """
        try:
            assert len(self.area_fractions) == len(Landuse)
        except AssertionError as err:
            raise WrongAreaFractionsException(
                number_of_fractions=len(self.area_fractions),
                number_of_landuses=len(Landuse)) from err
        try:
            assert 1 - EPS <= sum(self.area_fractions) <= 1 + EPS
        except AssertionError as err:
            raise WrongSumOfAreasException(
                fractions=self.area_fractions,
                accuracy=EPS) from err
        if isinstance(self.coordinates, list):
            self.coordinates = tuple(self.coordinates)
        # Validate input arguments
        self.validate_attributes()

    @classmethod
    def from_dict(cls: Type[ReservoirType], parameters: dict,
                  **kwargs) -> ReservoirType:
        """Initializes the class from a dictionary. Skips keys that are not
        featured as class's attribiutes."""
        return cls(**{
            k: v for k, v in parameters.items()
            if k in inspect.signature(cls).parameters}, **kwargs)

    @property
    def latitude(self) -> float:
        """Obtains latitude from coordinates."""
        return self.coordinates[0]

    @property
    def longitude(self) -> float:
        """Obtains longitude from coordinates."""
        return self.coordinates[1]

    @property
    def residence_time(self) -> float:
        """Calculate water residence time from inflow rate and reservoir
        volume where inflow rate is calculated on a catchment level.
        Residence time (or Water Residence Time, WRT) represents the average
        amount of time that a molecule of water spends in a reservoir or lake.

        With assumed units of m3 for volume and m3/year for inflow_rate
        the residence time is given in years."""
        return self.volume / self.inflow_rate

    @property
    def discharge(self) -> float:
        """Return discharge from the reservervoir.

        Over a long enough time-scale in comparison to residence time,
        discharge is equal to the inflow rate. Assumed unit: m3/year."""
        return self.inflow_rate

    @property
    def retention_coeff_emp(self) -> float:
        """Empirical retention coefficient (-) for solutes from
        regression with residence time in years being a regressor."""
        return 1.0 - 1.0 / (1.0 + (0.801 * self.residence_time))

    @property
    def retention_coeff_larsen(self) -> float:
        """Retention coefficient (-) using the model of Larsen and Mercier,
        (1976) for Phosphorus retention. Assumes residence time in years.
        """
        return 1.0 / (1.0 + 1.0 / math.sqrt(self.residence_time))

    @property
    def volume_km3(self) -> float:
        """Return volume in km3. Original value is in m3."""
        return self.volume / 10**9

    def validate_attributes(self) -> None:
        """Check object attributes and log and correct, if necessary, the
        suspicious or invalid data."""
        if self.water_intake_depth == "null":
            # If water intake depth value is not given, assume that
            # the intake is from deep in the reservoir and therefore,
            # degassing occurs.
            self.water_intake_depth = self.max_depth
        if self.water_intake_depth > self.max_depth:
            log.warning("Water intake depth greater than max depth")
            log.warning("Setting intake depth to max depth.")
            self.water_intake_depth = self.max_depth

    def mean_radiance_lat(self) -> float:
        """Selects representative mean horizontal radiance in (kWh/m2/d)
        for a reservoir depending on its latitude."""
        if 40 > self.latitude > -40:
            radiance = self.mean_radiance
        if 40 < self.latitude:
            radiance = self.mean_radiance_may_sept
        if -40 > self.latitude:
            radiance = self.mean_radiance_nov_mar
        return radiance

    def global_radiance(self, period: str = "d") -> float:
        """Calculates reservoir cumulative global horizontal radiance in
        (kWh/m2/period) depending on reservoir's latitude.
        Eq. A.29. from Praire2021.

        Note:
            Multiplier of 30.4 applied in the version of this equation
            published in G-Res technical documentation but not in Praire2021.
            This multiplier converts the unit of radiance from kWh/m2/day to
            kWh/m2/month. However, this results in very high CH4 emission
            estimates. Hence, set to 1.0."""
        number_months_above_0 = self.temperature.number_months_above(
            threshold=0)
        if period.lower() in ("d", "day"):
            multiplier = 1.0
        if period.lower() in ("m", "month"):
            multiplier = 30.4
        else:
            multiplier = 1.0
        return self.mean_radiance_lat() * number_months_above_0 * multiplier

    def compare_mean_depth(self) -> float:
        """Compares mean depth of the reservoir against the calculated mean
        depth from volume and area. Return relative difference between
        data and the calculate value, (%). Eqs. A.8./A.9./A.10. in
        Praire2021.
        """
        mean_depth_data = self.mean_depth
        mean_depth_vol = self.volume_km3 / self.area * 1000
        return 100 * (mean_depth_data - mean_depth_vol) / mean_depth_data

    def check_geometry(self, error_margin: float = 5) -> bool:
        """Checks if the mean depth in the data is near the value obtained
        from volume and surface area. If the difference is less than the
        error margin, returns True, otherwise returns False."""
        return bool(self.compare_mean_depth() < error_margin)

    def q_bath_shape(self) -> float:
        r"""Calculate q-bathymetric shape. EQ. A21 in Praire2021.

        .. math::
            s_{q,bath} = \frac{h_{max}}{h_{mean}} - 1

        where:
            :math: s_{q,bath}, h_{max}, h_{mean}
            are, respectively: q-bathymetric shape, maximum reservoir depth,
            mean reservoir depth.
        """
        return self.max_depth / self.mean_depth - 1.0

    def littoral_area_frac(self) -> float:
        r"""Calculate percentage of reservoir's surface area that is
        littoral, i.e. close to the shore. Eq. A22 in Praire2021.
        Return:
            Littoral area fraction in \%

        .. math::
            f_{lit} = 100 \left(1-\left(1-3/h_{max}\right)^{s_{q,bath}}\right)

        where:
            :math: f_{lit}, h_{max}, s_{q,bath}
            are, respectively: littoral area fraction (\%), maximum reservoir
            depth, (m), and q-bathymetric shape, (-).
        """
        return 100.0 * (1 - (1 - 3.0 / self.max_depth) ** self.q_bath_shape())

    def bottom_temperature(self) -> float:
        """Calculates bottom (hypolimnion) temperature in the reservoir from
        a 12x1 profile of monthly average air temperatures.
        Equation A.2. in Praire2021.

        Note:
            Mean Temperature of the Colder Month in Praire2021 was interpreted
                as the coldest mean monthly temperature in the 12x1 temperature
                profile.
        """
        if self.temperature.coldest > 1.4:
            hypolimnion_temp = (0.6565 * self.temperature.coldest) + 10.7
        else:
            hypolimnion_temp = (0.2345 * self.temperature.coldest) + 10.11
        return hypolimnion_temp

    def surface_temperature(self) -> float:
        """Calculates surface/epilimnion temperature as the mean temperature of
        the 4 warmest months in a year. Equation A.4. in Praire2021.
        """
        return self.temperature.mean_warmest(number_of_months=4)

    def bottom_density(self) -> float:
        """Calculates water density at the bottom of the reservoir.
        Equation A.3. in Praire2021.
        """
        return water_density(temp=self.bottom_temperature())

    def surface_density(self) -> float:
        """Calculates water density at the surface of the reservoir.
        Equation A.4. in Praire2021.
        """
        return water_density(temp=self.surface_temperature())

    def thermocline_depth(self, wind_speed: Optional[float] = None,
                          wind_height: float = 50) -> float:
        """Calculate thermocline depth required for the calculation of CH4
        degassing.

        Assumes that the surface water temperature is equal to
        the mean monthly air temperature from 4 warmest months in the year

        Follows the equation in Gorham and Boyce (1989)
        https://www.sciencedirect.com/science/article/pii/S0380133089714799
        `Influence of Lake Surface Area and Depth Upon Thermal Stratification
        and the Depth of the Summer Thermocline`, Gorham, Eville and Boyce,
        Farrell M. Journal of Great Lakes ResearchOpen AccessVolume 15,
        Issue 2, Pages 233 - 245, 1989.

        If wind_speed or monthly_temp is not supplied, a simplified equation
        from Hanna 1990 is used, as implemented in G-Res.

        Args:
            wind_speed: average annual wind speed at the location of the
                reservoir, (m/s).
            wind_height: height at which wind_speed is measured, (m)
        """
        if wind_speed is None:
            thermocline_depth = 10**(0.185 * math.log10(self.area) + 0.842)
            log.debug("Thermocline depth calculated with model of Hanna.")
        else:
            # Calculate CD coefficient and scale wind speed to 10m
            cd_coeff = cd_factor(wind_speed)
            wind_at_10m = scale_windspeed(wind_speed=wind_speed,
                                          wind_height=wind_height,
                                          new_height=10)
            # Calculate bottom and surface water densities.
            hypolimnion_density = self.bottom_density()
            epilimnion_density = self.surface_density()
            # Find thermocline depth in metres
            aux_var_1 = cd_coeff * air_density(
                self.temperature.mean_warmest(number_of_months=4)) * \
                wind_at_10m
            aux_var_2 = 9.80665 * (hypolimnion_density - epilimnion_density)
            aux_var_3 = math.sqrt(self.area) * 10**6
            thermocline_depth = 2 * math.sqrt(aux_var_1 / aux_var_2) * \
                math.sqrt(aux_var_3)
            log.debug(
                "Thermocline depth calculated with model of Gorham and Boyce.")
        return thermocline_depth

    def _k600_ch4(self, wind_speed: float, wind_height: float) -> float:
        """Estimates gas transfer velocity, k600 in cm/h for CH4 from wind
        speed in m/s and reservoir area in km2 and returns the estimate in m/d.
        Table 2, p. 1760, Model B, Vachon adn Praire, 2013
        https://cdnsciencepub.com/doi/10.1139/cjfas-2013-0241.
        Eq. A.16. in Praire2021.
        """
        if wind_height == 10:
            wind_at_10m = wind_speed
        else:
            wind_at_10m = scale_windspeed(wind_speed=wind_speed,
                                          wind_height=wind_height,
                                          new_height=10)
        # k600 in cm/h
        k600 = 2.51 + 1.48 * wind_at_10m + 0.39 * wind_at_10m * \
            math.log10(self.area)
        # return k600 in m/d
        return k600 * 0.24

    def _kh_ch4(self) -> float:
        """Calculate kh coefficient for CH4, Handbook of Physics and Chemistry,
        Lide, 1994. CRC Press, Boca Raton, USA.
        Unit derived through back-calculation: kgCH4/(m3*atm)
        Eq. A.17. in Praire2021.
        """
        # Calculate effective temperature in deg C for methane
        eff_temp = self.temperature.eff_temp(gas="ch4")
        eff_temp_scaled = (eff_temp + 273.15) / 100
        molecular_weight_ch4 = 18.0153
        aux_1 = -115.6477 - 6.1698 * eff_temp_scaled
        aux_2 = 155.5756 / eff_temp_scaled
        aux_3 = 65.2553 * math.log(eff_temp_scaled)
        kh_ch4 = 1000.0/molecular_weight_ch4 * math.exp(aux_1 + aux_2 + aux_3)
        return kh_ch4

    def _p_ch4(self) -> float:
        r"""
        Calculate partial pressure of CH4 in microatm (\mu atm)
        Rasillo 2014, https://doi.org/10.1111/gcb.12741
        Terhi Rasilo, Yves T. Prairie, Paul A. del Giorgio,
        Global Change Biology,
        `Large-scale patterns in summer diffusive CH4 fluxes across boreal
        lakes, and contribution to diffusive C emissions.`
        Eq. A.18. in Praire2021.
        """
        aux_1 = 1.46 + 0.03 * self.temperature.eff_temp(gas="ch4")
        aux_2 = - 0.29 * math.log10(self.area)
        return 10 ** (aux_1 + aux_2)

    def surface_ch4_conc(self) -> float:
        """Calcualate surface water CH4 concentration in mgCH4/m3 (mu g/L)
        Eq. A.19. in Praire2021.
        """
        return self._kh_ch4() * self._p_ch4()

    def ch4_emission_factor(self, wind_speed: float,
                            wind_height: float) -> float:
        """Calculate CH4 emission factor the water body, (kg CH4/ha/yr)
        Eq. A.20. in Praire2021.
        """
        ch4_molar = 16  # g/mol
        em_factor = self.surface_ch4_conc() * \
            self._k600_ch4(wind_speed, wind_height) * ch4_molar * 365/100
        # 365 converts from /d to /yr
        # 1/100 converts from 1/km2 to 1/ha
        return em_factor

    def retention_coeff(self, method: str = 'larsen') -> float:
        """Return retention coefficient using the chosen calculation method.

        Args:
            method: Retention coefficient calculation method.
        """
        if method == 'larsen':
            return self.retention_coeff_larsen
        if method in ['emp', 'empirical']:
            return self.retention_coeff_emp
        # Otherwise, use the Larsen and Mercier model
        log.warning('Residence time calculation method %s unknown. ' +
                    'Using the Larsen and Mercier model', method)
        return self.retention_coeff_larsen

    def reservoir_tp(self, inflow_conc: float,
                     method: str = 'larsen') -> float:
        """Calculate reservoir TP concentration in micrograms/L.

        Args:
            inflow_conc: TP concentration in the inflow, micrograms/L.
            method: retention coefficient esimation method.
        """
        return inflow_conc * (1.0 - self.retention_coeff(method=method))

    def trophic_status(self, inflow_conc: float) -> Enum:
        """Return reservoirs trophic status depending on the influent
        TP concentration.

        Args:
            inflow_conc: TP concentration in the inflow, micrograms/L.
        """
        reservoir_tp = self.reservoir_tp(inflow_conc)
        if reservoir_tp < 10.0:
            return TrophicStatus.OLIGOTROPHIC
        if reservoir_tp < 30.0:
            return TrophicStatus.MESOTROPHIC
        if reservoir_tp < 100.0:
            return TrophicStatus.EUTROPHIC
        # If concentration >= 100.0
        return TrophicStatus.HYPER_EUTROPHIC
