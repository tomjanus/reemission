"""Reservoir class for storing and validating reservoir parameters and calculation of reservoir characteristics and operational parameters from primary data.

.. _G-Res Technical Documentation: https://www.hydropower.org/publications/the-ghg-reservoir-tool-g-res-technical-documentation
.. _Praire2021: https://www.sciencedirect.com/science/article/pii/S1364815221001602
.. _G-Res: https://www.grestool.org/

"""
import logging
import math
import configparser
import inspect
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, TypeVar, Type
from reemission.temperature import MonthlyTemperature
from reemission.auxiliary import (
    water_density, cd_factor, scale_windspeed, air_density)
from reemission.utils import (
    read_config, save_return, get_package_file, load_yaml, debug_on_exception)
from reemission.constants import Landuse, TrophicStatus
from reemission.exceptions import (
    WrongAreaFractionsException,
    WrongSumOfAreasException)
from reemission.globals import internal


# Set up module logger
log = logging.getLogger(__name__)
config: configparser.ConfigParser = read_config(
    get_package_file("config/config.ini"))

internals_config = load_yaml(get_package_file("config/internal_vars.yaml"))

# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = config.getfloat("CALCULATIONS", "eps_reservoir_area_fractions")

ReservoirType = TypeVar('ReservoirType', bound='Reservoir')


@dataclass
class Reservoir:
    """Model of a generic reservoir.

    Attributes:
        coordinates (Tuple[float, float]): Reservoir's latitude and longitude, degrees.
        temperature (MonthlyTemperature): MonthlyTemperature object with a 12x1 monthly average temperature vector, degC.
        volume (float): Reservoir volume, m$^3$.
        max_depth (float): Maximum reservoir depth, m.
        mean_depth (float): Mean reservoir depth, m.
        inflow_rate (float): Flow of water entering the reservoir, m$^3$/year.
        area (float): Inundated area of the reservoir, km$^2$.
        soil_carbon (float): Mass of carbon in the inundated area, kg/m$^2$.
        area_fractions (List[float]): List of fractions of the inundated area allocated to specific landuse types given in Landuse Enum type per 3 soil type categories: mineral -> organic -> nodata.
        mean_radiance (float): Mean annual monthly horizontal radiance, kWh/m$^2$/d.
        mean_radiance_may_sept (float): Mean monthly horizontal radiance between May and September, kWh/m$^2$/d.
        mean_radiance_nov_mar (float): Mean monthly horizontal radiance between November and March, kWh/m$^2$/d.
        mean_monthly_windspeed (float): Mean annual monthly wind speed, m/s (**assumed to be measured at the height of 50 metres.**)
        water_intake_depth (float): Water intake depth below surface, m.
        name (str): Reservoir name. Default is "n/a".

    Note:
        * Sum of area fractions should be equal to 1 +/- EPS.
        * Reservoir volume in G-Res_ is given in km$^3$.
        * Water intake depth is used for estimating CH$_4$ emissions via degassing.
    """

    coordinates: Tuple[float, float]
    temperature: MonthlyTemperature
    volume: float
    max_depth: float
    mean_depth: float
    inflow_rate: float
    area: float
    soil_carbon: float
    area_fractions: List
    mean_radiance: float
    mean_radiance_may_sept: float
    mean_radiance_nov_mar: float
    mean_monthly_windspeed: float
    water_intake_depth: float
    name: str = "n/a"

    def __post_init__(self):
        """Post-initialization checks and validations.

        Validates the size and sum of area fractions and ensures coordinates are tuples.

        Raises:
            WrongAreaFractionsException: If the number of area fractions is not equal to the number of land uses times three.
            WrongSumOfAreasException: If the sum of area fractions does not equal 1 +/- EPS.
        """
        try:
            assert len(self.area_fractions) == 3 * len(Landuse)
        except AssertionError as err:
            message: str = \
                f"Wrong size of the reservoir {self.name} area fractions vector."
            raise WrongAreaFractionsException(
                number_of_fractions=len(self.area_fractions),
                number_of_landuses= 3 * len(Landuse),
                message=message) from err
        try:
            assert 1 - EPS <= sum(self.area_fractions) <= 1 + EPS
        except AssertionError as err:
            message: str = \
                f"Wrong values in reservoir {self.name} area fractions vector."
            raise WrongSumOfAreasException(
                fractions=self.area_fractions,
                accuracy=EPS,
                message=message) from err
        if isinstance(self.coordinates, list):
            self.coordinates = tuple(self.coordinates)
        # Validate input arguments
        self.validate_attributes()


    def validate_attributes(self) -> None:
        """Validate object attributes and correct if necessary."""
        if self.water_intake_depth == "null" or self.water_intake_depth is None:
            # If water intake depth value is not given, assume that
            # the intake is from deep in the reservoir and therefore,
            # degassing occurs.
            self.water_intake_depth = self.max_depth
        elif self.water_intake_depth > self.max_depth:
            log.warning(
                "Water intake depth in reservoir %s greater than max depth",
                self.name)
            log.warning("Setting intake depth to max depth.")
            self.water_intake_depth = self.max_depth

    @classmethod
    def from_dict(cls: Type[ReservoirType], parameters: dict,
                  **kwargs) -> ReservoirType:
        """Initialize the class from a dictionary.

        Skips keys that are not attributes of the class.

        Args:
            parameters (dict): Dictionary of parameters.

        Returns:
            ReservoirType: An instance of the Reservoir class.
        """
        return cls(**{
            k: v for k, v in parameters.items()
            if k in inspect.signature(cls).parameters}, **kwargs)

    @property
    def latitude(self) -> float:
        """Obtain latitude from coordinates.

        Returns:
            float: Latitude of the reservoir.
        """
        return self.coordinates[0]

    @property
    def longitude(self) -> float:
        """Obtain longitude from coordinates.

        Returns:
            float: Longitude of the reservoir.
        """
        return self.coordinates[1]

    @property
    def residence_time(self) -> float:
        """Calculate water residence time (WRT) from inflow rate and reservoir volume.

        Residence time represents the average amount of time that a molecule of water spends in a reservoir or lake.
        With assumed units of m$^3$ for `volume` and m$^3$/year for `inflow rate` the residence time is given in years.

        Returns:
            float: Water residence time, years.
        """
        return self.volume / self.inflow_rate

    @property
    def discharge(self) -> float:
        """Return discharge from the reservoir.

        Over a long enough timescale compared to residence time, discharge is equal to the inflow rate.

        Returns:
            float: Discharge, m$^3$/year.
        """
        return self.inflow_rate

    @property
    def retention_coeff_emp(self) -> float:
        r"""Empirical retention coefficient for solutes.

        Obtained from regression with residence time in years.
        
        .. math::
            f_R = 1 - \frac{1}{1+0.801\,WRT}

        Returns:
            float: Empirical retention coefficient, --
        """
        return 1.0 - 1.0 / (1.0 + (0.801 * self.residence_time))

    @property
    def retention_coeff_larsen(self) -> float:
        """Retention coefficient using the model of Larsen and Mercier (1976) for Phosphorus retention.

        Assumes residence time in years.

        Returns:
            float: Retention coefficient, --
        """
        return 1.0 / (1.0 + 1.0 / math.sqrt(self.residence_time))

    @property
    def volume_km3(self) -> float:
        """Return volume in km$^3$.

        Note:
           The attribute value is in m$^3$.

        Returns:
            float: Volume, km$^3$.
        """
        return self.volume / 10**9

    @property
    @save_return(internal, True)
    def wat_area_frac_pre(self) -> float:
        """Return the fraction of reservoir area that was water prior to impoundment.

        Returns:
            float: Fraction of the reservoir area that was water before impoundment.
        """
        return sum([self.area_fractions[i] for i in (3, 12, 21)])

    @save_return(internal, internals_config['mean_radiance_lat']['include'])
    def mean_radiance_lat(self) -> float:
        """Select representative mean horizontal radiance for a reservoir depending on its latitude.

        Returns:
            float: Mean horizontal radiance in kWh/m$^2$/d.
        """
        if 40 > self.latitude > -40:
            radiance = self.mean_radiance
        if 40 < self.latitude:
            radiance = self.mean_radiance_may_sept
        if -40 > self.latitude:
            radiance = self.mean_radiance_nov_mar
        return radiance

    @save_return(internal, internals_config['global_radiance']['include'])
    def global_radiance(self, period: str = "d") -> float:
        """Calculate cumulative global horizontal radiance as a function of reservoir's latitude.

        Eq. A.29. from Praire2021_.

        Args:
            period (str): Time period. Options are "d" for day and "m" for month.

        Returns:
            float: Global horizontal radiance, kWh/m$^2$/period.       
        
        Caution:
            The multiplier of 30.4 was included in the published in `G-Res Technical Documentation`_ but not in Praire2021_.
            This multiplier converts the unit of radiance from kWh/m$^2$/day to kWh/m$^2$/month. However, this results in very high CH$_4$ emission estimates. Hence, we set it to 1.0.
        """
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
        r"""Compare the mean depth of the reservoir against the calculated mean
        depth from volume and area.

        Returns the relative difference between the data and the calculated value in percents.
        Refer to Eqs. A.8./A.9./A.10. in Praire2021_.
        
        .. math::
           100 \, \frac{\bar{h}_{\textrm{from data}}-\bar{h}_{\textrm{from volume}}}{\bar{h}_{\textrm{from data}}}
        
        Returns:
            float: The relative difference between the data and the calculated mean depth, %.
        """
        mean_depth_data = self.mean_depth
        mean_depth_vol = self.volume_km3 / self.area * 1_000
        return 100 * (mean_depth_data - mean_depth_vol) / mean_depth_data

    def check_geometry(self, error_margin: float = 5) -> bool:
        """Check if the mean depth in the input data is near the value obtained
        from volume and surface area.

        If the difference is less than the error margin, returns True, otherwise returns False.
        
        Args:
            error_margin (float): The allowable error margin in percent. Default is 5.

        Returns:
            bool: True if the difference is less than the error margin, otherwise False.
        """
        return bool(self.compare_mean_depth() < error_margin)

    def q_bath_shape(self) -> float:
        r"""Calculate q-bathymetric shape. Refer to Eq. A21 in Praire2021_.

        .. math::
            s_{q,bath} = \frac{h_{max}}{h_{mean}} - 1

        where:
            :math:`s_{q,bath}`, :math:`h_{max}`, and :math:`h_{mean}`
            are, respectively, the q-bathymetric shape, maximum reservoir depth,
            and mean reservoir depth.

        Returns:
            float: The q-bathymetric shape.
        """
        return self.max_depth / self.mean_depth - 1.0

    @save_return(internal, internals_config['littoral_area_frac']['include'])
    def littoral_area_frac(self) -> float:
        r"""Calculate the percentage of the reservoir's surface area that is
        littoral (close to the shore). Refer to Eq. A22 in Praire2021_.

        Ensure that the return value is always greater than 0.

        Returns:
            float: Littoral area fraction in percent.

        .. math::
            f_{lit} = 100 \left(1-\left(1-3/h_{max}\right)^{s_{q,bath}}\right)

        where:
            :math:`f_{lit}`, :math:`h_{max}`, and :math:`s_{q,bath}`
            are, respectively, the littoral area fraction (%), maximum reservoir
            depth (m), and q-bathymetric shape.
        """
        if self.max_depth < 3.0:
            # return 100% littoral area fraction for shallow systems
            return 100       
        return max(100.0 * (1 - (1 - 3.0 / self.max_depth) ** self.q_bath_shape()),0)

    @save_return(internal, internals_config['bottom_temperature']['include'])
    def bottom_temperature(self) -> float:
        """Calculate the bottom (hypolimnion) temperature in the reservoir from
        a 12x1 profile of monthly average air temperatures. Refer to Equation A.2. in Praire2021_.

        Note:
            Mean Temperature of the Colder Month in Praire2021_ was interpreted
            as the coldest mean monthly temperature in the 12x1 temperature profile.

        Returns:
            float: The bottom temperature of the reservoir, degrees Celcius.
        """
        if self.temperature.coldest > 1.4:
            hypolimnion_temp = (0.6565 * self.temperature.coldest) + 10.7
        else:
            hypolimnion_temp = (0.2345 * self.temperature.coldest) + 10.11
        return hypolimnion_temp

    @save_return(internal, internals_config['surface_temperature']['include'])
    def surface_temperature(self) -> float:
        """Calculate the surface (epilimnion) temperature as the mean temperature of
        the 4 warmest months in a year. Refer to Equation A.4. in Praire2021_.

        Returns:
            float: The surface temperature of the reservoir, degrees Celcius.
        """
        return self.temperature.mean_warmest(number_of_months=4)

    @save_return(internal, internals_config['bottom_density']['include'])
    def bottom_density(self) -> float:
        """Calculate the water density in kg/m$^3$ at the bottom of the reservoir.
        Refer to Equation A.3. in Praire2021_.

        Returns:
            float: The water density at the bottom of the reservoir, kg/m$^3$.
        """
        return water_density(temp=self.bottom_temperature())

    @save_return(internal, internals_config['surface_density']['include'])
    def surface_density(self) -> float:
        """Calculate the water density in kg/m$^3$ at the surface of the reservoir.
        Refer to Equation A.4. in Praire2021_.

        Returns:
            float: The water density at the surface of the reservoir, kg/m$^3$.
        """
        return water_density(temp=self.surface_temperature())

    @save_return(internal, internals_config['thermocline_depth']['include'])
    def thermocline_depth(self, wind_speed: Optional[float] = None,
                          wind_height: float = 50) -> float:
        """Calculate the thermocline depth required for the calculation of CH$_4$
        degassing.

        Assumes that the surface water temperature is equal to
        the mean monthly air temperature from the 4 warmest months in the year.

        Follows the equation in Gorham and Boyce (1989):
        https://www.sciencedirect.com/science/article/pii/S0380133089714799
        `Influence of Lake Surface Area and Depth Upon Thermal Stratification
        and the Depth of the Summer Thermocline`, Gorham, Eville and Boyce,
        Farrell M. Journal of Great Lakes ResearchOpen AccessVolume 15,
        Issue 2, Pages 233 - 245, 1989.

        If wind_speed or monthly_temp is not supplied, a simplified equation
        from Hanna 1990 is used, as implemented in G-Res_.

        Args:
            wind_speed (Optional[float]): Average annual wind speed at the location of the reservoir, m/s.
            wind_height (float): Height at which wind_speed is measured, in meters. Default is 50 meters.

        Returns:
            float: The thermocline depth of the reservoir, m.
        """
        if wind_speed is None:
            #thermocline_depth = 10**(0.185 * math.log10(self.area) + 0.842)
            thermocline_depth = 6.95 * self.area**0.185
            log.debug("Thermocline depth calculated with model of Hanna.")
        else:
            # Calculate CD coefficient and scale wind speed to 10m
            cd_coeff = cd_factor(wind_speed)
            wind_at_10m = scale_windspeed(
                wind_speed=wind_speed, wind_height=wind_height, new_height=10)
            # Find thermocline depth in metres
            aux_var_1 = cd_coeff * air_density(
                self.temperature.mean_warmest(number_of_months=4)) * \
                wind_at_10m**2
            # It is possible that the second auxiliary variable turns out negative
            # due to some previous empirical calculations not working properly for
            # some combinations of input variables.
            aux_var_2 = 9.80665 * (self.bottom_density() - self.surface_density())
            aux_var_3 = math.sqrt(self.area * 10**6)
            try:
                thermocline_depth = 2 * math.sqrt(aux_var_1 / aux_var_2) * \
                    math.sqrt(aux_var_3)
                log.debug(
                    "Thermocline depth calculated with model of Gorham and Boyce.")
            except ValueError:
                thermocline_depth = 6.95 * self.area**0.185
                main_msg: str = \
                    "Problem with thermocline depth calculation using Gorham and Boyce model."
                extra_msg: str = \
                    "Area, depth, wind and temperature inputs produce errorenous output.\n"
                extra_msg += "Thermocline depth calculated with the model of Hanna instead."
                log.debug(main_msg, extra={'detail': extra_msg})
        return thermocline_depth

    @staticmethod
    def _k600_ch4(
            waterbody_area: float, wind_speed: float,
            wind_height: float) -> float:
        """Estimate gas transfer velocity, k600 in cm/h for CH$_4$ from wind
        speed in m/s and waterbody area in km$^2$ and return the estimate in m/d.
        Refer to Table 2, p. 1760, Model B, Vachon and Prairie, 2013:
        https://cdnsciencepub.com/doi/10.1139/cjfas-2013-0241.
        Eq. A.16. in Praire2021_.

        Args:
            waterbody_area (float): The area of the waterbody, km$^2$.
            wind_speed (float): The wind speed, m/s.
            wind_height (float): The height at which the wind speed is measured, m.

        Returns:
            float: The gas transfer velocity, k600, m/d.
        """
        if wind_height == 10:
            wind_at_10m = wind_speed
        else:
            wind_at_10m = scale_windspeed(
                wind_speed=wind_speed, wind_height=wind_height, new_height=10)
        # k600 in cm/h
        k600 = 2.51 + 1.48 * wind_at_10m + 0.39 * wind_at_10m * \
            math.log10(waterbody_area)
        # return k600 in m/d
        return k600 * 0.24

    def _kh_ch4(self) -> float:
        r"""Calculate the k$_h$ coefficient for CH$_4$. Refer to Handbook of Physics and Chemistry,
        Lide, 1994. CRC Press, Boca Raton, USA.
        
        Note:
           Unit derived through back-calculation: kg CH$_4$/(m$^3$*atm).
        Refer to Eq. A.17. in Praire2021_.

        Returns:
            float: The k$_h$ coefficient for CH$_4$, kg CH$_4$/(m$^3$*atm).
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

    def _p_ch4(self, waterbody_area: float) -> float:
        r"""Calculate the partial pressure of CH$_4$ in microatm ($\mu$atm).
        Refer to Rasillo 2014: https://doi.org/10.1111/gcb.12741.
        Terhi Rasilo, Yves T. Prairie, Paul A. del Giorgio,
        Global Change Biology,
        `Large-scale patterns in summer diffusive CH$_4$ fluxes across boreal
        lakes, and contribution to diffusive C emissions.`
        Eq. A.18. in Praire2021_.

        Args:
            waterbody_area (float): The area of the waterbody, km$^2$.

        Returns:
            float: The partial pressure of CH$_4$, $\mu atm$.
        """
        aux_1 = 1.46 + 0.03 * self.temperature.eff_temp(gas="ch4")
        aux_2 = - 0.29 * math.log10(waterbody_area)
        return 10 ** (aux_1 + aux_2)

    @save_return(internal, internals_config['surface_ch4_conc']['include'])
    def surface_ch4_conc(self, waterbody_area: float) -> float:
        r"""Calculate the surface water CH$_4$ concentration in mg CH$_4$/m$^3$ ($\mu$g/L).
        Refer to Eq. A.19. in Praire2021_.

        Args:
            waterbody_area (float): The area of the waterbody, km$^2$.

        Returns:
            float: The surface water CH$_4$ concentration in mg CH$_4$/m$^3$ ($\mu$g/L).
        """
        return self._kh_ch4() * self._p_ch4(waterbody_area)

    def ch4_preemission_factor(self, wind_height: float = 50) -> float:
        r"""Calculate the CH$_4$ emission factor for the water body, in kg CH$_4$/ha/yr.
        Refer to Eq. A.20. in Praire2021_.

        This value is used for estimating preimpoundment reservoir CH$_4$
        emissions from water bodies present in the inundated area prior to
        inundation on top of what was emitted from soil (based on inundated
        soil land cover composition and soil type).

        Args:
            wind_height (float): The height at which the wind speed is measured, in meters. Default is 50 meters.

        Returns:
            float: The CH$_4$ emission factor for the water body, kg CH$_4$/ha/yr.
        """
        # Waterbody area is the sum of water areas within the reservoir area
        # prior to impounment (construction of the reservoir)
        waterbody_area = self.wat_area_frac_pre * self.area
        if waterbody_area < 1E-6:
            return 0.0
        ch4_molar = 16  # g/mol
        em_factor = self.surface_ch4_conc(waterbody_area) * \
            self._k600_ch4(
                waterbody_area, self.mean_monthly_windspeed, wind_height) * \
            ch4_molar * 365/100
        # 365 converts from /d to /yr
        # 1/100 converts from 1/km2 to 1/ha
        return em_factor


    @save_return(internal, internals_config['retention_coeff']['include'])
    def retention_coeff(self, method: str) -> float:
        """Return the retention coefficient using the chosen calculation method.

        Args:
            method (str): Retention coefficient calculation method.

        Returns:
            float: The retention coefficient based on the selected method.
        """
        if method == 'larsen':
            ret_coeff = self.retention_coeff_larsen
        elif method in ['emp', 'empirical']:
            ret_coeff = self.retention_coeff_emp
        else:
            # Otherwise, use the Larsen and Mercier model
            log.warning('Residence time calculation method %s unknown. ' +
                        'Using the Larsen and Mercier model', method)
            ret_coeff = self.retention_coeff_larsen
        return ret_coeff

    def reservoir_conc(self, inflow_conc: float,
                     method: Optional[str] = None) -> float:
        """Calculate the reservoir concentration based on inflow concentration and
        the reservoir retention coefficient.

        Args:
            inflow_conc (float): The inflow concentration, usually TP/TN concentration, $\mu$grams/L.
            method (Optional[str]): Retention coefficient estimation method.

        Returns:
            float: The reservoir concentration, same unit as `inflow_conc`
        """
        # Method for calculating retention coefficient in reservoirs:
        # empirical/larsen
        if method is None:
            method = config['CALCULATIONS']["ret_coeff_method"]
        return float(inflow_conc * (1.0 - self.retention_coeff(method=method)))


    @save_return(internal, internals_config['trophic_status']['include'])
    def trophic_status(self, tp_inflow_conc: float, as_value: bool = True) -> Enum | str:
        """Return the reservoir's trophic status depending on the influent
        TP concentration.

        Args:
            tp_inflow_conc (float): The TP concentration in the inflow, in $\mu$grams/L.
            as_value (bool): Whether to return the value or the Enum. Default is True.

        Returns:
            Enum | str: The trophic status of the reservoir.
        """
        reservoir_tp = self.reservoir_conc(tp_inflow_conc)
        if reservoir_tp < 10.0:
            trophic_status = TrophicStatus.OLIGOTROPHIC
        if reservoir_tp < 30.0:
            trophic_status = TrophicStatus.MESOTROPHIC
        if reservoir_tp < 100.0:
            trophic_status = TrophicStatus.EUTROPHIC
        else:
            trophic_status = TrophicStatus.HYPER_EUTROPHIC
        if as_value:
            return trophic_status.value
        return trophic_status
