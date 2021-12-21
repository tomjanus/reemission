""" Module providing data and calculations relating to catchments """
import os
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, ClassVar
from .utils import read_table, find_enum_index
from .biogenic import BiogenicFactors
from .constants import Landuse

# Set up module logger
log = logging.getLogger(__name__)
# Load path to Yaml tables
module_dir = os.path.dirname(__file__)
TABLES = os.path.abspath(
    os.path.join(module_dir, '..', '..', 'data', 'emissions'))
# Provide tables as module variables
tn_coeff_table: ClassVar[Dict] = read_table(
    os.path.join(TABLES, 'McDowell', 'landscape_TN_export.yaml'))
tp_coeff_table: ClassVar[Dict] = read_table(
    os.path.join(TABLES, 'McDowell', 'landscape_TP_export.yaml'))
p_loads_pop: ClassVar[Dict] = read_table(
    os.path.join(TABLES, 'phosphorus_loads.yaml'))
p_exports: ClassVar[Dict] = read_table(
    os.path.join(TABLES, 'phosphorus_exports.yaml'))
# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = 0.01


@dataclass
class Catchment:
    """ Class representing a generic catchment """
    area: float  # Catchment area. ha
    runoff: float  # Mean annual runoff, mm/year
    population: int  # Population, capita
    slope: float  # Catchment mean slope, %
    precip: float  # Mean annual precipitation, mm/year
    etransp: float  # Mean annual evapotranspiration, mm/year
    soil_wetness: float  # Soil wetness in mm over profile
    area_fractions: List[float]  # Fractions of catchment area allocated, -
    biogenic_factors: BiogenicFactors  # categorical descriptors playing part
    # in the determination of the trophic status of the the reservoir

    def __post_init__(self):
        """ Check if the provided list of landuse fractions has the same
            length as the list of landuses. If False, set area_fractions to
            None
        """
        try:
            assert len(self.area_fractions) == len(Landuse)
        except AssertionError:
            log.error(
                'List of area fractions not equal to number of landuses.')
            log.error('Setting fractions to a vector of all zeros.')
            self.area_fractions = [0] * len(Landuse)

        try:
            assert 1 - EPS <= sum(self.area_fractions) <= 1 + EPS
        except AssertionError:
            log.error(
                'Sum of area fractions is not equal 1.0.')
            log.error('Setting fractions to a vector of all zeros.')
            self.area_fractions = [0] * len(Landuse)

    def _landuse_area(self, landuse_fraction):
        """ Return landuse area from catchment area and landuse fraction """
        return self.area * landuse_fraction

    @property
    def discharge(self) -> float:
        """ Calculate mean annual discharge in m3/year from runoff in mmm/year
            and area in ha """
        return 10.0 * self.runoff * self.area

    def phosphorus_load_pop_gres(self) -> float:
        """ Return phosphorus load in kg P yr-1 from human activity from the
            population and the level of wastewater treament.
            Follows the methodology applied in g-res tool:
            https://g-res.hydropower.org/. """
        treatment_factor = self.biogenic_factors.treatment_factor
        load = 0.002 * 365.25 * self.population * p_loads_pop[
            treatment_factor.value]
        return load

    def phosphorus_load_land_gres(self) -> float:
        """ Calculate phosphorus load from land in the catchment, considering
            differences in P emissions across different landuse types.
            Phosphorus load returned in kg P yr-1.
            Follows the methodology applied in g-res tool:
            https://g-res.hydropower.org/. """
        intensity = self.biogenic_factors.landuse_intensity
        landuse_names = [landuse.value for landuse in Landuse]

        # Define two inner funtions to determine land cover export coefficients
        # for two instances in which the coefficients in the phosphorus exports
        # table additionally depend on the catchment area and hence, are not
        # constant
        def fun_exp(area_fraction: float, fun_name: str) -> float:
            """ Regression vs area fraction for P export from crops and
                forest """
            # TODO: Explain what are these two exponential functions and how
            #       to choose between them
            if fun_name == 'fun_exp1':
                reg_coeffs = (1.818, 0.227)
            elif fun_name == 'fun_exp2':
                reg_coeffs = (0.914, 0.014)
            else:
                log.error(
                    'Regression function %s unknown. Returning zero.', fun_name)
                return 0.0
            try:
                p_export_coeff = 0.01 * 10**(
                    reg_coeffs[0] - reg_coeffs[1] * math.log10(
                        self._landuse_area(area_fraction)))
            except ValueError:
                p_export_coeff = 0.0
            return p_export_coeff

        load = 0
        for landuse, area_fraction in zip(landuse_names, self.area_fractions):
            # iterate and calculate the total phosphorus load
            coefficient = p_exports[landuse][intensity.value]
            if coefficient in ('fun_exp1', 'fun_exp2'):
                coefficient = fun_exp(area_fraction, coefficient)
            load += coefficient * area_fraction * self.area
        return load

    def phosphorus_load_gres(self):
        """ Return annual discharge of P from catchment to the reservoir in
            kg P yr-1 using g-res approach """
        return self.phosphorus_load_pop_gres() + \
            self.phosphorus_load_land_gres()

    def phosphorus_load_mcdowell(self):
        """ Calculate annual discharge of P from catchment to the reservoir
            in kg P yr-1 using McDowell regression """
        inflow_p = self._inflow_p_mcdowell()  # micrograms/L
        # discharge is given in m3/year
        return 1e-6 * inflow_p * self.discharge

    def _inflow_p_gres(self) -> float:
        """ Calculate influent phosphorus concentration to the reservoir
            in micrograms/L following the G-Res approach. """
        load_pop = self.phosphorus_load_pop_gres()
        load_land = self.phosphorus_load_land_gres()
        load_total = load_pop + load_land
        return 10**6 * load_total / self.discharge

    def _inflow_p_mcdowell(self, init_p: float = 5.0, eps=1e-6) -> float:
        """ Calculate influent phosphorus concetration to the reservoir
            in micrograms/L using regression model of McDowell 2020 """
        biome = self.biogenic_factors.biome
        # Calculate natural logarithm of total phosphorus conc. in mg/L
        # Find percentage of catchment area allocated to crops
        crop_index = find_enum_index(enum=Landuse, to_find=Landuse.CROPS)
        crop_percent = 100.0 * self.area_fractions[crop_index]
        # Find coefficients from the McDowell table of regression coefficients
        intercept = tp_coeff_table['intercept']['coeff']
        olsen_p = tp_coeff_table['olsen_p']['coeff']
        prec_coeff = tp_coeff_table['mean_prec']['coeff']
        slope_coeff = tp_coeff_table['mean_slope']['coeff']
        cropland_coeff = tp_coeff_table['cropland']['coeff']
        et_coeff = tp_coeff_table['pet']['coeff']
        biome_coeff = tp_coeff_table['biome'][biome.value]['coeff']
        bias_corr = tp_coeff_table['corr']

        # Define the inner function calculating the final output P conc.
        def find_inflow_p(p_conc):
            """ Calculate inflow TP using McDowell's regression coeffs """
            ln_tp = intercept + olsen_p * p_conc + \
                prec_coeff*self.precip/12 + \
                slope_coeff * self.slope + cropland_coeff * crop_percent + \
                et_coeff * self.etransp / 12 + biome_coeff
            inflow_p = 10**3 * bias_corr * math.exp(ln_tp)
            return inflow_p

        # Define the wrapper function for running find_inflow in a recurrent
        # manner until convergence
        def recurrence(p_conc):
            """ Calls find_inflow_tp in a recurrent manner until output
                 concentration is equal to input concentration (within set
                 accurracy)"""
            if p_conc - find_inflow_p(p_conc) > eps:
                recurrence(find_inflow_p(p_conc))
            else:
                return find_inflow_p(p_conc)
        return recurrence(init_p)

    def median_inflow_p(self, method: str = "g-res") -> float:
        """ Calculate median influent total phosphorus concentration in
            micrograms/L entering the reservoir with runoff """
        if method == "g-res":
            return self._inflow_p_gres()
        if method == "mcdowell":
            return self._inflow_p_mcdowell()

        log.warning('Unrecognize method %s. ', method +
                    '. Using default g-res method.')
        return self._inflow_p_gres()

    def median_inflow_n(self) -> float:
        """ Calculate median influent total nitrogen concentration in
            micrograms/L entering the reservoir with runoff.
            Contrary to phosphorus, no other method than McDowell is used.
        """
        biome = self.biogenic_factors.biome
        intercept = tn_coeff_table['intercept']['coeff']
        prec_coeff = tn_coeff_table['mean_prec']['coeff']
        slope_coeff = tn_coeff_table['mean_slope']['coeff']
        cropland_coeff = tn_coeff_table['cropland']['coeff']
        soil_wet_coeff = tn_coeff_table['soil_wet']['coeff']
        biome_coeff = tn_coeff_table['biome'][biome.value]['coeff']
        bias_corr = tn_coeff_table['corr']
        # Find percentage of catchment area allocated to crops
        crop_index = find_enum_index(enum=Landuse, to_find=Landuse.CROPS)
        crop_percent = 100.0 * self.area_fractions[crop_index]
        inflow_n = 10**3 * math.exp(
            intercept + prec_coeff*self.precip/12 + slope_coeff*self.slope +
            cropland_coeff*crop_percent + soil_wet_coeff*self.soil_wetness +
            biome_coeff) * bias_corr
        return inflow_n

    def nitrogen_load(self) -> float:
        """ Calculate total nitrogen (TN) load in kg N yr-1 entering the
            reservoir with catchment runoff """
        inflow_n = self.median_inflow_n()
        return 1e-5 * self.area * self.runoff * inflow_n

    def phosphorus_load(self) -> float:
        """ Calculate total phosphorus (TP) load in kg P yr-1 entering the
            reservoir with catchment runoff """
        inflow_p = self.median_inflow_p()
        return 1e-5 * self.area * self.runoff * inflow_p
