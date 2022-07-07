"""Catchment-related processes."""
import os
import inspect
import logging
import pathlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar, Type
from reemission.utils import read_table, find_enum_index
from reemission.biogenic import BiogenicFactors
from reemission.constants import Landuse

# Set up module logger
log = logging.getLogger(__name__)
# Load path to Yaml tables
MODULE_DIR = os.path.dirname(__file__)
TABLES = os.path.abspath(os.path.join(MODULE_DIR, 'parameters'))
# Provide tables as module variables
tn_coeff_table: Dict = read_table(
    pathlib.Path(TABLES, 'McDowell', 'landscape_TN_export.yaml'))
tp_coeff_table: Dict = read_table(
    pathlib.Path(TABLES, 'McDowell', 'landscape_TP_export.yaml'))
p_loads_pop: Dict = read_table(
    pathlib.Path(TABLES, 'phosphorus_loads.yaml'))
p_exports: Dict = read_table(
    pathlib.Path(TABLES, 'phosphorus_exports.yaml'))

# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = 0.01

CatchmentType = TypeVar('CatchmentType', bound='Catchment')


@dataclass
class Catchment:
    """Class representing a generic catchment.

    Attrributes:
        area: Catchment area, km2
        riv_length: River length before impoundment, km
        runoff: Mean annual runoff, mm/year
        population: Population in the catchment, capita
        slope: Catchment mean slope, %
        precip: Mean annual precipitation, mm/year
        etransp: Mean annual evapotranspiration, mm/year
        soil_wetness: Soil wetness in mm over profile
        area_fractions: List of fractions of land representing different
            land uses.
        biogenic_factors: biogenic.BiogenicFactor object with categorical
            descriptors used in the determination of the trophic status of the
            reservoir.
    """

    area: float
    riv_length: float
    runoff: float
    population: int
    slope: float
    precip: float
    etransp: float
    soil_wetness: float
    area_fractions: List[float]
    biogenic_factors: BiogenicFactors

    def __post_init__(self) -> None:
        """Check if the provided list of landuse fractions has the same
        length as the list of landuses.

        If False, set area_fractions to None.
        """
        try:
            assert len(self.area_fractions) == len(Landuse)
        except AssertionError:
            log.error('List of area fractions not equal to number of landuses.')
            log.error('Setting fractions to a vector of all zeros.')
            self.area_fractions = [0.0] * len(Landuse)

        try:
            assert 1 - EPS <= sum(self.area_fractions) <= 1 + EPS
        except AssertionError:
            log.error('Sum of area fractions is not equal 1.0.')
            log.error('Setting fractions to a vector of all zeros.')
            self.area_fractions = [0.0] * len(Landuse)

    @classmethod
    def from_dict(cls: Type[CatchmentType], parameters: dict,
                  **kwargs) -> CatchmentType:
        """Initializes the class from a dictionary. Skips keys that are not
        featured as class's attribiutes."""
        return cls(**{
            k: v for k, v in parameters.items()
            if k in inspect.signature(cls).parameters}, **kwargs)

    @property
    def population_density(self) -> float:
        """Derive population density from catchment population and catchment
        area. [capita/km2]. From Eq. A.25. in Praire2021.
        """
        return self.population/self.area

    @property
    def area_ha(self) -> float:
        """Get area in ha from area in [km2]."""
        return self.area * 100.0

    def landuse_area(self, landuse_fraction: float) -> float:
        """Return landuse area [km2] from catchment area and landuse
        fraction.
        """
        return self.area * landuse_fraction

    def landuse_area_ha(self, landuse_fraction: float) -> float:
        """Return landuse area [ha] from catchment area and landuse
        fraction."""
        return self.area_ha * landuse_fraction

    @property
    def discharge(self) -> float:
        """Calculate mean annual discharge in [m3/year] from runoff in
        [mm/year] and area in [km2]."""
        return 1_000 * self.runoff * self.area

    def river_area_before_impoundment(
            self, river_width: Optional[float] = None) -> float:
        r"""Calculates the area taken up by the river in the impounded
        (reservoir) area prior to impoundement.

        .. math::
            :nowrap:
            \begin{equation}
                A_{pre} = 5.9 \, 10^{-6} \, L_{river} \, A_{catchment}^{0.32}
            \end{equation}
        where:
            :math:`A_{pre}`: pre-inundation area, [km2]
            :math:`A_{catchment}`: catchment area, [km2]
            :math: `L_{river}`: lentgh of the river prior to impoundment, [km]

        Args:
            river_width: River width in km

        Returns:
            River area in km2
        """
        if river_width is None:
            # Use simple allometric formula of Whipple et al. 2013 (G-Res)
            river_width = 5.9 * self.area**0.32
        return 1E-6 * river_width * self.riv_length

    def p_human_input_gres(self) -> float:
        """Return phosphorus load/input in [kgP yr-1] from human activity from
        population and the level of wastewater treament.
        Follows the methodology applied in G-Res
        https://g-res.hydropower.org/.
        """
        treatment_factor = self.biogenic_factors.treatment_factor
        load = 0.002 * 365.25 * self.population * \
            p_loads_pop[treatment_factor.value]
        return load

    def p_land_input_gres(self) -> float:
        """Calculate phosphorus load/iput from land in the catchment,
        considering differences in P emissions across different landuse types.
        Phosphorus load returned in [kgP yr-1].
        Follows the methodology applied in G-Res:
        https://g-res.hydropower.org/."""
        intensity = self.biogenic_factors.landuse_intensity
        landuse_names = [landuse.value for landuse in Landuse]

        # Define two inner funtions to determine land cover export coefficients
        # for two instances in which the coefficients in the phosphorus exports
        # table additionally depend on the catchment area and hence, are not
        # constant
        def fun_exp(area_fraction: float, fun_name: str) -> float:
            """Regression vs area fraction for P export from crops and
            forest."""
            if fun_name == 'crop_export':
                reg_coeffs = (1.818, 0.227)
            elif fun_name == 'forest_export':
                reg_coeffs = (0.914, 0.014)
            else:
                log.error(
                    'Regression function %s unknown. Returning zero.',
                    fun_name)
                return 0.0
            if area_fraction < 1e-6:
                return 0.0
            try:
                # Equation gives P export coefficient in kgP/ha/yr
                # Eq. A.23. and A.24. from Praire2021
                p_export_coeff = 0.01 * 10 ** (
                    reg_coeffs[0] - math.log10(
                        self.landuse_area(area_fraction)) * reg_coeffs[1])
            except ValueError:
                log.error(f"Error processing {fun_name}, area fraction {area_fraction}")
                log.error('Export coefficient could not be calculated. Returning 0.')
                p_export_coeff = 0.0
            return p_export_coeff

        load = 0.0
        for landuse, area_fraction in zip(landuse_names, self.area_fractions):
            # iterate and calculate the total phosphorus load
            coefficient = p_exports[landuse][intensity.value]
            if coefficient in ('crop_export', 'forest_export'):
                coefficient = fun_exp(area_fraction, coefficient)
            load += coefficient * area_fraction * self.area_ha
        return load

    def p_input_gres(self):
        """Return annual input of P from catchment to the reservoir in
        [kg P yr-1] using g-res approach."""
        return self.p_human_input_gres() + self.p_land_input_gres()

    def p_input_mcdowell(self):
        """Calculate annual input of P from catchment to the reservoir
        in [kg P yr-1] using McDowell regression."""
        # 1e-6 converts mg/m3 to kg/m3; discharge is given in m3/year
        return 1e-6 * self.inflow_p_conc_mcdowell() * self.discharge

    def inflow_p_conc_mcdowell(self, init_p: float = 5.0, eps=1e-6) -> float:
        """Calculate influent phosphorus concetration to the reservoir
        in micrograms/L using regression model of McDowell 2020."""
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
        def inflow_tp(p_conc) -> float:
            """Calculate inflow TP in [micrograms/L] using McDowell's
            regression coefficients.
            """
            ln_tp = (
                intercept
                + olsen_p * p_conc
                + prec_coeff * self.precip / 12
                + slope_coeff * self.slope
                + cropland_coeff * crop_percent
                + et_coeff * self.etransp / 12
                + biome_coeff)
            inflow_p = 10**3 * bias_corr * math.exp(ln_tp)
            return inflow_p

        # Define the wrapper function for running find_inflow in a recurrent
        # manner until convergence
        def recurrent_tp(tp_concentration: float) -> float:
            """Calls find_inflow_tp in a recurrent manner until output
            concentration is equal to input concentration (within set
            accurracy) eps."""
            if tp_concentration - inflow_tp(tp_concentration) > eps:
                return recurrent_tp(inflow_tp(tp_concentration))
            return inflow_tp(tp_concentration)

        return recurrent_tp(init_p)

    def inflow_p_conc_gres(self) -> float:
        """Calculate influent phosphorus concentration to the reservoir
        in [micrograms/L] following the G-Res approach."""
        load_pop = self.p_human_input_gres()
        load_land = self.p_land_input_gres()
        load_total = load_pop + load_land
        return 10**6 * load_total / self.discharge

    def inflow_p_conc(self, method: str = "g-res") -> float:
        """Calculate median influent total phosphorus concentration in
        [micrograms/L] entering the reservoir with runoff."""
        if method == "g-res":
            output = self.inflow_p_conc_gres()
        elif method == "mcdowell":
            output = self.inflow_p_conc_mcdowell()
        else:
            log.warning('Unrecognized method %s. ', method + '. Using the default G-Res method.')
            output = self.inflow_p_conc_gres()
        return output

    def inflow_n_conc(self) -> float:
        """Calculate median influent total N concentration in [micrograms/L]
        entering the reservoir with runoff.

        Contrary to Phosphorus, no other method than McDowell is available for.
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
        inflow_n = (
            10**3
            * math.exp(
                intercept
                + prec_coeff * self.precip / 12
                + slope_coeff * self.slope
                + cropland_coeff * crop_percent
                + soil_wet_coeff * self.soil_wetness
                + biome_coeff
            ) * bias_corr)
        return inflow_n

    def nitrogen_load(self) -> float:
        """Calculate total nitrogen (TN) load in kg N yr-1 entering the
        reservoir with catchment runoff."""
        inflow_n = self.inflow_n_conc()
        # 1e-6 converts mg/m3 (microgram/L) to kg/m3
        return 1e-6 * self.discharge * inflow_n

    def phosphorus_load(self) -> float:
        """Calculate total phosphorus (TP) load in kg P yr-1 entering the
        reservoir with catchment runoff."""
        inflow_p = self.inflow_p_conc()
        # 1e-6 converts mg/m3 (microgram/L) to kg/m3
        return 1e-6 * self.discharge * inflow_p
