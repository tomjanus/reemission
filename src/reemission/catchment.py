"""
Catchment-related processes.

This module provides functionalities for calculating various catchment-related metrics, 
including phosphorus and nitrogen loads, using different methodologies.

Attributes:
    INI_FILE (str): Path to the configuration file.
    TABLES (str): Path to the parameters directory.
    tn_coeff_table (dict): TN coefficient table loaded from YAML.
    tp_coeff_table (dict): TP coefficient table loaded from YAML.
    p_loads_pop (dict): Phosphorus loads population table loaded from YAML.
    p_exports (dict): Phosphorus exports table loaded from YAML.
    config (configparser.ConfigParser): Loaded configuration settings.
    internals_config (dict): Internal variables configuration loaded from YAML.
    EPS (float): Margin of error for the sum of land use fractions.
    
.. _G-Res Technical Documentation: https://www.hydropower.org/publications/the-ghg-reservoir-tool-g-res-technical-documentation
.. _Praire2021: https://www.sciencedirect.com/science/article/pii/S1364815221001602
.. _G-Res: https://www.grestool.org/
"""
import inspect
import configparser
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, TypeVar, Type
from reemission.utils import (
    read_table, find_enum_index, read_config, save_return, get_package_file, load_yaml)
from reemission.biogenic import BiogenicFactors
from reemission.constants import Landuse
from reemission.exceptions import WrongSumOfAreasException, WrongAreaFractionsException
from reemission.globals import internal


# Set up module logger
log = logging.getLogger(__name__)
# Load path to Yaml tables
INI_FILE = get_package_file("config/config.ini")
TABLES = get_package_file("parameters")

tn_coeff_table = read_table(
    TABLES / "McDowell/landscape_TN_export.yaml", 
    schema_file=get_package_file('schemas/landscape_TN_export_schema.json'))
tp_coeff_table = read_table(
    TABLES / "McDowell/landscape_TP_export.yaml",
    schema_file=get_package_file('schemas/landscape_TP_export_schema.json'))
p_loads_pop = read_table(
    TABLES / "phosphorus_loads.yaml",
    schema_file=get_package_file('schemas/phosphorus_loads_schema.json'))
p_exports = read_table(
    TABLES / "phosphorus_exports.yaml",
    schema_file=get_package_file('schemas/phosphorus_exports_schema.json'))

config: configparser.ConfigParser = read_config(INI_FILE)
internals_config = load_yaml(get_package_file("config/internal_vars.yaml"))

# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = config.getfloat("CALCULATIONS", "eps_catchment_area_fractions")

CatchmentType = TypeVar('CatchmentType', bound='Catchment')


@dataclass
class Catchment:
    """
    Representation of a generic catchment area.

    Attributes:
        area (float): Catchment area in km$^2$.
        riv_length (float): River length before impoundment in km.
        runoff (float): Mean annual runoff in mm/year.
        population (int): Population in the catchment.
        slope (float): Catchment mean slope in %.
        precip (float): Mean annual precipitation in mm/year.
        etransp (float): Mean annual evapotranspiration in mm/year.
        soil_wetness (float): Soil wetness in mm over profile.
        mean_olsen (float): Mean P content in soil in kg/ha.
        area_fractions (List[float]): List of fractions of land representing different land uses.
        biogenic_factors (BiogenicFactors): BiogenicFactors object with categorical descriptors used 
            in the determination of the trophic status of the reservoir.
        name (str): Name of the catchment (default "n/a").

    Raises:
        WrongAreaFractionsException: If number of area fractions in the list is not equal to 
            the number of land uses.
        WrongSumOfAreasException: If area fractions do not sum to 1 ± accuracy coefficient EPS.
    """

    area: float
    riv_length: float
    runoff: float
    population: int
    slope: float
    precip: float
    etransp: float
    soil_wetness: float
    mean_olsen: float
    area_fractions: List[float]
    biogenic_factors: BiogenicFactors
    name: str = "n/a"

    def __post_init__(self) -> None:
        """Validates the provided list of land use fractions.

        Note:
            If False, set area_fractions to None.
        """
        try:
            assert len(self.area_fractions) == len(Landuse)
        except AssertionError as err:
            message: str = \
                "Wrong size of the catchment area fractions vector " + \
                f"for reservoir {self.name}."
            raise WrongAreaFractionsException(
                number_of_fractions=len(self.area_fractions),
                number_of_landuses=len(Landuse),
                message=message) from err

        try:
            assert 1 - EPS <= sum(self.area_fractions) <= 1 + EPS
        except AssertionError as err:
            message = \
                "Wrong values in the catchment area fractions vector " + \
                f"for reservoir {self.name}."
            raise WrongSumOfAreasException(
                fractions=self.area_fractions,
                accuracy=EPS,
                message=message) from err

    @classmethod
    def from_dict(cls: Type[CatchmentType], parameters: dict,
                  **kwargs) -> CatchmentType:
        """
        Initializes the class from a dictionary. Skips keys that are not featured as class's attributes.

        Args:
            cls (Type[CatchmentType]): The class type.
            parameters (dict): The dictionary of parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            CatchmentType: An instance of the Catchment class.
        """
        return cls(**{
            k: v for k, v in parameters.items()
            if k in inspect.signature(cls).parameters}, **kwargs)

    @property
    def population_density(self) -> float:
        """
        Calculates the population density.
        
        Note:
            Population density is derived from catchment population and catchment
            area. [capita/km$^2$]. From Eq. A.25. in Praire2021_.

        Returns:
            float: Population density in capita/km$^2$.
        """
        return self.population/self.area

    @property
    def area_ha(self) -> float:
        """
        Converts area from km$^2$ to hectares.

        Returns:
            float: Area in hectares.
        """
        return self.area * 100.0

    def landuse_area(self, landuse_fraction: float) -> float:
        """
        Calculates land use area from catchment area and land use fraction.

        Args:
            landuse_fraction (float): Fraction of land use.

        Returns:
            float: Land use area in km$^2$.
        """
        return self.area * landuse_fraction

    def landuse_area_ha(self, landuse_fraction: float) -> float:
        """
        Calculates land use area in hectares from catchment area and land use fraction.

        Args:
            landuse_fraction (float): Fraction of land use.

        Returns:
            float: Land use area in hectares.
        """
        return self.area_ha * landuse_fraction

    @property
    def discharge(self) -> float:
        """
        Calculates mean annual discharge.
        
        Note:
            mean annual discharge in m$^3$/year is calculated from runoff in
            mm/year and area in km$^2$

        Returns:
            float: Mean annual discharge in m$^3$/year.
        """
        return 1_000 * self.runoff * self.area

    @property
    def discharge_cumecs(self) -> float:
        """
        Returns mean annual discharge in cubic meters per second.

        Returns:
            float: Mean annual discharge in m$^3$/s.
        """
        return self.discharge/(365.25*24*60*60)

    def river_area_before_impoundment(
            self, river_width: Optional[float] = None) -> float:
        r"""
        Calculates the area taken up by the river in the impounded area prior to impoundment.

        Args:
            river_width (Optional[float]): River width in meters. If None, a default allometric 
                formula is used.
                
        **Formula:**
        
        .. math::
            \begin{equation}
                A_{pre} = 5.9 \, 10^{-6} \, L_{river} \, A_{catchment}^{0.32}
            \end{equation}
        where:
            * $A_{pre}$ : pre-inundation area, [km$^2$]
            * $A_{catchment}$ : catchment area, [km$^2$]
            * $L_{river}$ : lentgh of the river prior to impoundment, [km]

        Returns:
            float: River area in km$^2$.
        """
        if river_width is None:
            # Use simple allometric formula of Whipple et al. 2013 (G-Res)
            river_width = 5.9 * self.area**0.32
        return 1E-3 * river_width * self.riv_length

    def p_human_input_gres(self) -> float:
        """
        Calculates phosphorus load/input from human activity and level of wastewater treatment.

        Note:
            Follows the methodology applied in G-Res_

        Returns:
            float: Phosphorus load in kgP/year.
        """
        treatment_factor = self.biogenic_factors.treatment_factor
        load = 0.002 * 365.25 * self.population * \
            p_loads_pop[treatment_factor.value]
        return load

    def p_land_input_gres(self) -> float:
        """
        Calculates phosphorus load/input from land in the catchment.

        Note:
            Considers differences in P emissions across different landuse types
            Follows the methodology applied in G-Res_

        Returns:
            float: Phosphorus load in kgP/year.
        """
        intensity = self.biogenic_factors.landuse_intensity
        landuse_names = [landuse.value for landuse in Landuse]
        # Area marging below which the output is output as zero
        EPS = 1e-6

        # Define two inner funtions to determine land cover export coefficients
        # for two instances in which the coefficients in the phosphorus exports
        # table additionally depend on the catchment area and hence, are not
        # constant
        def fun_exp(area_fraction: float, fun_name: str) -> float:
            """
            Regression function for P export from crops and forest.

            Args:
                area_fraction (float): Fraction of land use area.
                fun_name (str): Name of the function.

            Returns:
                float: Phosphorus export coefficient.
            """
            if fun_name == 'crop export':
                reg_coeffs = (1.818, 0.227)
            elif fun_name == 'forest export':
                reg_coeffs = (0.914, 0.014)
            else:
                log.error(
                    'Regression function %s unknown. Returning zero.',
                    fun_name)
                return 0.0
            if area_fraction < EPS:
                return 0.0
            try:
                # Equation gives P export coefficient in kgP/ha/yr
                # Eq. A.23. and A.24. from Praire2021
                p_export_coeff = 0.01 * 10 ** (
                    reg_coeffs[0] - math.log10(
                        self.landuse_area(area_fraction)) * reg_coeffs[1])
            except ValueError:
                log.error(
                    f"Error processing {fun_name}, area fraction {area_fraction}")
                log.error(
                    'Export coefficient could not be calculated. Returning 0.')
                p_export_coeff = 0.0
            return p_export_coeff

        load: float = 0.0
        for landuse, area_fraction in zip(landuse_names, self.area_fractions):
            # iterate and calculate the total phosphorus load
            coefficient = p_exports[landuse][intensity.value]
            if coefficient in ('crop export', 'forest export'):
                coefficient = fun_exp(area_fraction, coefficient)
            load += coefficient * area_fraction * self.area_ha
        return load

    def p_input_gres(self) -> float:
        """
        Calculates annual input of phosphorus from catchment to the reservoir.

        Note:
            Follows the G-Res_ approach.

        Returns:
            float: Annual phosphorus input in kgP/year.
        """
        return self.p_human_input_gres() + self.p_land_input_gres()

    def p_input_mcdowell(self) -> float:
        """
        Calculates annual input of phosphorus from catchment to the reservoir using McDowell regression.

        Returns:
            float: Annual phosphorus input in kgP/year.
        """
        # 1e-6 converts mg/m3 to kg/m3; discharge is given in m3/year
        return 1e-6 * self.inflow_p_conc_mcdowell() * self.discharge

    def inflow_p_conc_mcdowell(self) -> float:
        """
        Calculates influent phosphorus concentration to the reservoir using McDowell2020 regression model.

        Returns:
            float: Influent phosphorus concentration in $\mu$g/L.
        """
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
        def inflow_tp() -> float:
            """Calculate inflow TP in [micrograms/L] using McDowell's
            regression coefficients.
            """
            ln_tp = (
                intercept
                + olsen_p * self.mean_olsen
                + prec_coeff * self.precip / 12
                + slope_coeff * self.slope
                + cropland_coeff * crop_percent
                + et_coeff * self.etransp / 12
                + biome_coeff)
            inflow_p = 10**3 * bias_corr * math.exp(ln_tp)
            return inflow_p

        return inflow_tp()

    def inflow_p_conc_gres(self) -> float:
        """
        Calculates influent phosphorus concentration to the reservoir following the G-Res_ approach.

        Returns:
            float: Influent phosphorus concentration in micrograms/L.
        """
        load_pop = self.p_human_input_gres()
        load_land = self.p_land_input_gres()
        load_total = load_pop + load_land
        return 10**6 * load_total / self.discharge

    @save_return(internal, internals_config['inflow_p_conc']['include'])
    def inflow_p_conc(self, method: str) -> float:
        """
        Calculates median influent total phosphorus concentration.

        Args:
            method (str): Method to use for calculation ("g-res" or "mcdowell").

        Returns:
            float: Median influent total phosphorus concentration in micrograms/L.
        """
        if method == "g-res":
            output = self.inflow_p_conc_gres()
        elif method == "mcdowell":
            output = self.inflow_p_conc_mcdowell()
        else:
            log.warning(
                'Unrecognized method %s. ',
                method + '. Using the default G-Res method.')
            output = self.inflow_p_conc_gres()
        return output

    @save_return(internal, internals_config['inflow_n_conc']['include'])
    def inflow_n_conc(self) -> float:
        """
        Calculates median influent total nitrogen concentration.

        Attention:
            Contrary to Phosphorus, no other method than McDowell is available.
        Returns:
            float: Median influent total nitrogen concentration in micrograms/L.
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

    @save_return(internal, internals_config['nitrogen_load']['include'])
    def nitrogen_load(self) -> float:
        """
        Calculates total nitrogen load entering the reservoir with catchment runoff.

        Returns:
            float: Total nitrogen load in kgN/year.
        """
        inflow_n = self.inflow_n_conc()
        # 1e-6 converts mg/m3 (microgram/L) to kg/m3
        return 1e-6 * self.discharge * inflow_n

    @save_return(internal, internals_config['phosphorus_load']['include'])
    def phosphorus_load(self, method: str) -> float:
        """
        Calculates total phosphorus load entering the reservoir with catchment runoff.

        Args:
            method (str): P export calculation method: "g-res" or "mcdowell".

        Returns:
            float: Total phosphorus load in kgP/year.
        """
        inflow_p = self.inflow_p_conc(method)
        # 1e-6 converts mg/m3 (microgram/L) to kg/m3
        return 1e-6 * self.discharge * inflow_p
