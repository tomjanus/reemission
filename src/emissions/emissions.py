""" Module containing classes for the calculation of GHG emissions
    resulting from inundation """
import os
import configparser
import math
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from numpy import mean
import yaml

# Get relative imports to data
module_dir = os.path.dirname(__file__)
INI_FILE = os.path.abspath(
    os.path.join(module_dir, '..', '..', 'config', 'emissions', 'config.ini'))
TABLES = os.path.abspath(
    os.path.join(module_dir, '..', '..', 'data', 'emissions'))


class Climate(Enum):
    """ Enumeration class with climate types """
    BOREAL = "Boreal"
    SUBTROPICAL = "Subtropical"
    TEMPERATE = "Temperate"
    TROPICAL = "Tropical"


class Landuse(Enum):
    """ Enumeration class with landuse types """
    BARE = "bare"
    SNOW_ICE = "snow_ice"
    URBAN = "urban"
    WATER = "water"
    WETLANDS = "wetlands"
    CROPS = "crops"
    SHRUBS = "shrubs"
    FOREST = "forest"


class SoilType(Enum):
    """ Enumeration class with soil types """
    MINERAL = "mineral"
    ORGANIC = "organic"


def read_config(file_path: Path) -> dict:
    """
        Read the .ini file with global configuration parameters and return
        the parsed config object
        :param file_path: path to the .ini file
        :return config: parsed .ini file
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def read_table(file_path: Path) -> Optional[dict]:
    """ Read yaml table given in file_path """
    try:
        stream = open(file_path, "r")
        return yaml.safe_load(stream)
    except FileNotFoundError as exc:
        print(exc)
        print(f"File in {file_path} not found.")
    except yaml.YAMLError as exc:
        print(exc)
        print(f"File in {file_path} cannot be parsed.")
    finally:
        stream.close()


@dataclass
class MonthlyTemperature:
    """ DataClass storing a monthly temperature profile """
    temp_profile: List[float]

    def __post_init__(self):
        assert len(self.temp_profile) == 12

    def calculate_eff_temp(self, coeff: float = 0.05) -> float:
        """ Calculate effective annual Temperature CO2 (deg C; for CO2
        diffusion estimation) """
        return math.log10(
            mean([10**(temp * coeff) for temp in self.temp_profile]))/coeff


class Emission(ABC):
    """ Abstract emission class that acts as a base class for all emissions """
    catchment_area: float
    reservoir_area: float
    config: dict
    preinund_area: float

    def __init__(self, catchment_area, reservoir_area,
                 config_file=INI_FILE):
        self.catchment_area = catchment_area
        self.reservoir_area = reservoir_area  # in km2
        self.config = read_config(config_file)
        self.preinund_area = self.__calculate_pre_inund_area()

    def __calculate_pre_inund_area(self) -> float:
        """ Calculate pre inundatation area of a waterbody based on
            the catchment area, using regression """
        return 2.125 * 5.9 * 10**(-3) * (0.01 * self.catchment_area)**0.32

    @abstractmethod
    def calculate_profile(self, *args, **kwargs) -> List[float]:
        """ Abstract method for calculating an emission profile """

    @abstractmethod
    def calculate_total(self, *args, **kwargs) -> float:
        """ Abstract method for calculating total emission (factor) """


class CarbonDioxideEmission(Emission):
    """ Class for calculating CO2 emissions from reservoirs """
    eff_temp: float
    soil_carbon: float
    reservoir_tp: float
    pre_impoundment_table: dict
    area_fractions: List[float]

    def __init__(self, catchment_area, reservoir_area, eff_temp,
                 soil_carbon, reservoir_tp, area_fractions,
                 config_file=INI_FILE):
        super().__init__(catchment_area, reservoir_area, config_file)
        # Initialise input data specific to carbon dioxide emissions
        self.eff_temp = eff_temp  # EFF temp CO2
        self.soil_carbon = soil_carbon  # in kg/m2 (section 4.12; sheet 'input data')
        self.reservoir_tp = reservoir_tp  # in ppb / ug L-1 (section 5.10; sheet 'input data')
        try:
            assert len(area_fractions) == len(Landuse)
            self.area_fractions = area_fractions
        except AssertionError:
            print(
                'List of area fractions not equal to number of landuse types')
            self.area_fractions = None
        # Read the tables
        self.pre_impoundment_table = read_table(
            os.path.join(TABLES, 'Carbon_Dioxide', 'pre-impoundment.yaml'))
        self.par = self.__initialize_parameters_from_config(
            ['c_1', 'age', 'temp', 'resArea', 'soilC', 'ResTP', 'calc',
             'conv_coeff'])

    def __initialize_parameters_from_config(self, list_of_constants: list) \
            -> SimpleNamespace:
        """ Read constants (parameters) from config file """
        const_dict = {name: self.config.getfloat('CARBON_DIOXIDE', name)
                      for name in list_of_constants}
        return SimpleNamespace(**const_dict)

    def __fluxes_per_year(self,
                          years: Tuple[int] = (1, 5, 10, 20, 30, 40, 50, 100)) \
            -> Optional[list]:
        """ Calculate CO2 fluxes for a list of years given as an argument """
        def find_flux(year: int) -> float:
            """ Inner function for calculating flux for a defined year
                return flux in g CO2eq m-2 yr-1
            """
            flux = self.par.conv_coeff * 10.0 ** (
                self.par.c_1 +
                math.log10(year) * self.par.age +
                self.eff_temp * self.par.temp +
                math.log10(self.reservoir_area) * self.par.resArea +
                self.soil_carbon * self.par.soilC +
                math.log10(self.reservoir_tp) * self.par.ResTP) * \
                (1 - (self.preinund_area/self.reservoir_area))
            return flux

        if len(years) == 1:
            return find_flux(years[0])

        # Calculate flux per each year in the list of years
        return [find_flux(year) for year in years]

    def __gross_total_emission(self) -> float:
        """
        Calculate gross total CO2 emissions in g CO2eq m-2 yr-1
        from a reservoir over 100 years
        """
        flux = self.par.conv_coeff * 10.0 ** (
            self.par.c_1 +
            self.eff_temp * self.par.temp +
            math.log10(self.reservoir_area) * self.par.resArea +
            self.soil_carbon * self.par.soilC +
            math.log10(self.reservoir_tp) * self.par.ResTP) * \
            (1 - (self.preinund_area/self.reservoir_area)) * \
            ((100**(self.par.calc+1) -
              0.5**(self.par.calc+1)) / ((self.par.calc+1)*(100-0.5)))
        return flux

    def __net_total_emission(self) -> float:
        """
        Calculate net total CO2 emissions, i.e. gross - non anthropogenic
        (in g CO2eq m-2 yr-1) from a reservoir over 100 years
        """
        return (self.__gross_total_emission() -
                self.__fluxes_per_year(years=(100,)))

    def __pre_impoundment_emission(self, climate: str, soil_type: str) -> float:
        """
        Calculate CO2 emissions  g CO2eq m-2 yr-1 from the inundated area
        prior to impoundment
        """
        # TODO: Check if sum of fractions == 1, otherwise raise error
        __list_of_landuses = list(Landuse.__dict__['_member_map_'].values())

        emissions = []
        for landuse, fraction in zip(__list_of_landuses, self.area_fractions):
            # Calculate area in ha (not km2) allocated to each landuse
            area_landuse = (100 * self.reservoir_area) * fraction
            coeff = self.pre_impoundment_table.get(
                climate.value, {}).get(
                    soil_type.value, {}).get(
                        landuse.value, 0)
            emissions.append(area_landuse * coeff)
        # Total emission in t CO2-C /yr
        tot_emission = sum(emissions)
        # Total emission in g CO2eq m-2 yr-1
        return tot_emission/((100 * self.reservoir_area)*0.01)

    def calculate_profile(self, climate: str,
                          soil_type: str,
                          years: Tuple[int] = (1, 5, 10, 20, 30, 40, 50, 100)) \
            -> List[float]:
        """ Calculate CO2 emissions for a list of years given as an argument
        Flux at year x age - pre-impoundment emissions - non-anthropogenic
        emissions """
        pre_impoundment = self.__pre_impoundment_emission(climate, soil_type)
        integrated_emission = self.__fluxes_per_year((100,))
        fluxes_profile = self.__fluxes_per_year(years)
        final_profile = [flux - integrated_emission - pre_impoundment for
                         flux in fluxes_profile]
        return final_profile

    def calculate_total(self, climate: str, soil_type: str) -> float:
        """ Overall integrated emissions for lifetime, taken as 100 yrs
            unit:  g CO2eq m-2 yr-1 """
        net_total_emission = self.__net_total_emission()
        pre_impoundment_emission = \
            self.__pre_impoundment_emission(climate, soil_type)
        return net_total_emission - pre_impoundment_emission


class MethaneEmission(Emission):
    """ Class for calculating methane emissions from reservoirs """


class NitrousOxideEmission(Emission):
    """ Class for calculating NO2 emissions from reservoirs """
