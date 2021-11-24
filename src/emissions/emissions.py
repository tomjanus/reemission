""" Module containing classes for the calculation of GHG emissions
    resulting from inundation """
import os
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Optional, Type
from abc import ABC, abstractmethod
from .utils import read_config, read_table
from .constants import Landuse, N_MOLAR, P_MOLAR, O_MOLAR, N2O_GWP100
from .catchment import Catchment
from .reservoir import Reservoir
from .temperature import MonthlyTemperature

# Get relative imports to data
module_dir = os.path.dirname(__file__)
INI_FILE = os.path.abspath(
    os.path.join(module_dir, '..', '..', 'config', 'emissions', 'config.ini'))
TABLES = os.path.abspath(
    os.path.join(module_dir, '..', '..', 'data', 'emissions'))


@dataclass
class Emission(ABC):
    """ Abstract emission class that acts as a base class for all emissions """
    catchment: Type[Catchment]
    reservoir: Type[Reservoir]
    preinund_area: float
    config: dict

    def __init__(self, catchment, reservoir, preinund_area=None,
                 config_file=INI_FILE):
        self.catchment = catchment
        self.reservoir = reservoir  # in km2
        self.config = read_config(config_file)
        if preinund_area is None:
            self.preinund_area = self.__calculate_pre_inund_area()

    def __calculate_pre_inund_area(self) -> float:
        """ Calculate pre inundatation area of a waterbody based on
            the catchment area, using regression """
        return 2.125 * 5.9 * 10**(-3) * (0.01 * self.catchment.area)**0.32

    @abstractmethod
    def profile(self, years: Tuple[int]) -> List[float]:
        """ Abstract method for calculating an emission profile """

    @abstractmethod
    def factor(self, number_of_years: int) -> float:
        """ Abstract method for calculating total emission (factor) """


@dataclass
class CarbonDioxideEmission(Emission):
    """ Class for calculating CO2 emissions from reservoirs """
    eff_temp: float
    p_calc_method: str
    par: Type[SimpleNamespace]
    pre_impoundment_table: dict

    def __init__(self, catchment, reservoir, eff_temp, preinund_area=None,
                 p_calc_method='g-res', config_file=INI_FILE):

        super().__init__(catchment=catchment,
                         reservoir=reservoir,
                         config_file=config_file,
                         preinund_area=preinund_area)

        # Initialise input data specific to carbon dioxide emissions
        self.eff_temp = eff_temp  # EFF temp CO2
        if p_calc_method not in ('g-res', 'mcdowell'):
            p_calc_method = 'g-res'
            print('P calculation method %s unknown. ' % p_calc_method +
                  ' Initializing with default g-res method')
        self.p_calc_method = p_calc_method
        # Read the tables
        self.par = self.__initialize_parameters_from_config(
            ['c_1', 'age', 'temp', 'resArea', 'soilC', 'ResTP', 'calc',
             'conv_coeff'])
        self.pre_impoundment_table = read_table(
            os.path.join(TABLES, 'Carbon_Dioxide', 'pre-impoundment.yaml'))

    def __initialize_parameters_from_config(self, list_of_constants: list) \
            -> SimpleNamespace:
        """ Read constants (parameters) from config file """
        const_dict = {name: self.config.getfloat('CARBON_DIOXIDE', name)
                      for name in list_of_constants}
        return SimpleNamespace(**const_dict)

    @property
    def reservoir_tp(self) -> float:
        """ Return reservoir total phosphorus concentration """
        reservoir_tp = self.reservoir.reservoir_tp(
            inflow_conc=self.catchment.median_inflow_p(
                method=self.p_calc_method))
        return reservoir_tp

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
                math.log10(self.reservoir.area) * self.par.resArea +
                self.reservoir.soil_carbon * self.par.soilC +
                math.log10(self.reservoir_tp) * self.par.ResTP) * \
                (1 - (self.preinund_area/self.reservoir.area))
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
            math.log10(self.reservoir.area) * self.par.resArea +
            self.reservoir.soil_carbon * self.par.soilC +
            math.log10(self.reservoir_tp) * self.par.ResTP) * \
            (1 - (self.preinund_area/self.reservoir.area)) * \
            ((100**(self.par.calc+1) -
              0.5**(self.par.calc+1)) / ((self.par.calc+1)*(100-0.5)))
        return flux

    def __net_total_emission(self, number_of_years: int) -> float:
        """
        Calculate net total CO2 emissions, i.e. gross - non anthropogenic
        (in g CO2eq m-2 yr-1) from a reservoir over a number of years
        given in number_of_years
        """
        return (self.__gross_total_emission() -
                self.__fluxes_per_year(years=(number_of_years,)))

    def __pre_impoundment_emission(self) -> float:
        """
        Calculate CO2 emissions  g CO2eq m-2 yr-1 from the inundated area
        prior to impoundment
        """
        __list_of_landuses = list(Landuse.__dict__['_member_map_'].values())
        climate = self.catchment.biogenic_factors.climate
        soil_type = self.catchment.biogenic_factors.soil_type
        emissions = []
        for landuse, fraction in zip(__list_of_landuses,
                                     self.reservoir.area_fractions):
            # Area in ha allocated to each landuse
            area_landuse = 100 * self.reservoir.area * fraction
            coeff = self.pre_impoundment_table.get(
                climate.value, {}).get(
                    soil_type.value, {}).get(
                        landuse.value, 0)
            emissions.append(area_landuse * coeff)
        # Total emission in t CO2-C /yr
        tot_emission = sum(emissions)
        # Total emission in g CO2eq m-2 yr-1
        return tot_emission/self.reservoir.area

    def profile(self,
                years: Tuple[int] = (1, 5, 10, 20, 30, 40, 50, 100)) \
            -> List[float]:
        """ Calculate CO2 emissions for a list of years given as an argument
        Flux at year x age - pre-impoundment emissions - non-anthropogenic
        emissions """
        pre_impoundment = self.__pre_impoundment_emission()
        integrated_emission = self.__fluxes_per_year((years[-1],))
        fluxes_profile = self.__fluxes_per_year(years)
        final_profile = [flux - integrated_emission - pre_impoundment for
                         flux in fluxes_profile]
        return final_profile

    def factor(self, number_of_years: int = 100) -> float:
        """ Overall integrated emissions for lifetime, taken by default
            as 100 yrs, unit:  g CO2eq m-2 yr-1 """
        net_total_emission = self.__net_total_emission(number_of_years)
        pre_impoundment_emission = \
            self.__pre_impoundment_emission()
        return net_total_emission - pre_impoundment_emission


@dataclass
class MethaneEmission(Emission):
    """ Class for calculating methane emissions from reservoirs """
    monthly_temp: Type[MonthlyTemperature]
    mean_ir: float  # mean InfraRed Radiation in kwh-1d-1

    def __init__(self, catchment, reservoir, monthly_temp, mean_ir,
                 preinund_area=None,
                 config_file=INI_FILE):
        self.monthly_temp = monthly_temp
        self.mean_ir = mean_ir
        super().__init__(catchment=catchment,
                         reservoir=reservoir,
                         config_file=config_file,
                         preinund_area=preinund_area)
        # List of parameters required for CH4 emission calculations
        par_list = ['int_diff', 'age_diff', 'littoral_diff', 'eff_temp_CH4',
                    'int_ebull', 'littoral_ebull', 'irrad_ebull', 'int_degas',
                    'tw_degas', 'ch4_diff', 'ch_diff_age_term', 'conv_coeff']
        # Read the parameters from config
        self.par = self.__initialize_parameters_from_config(par_list)

    def __initialize_parameters_from_config(self, list_of_constants: list) \
            -> SimpleNamespace:
        """ Read constants (parameters) from config file """
        const_dict = {name: self.config.getfloat('METHANE', name)
                      for name in list_of_constants}
        return SimpleNamespace(**const_dict)

    def thermocline_depth(self, wind_speed: float,
                          wind_height: float = 50) -> float:
        """ Calculate thermocline depth required for the calculation of CH4
            degassing. Assumes that the surface water temperature is equal to
            the mean monthly air temperature from 4 warmest months in the year
        """
        def water_density(temp: float) -> float:
            """ Calculate water density in kg/m3 as a function of temperature
                in deg C """
            density = 1000 * (
                1-((temp+288.9414)/(508929.2*(temp+68.12963)))*(temp-3.9863)**2)
            return density

        # Calculate CD coefficient
        cd_coeff = 0.001 if wind_speed < 5.0 else 0.000015
        # Calculate air density (units?)
        air_density = 101325.0/(
            287.05*(self.monthly_temp.mean_warmest(number_of_months=4)+273.15))
        wind_at_10m = wind_speed / (1 - 10/4 * math.log10(10.0/wind_height) *
                                    math.sqrt(cd_coeff))
        if self.monthly_temp.coldest > 1.4:
            hypolimnion_temp = (0.6565 * self.monthly_temp.coldest) + 10.7
        else:
            hypolimnion_temp = (0.2345 * self.monthly_temp.coldest) + 10.11
        hypolimnion_density = water_density(temp=hypolimnion_temp)
        epilimnion_temp = self.monthly_temp.mean_warmest(number_of_months=4)
        epilimnion_density = water_density(temp=epilimnion_temp)
        # Find thermocline depth in metres
        aux_var_1 = cd_coeff * air_density * wind_at_10m
        aux_var_2 = 9.80665 * (hypolimnion_density - epilimnion_density)
        aux_var_3 = math.sqrt(self.reservoir.area*10**6)
        depth = 2 * math.sqrt(aux_var_1/aux_var_2) * math.sqrt(aux_var_3)
        return depth

    def ebullition(self):
        """ Calculate CH4 emission in g CO2eq m-2 yr-1 through ebullition
            Ebullition fluxes are not time-dependent, hence no emission profile
            is calculated """
        # Bathymetric shape (-)
        g_bath_shape = (self.reservoir.max_depth/self.reservoir.mean_depth)-1.0
        # Percentage of surface area that is littoral (near the shore)
        littoral_perc = (1-(1-(3/self.reservoir.max_depth))**g_bath_shape)*100
        # Calculate CH4 emission in mg CH4-C m-2 d-1
        emission_in_ch4 = 10**(
            self.par.int_ebull +
            self.par.littoral_ebull*math.log10(littoral_perc/100) +
            365*self.par.irrad_ebull*self.mean_ir/30.4)
        # Calculate CH4 emission in g CO2eq m-2 yr-1
        emission_in_co2 = emission_in_ch4 * self.par.conv_coeff
        return emission_in_co2


    def factor(self, number_of_years: int = 666) -> float:
        return 0.0

    def profile(self,
                years: Tuple[int] = (1, 5, 10, 20, 30, 40, 50, 100)) \
            -> List[float]:
        return [0] * len(years)


@dataclass
class NitrousOxideEmission(Emission):
    """ Class for calculating NO2 emissions from reservoirs. Provides option to
        calculate the emission using two alternative methods (models) """
    model: str

    def __init__(self, catchment, reservoir, preinund_area=None,
                 config_file=INI_FILE, model='model 1'):
        if model not in ('model 1', 'model 2'):
            print('Model %s unknown. ' % model +
                  'Initializing with default model 1')
            model = 'model 1'
        self.model = model
        super().__init__(catchment=catchment,
                         reservoir=reservoir,
                         config_file=config_file,
                         preinund_area=preinund_area)

    def total_to_unit(self, emission: float) -> float:
        """ Convert emission from kgN yr-1 to mmolN/m^2/yr """
        return emission / N_MOLAR / self.reservoir.area

    def unit_to_total(self, unit_emission: float) -> float:
        """ Convert emission from mmolN/m^2/yr to kg yr-1 """
        return unit_emission * self.reservoir.area * N_MOLAR

    def tn_fixation_load(self) -> float:
        """ Calculate total N internal fixation load following the method
            described in Maarva et al (2018)
            --------------------------------------------------------------
            Total N fixation depends on water residence time in the reservoir
            and molar TN:TP stoichiometry. It is formulated as the % of the
            riverine inflow TN load using the following formula:
            tn_fix (%) = [ 37.2 / (1 + exp(0.5 * tn_tp_ratio – 6.877))  ] * mu
            where:
            mu = erf ((residence_time - 0.028) / 0.04), with residence_time
                given in years
            Molar weights of P and N are as follows:
            * P_molar = 30.97 gP / mole
            * N_molar = 14 gN / mole
            --------------------------------------------------------------
            To account for uncertainties in the tn_fix estimates, a normal
            distribution with standard deviation of +/-10% was assumed
            around the predict tn_fix values (Akbarzahdeh 2019)
            --------------------------------------------------------------
        """
        tp_load_annual = self.catchment.phosphorus_load()  # kg P / yr
        tn_load_annual = self.catchment.nitrogen_load()  # kg N / yr
        mu_coeff = max(0, math.erf((self.reservoir.residence_time-0.028)/0.04))
        #  molar ratio of inflow TP and TN loads (-)
        tn_tp_ratio = (tn_load_annual/N_MOLAR) / (tp_load_annual/P_MOLAR)
        tn_fix_percent = (37.2/(1+math.exp(0.5*tn_tp_ratio-6.877))) * mu_coeff
        # Calculate total internal N fixation in kg/yr
        return 0.01 * tn_fix_percent * tn_load_annual

    def factor(self, number_of_years: int = 666) -> float:
        """ Return N2O emission in gCO2eq/m2/yr. N2O emissions are not
            calculated over a defined time horizon as e.g. CO2. Thus,
            the time horizon for N2O is given the number of the beast """
        if self.model == "model 1":
            return self.__n2o_emission_m1_co2()
        if self.model == "model 2":
            return self.__n2o_emission_m2_co2()
        return None

    def profile(self,
                years: Tuple[int] = (1, 5, 10, 20, 30, 40, 50, 100)) \
            -> List[float]:
        """ Return N2O emission profile for the years defined in parameer
            years. Only done for the purpose of keeping consistency with other
            emissions, since N2O does not have an emission profile. Thus,
            the returned profile is a straight line with values equal to
            the N2O emission factor """
        return [self.factor()] * len(years)

    def __n2o_emission_m1_co2(self) -> float:
        """ Calculate N2O emission in gCO2eq m-2 yr-1 according to model 1 """
        # 1. Calculate total N2O emission (kgN yr-1)
        total_n2o_emission = self.__n2o_denitrification_m1() + \
            self.__n2o_nitrification_m1()
        # 2. Calculate unit total N2O emission in mmolN/m^2/yr
        unit_n2o_emission = self.total_to_unit(total_n2o_emission)
        # 3. Calculate emission in gCO2eq/m2/yr
        total_n2o = N_MOLAR * (1+O_MOLAR/(2*N_MOLAR)) * N2O_GWP100 * \
            unit_n2o_emission * 10**(-3)
        return total_n2o

    def __n2o_emission_m2_co2(self) -> float:
        """ Calculate N2O emission in gCO2eq m-2 yr-1 according to model 2 """
        total_n2o = N_MOLAR * (1+O_MOLAR/(2*N_MOLAR)) * N2O_GWP100 * \
            self.__unit_n2o_emission_m2() * 10**(-3)
        return total_n2o

    def __n2o_denitrification_m1(self) -> float:
        """ Calculate N2O emission (kgN yr-1) from denitrification using
            Model 1
            0.009 * [tn_catchment_load + tn_fixation_load] *
                [0.3833 * erf(0.4723 * residence time(yrs))]
        """
        n2o_emission_den = 0.009 * (
            self.catchment.nitrogen_load() + self.tn_fixation_load()) * \
            (0.3833*math.erf(0.4723*self.reservoir.residence_time))
        return n2o_emission_den

    def __n2o_nitrification_m1(self) -> float:
        """ Calculate N2O emission (kgN yr-1) from nitrification using
            Model 1
            0.009 * [tn_catchment_load + tn_fixation_load] *
                [0.5144 * erf(0.3692 * water residence time(yrs))]
        """
        n2o_emission_nitr = 0.009 * (
            self.catchment.nitrogen_load() + self.tn_fixation_load()) * \
            (0.5144*math.erf(0.3692*self.reservoir.residence_time))
        return n2o_emission_nitr

    def __n2o_emission_m2_n(self) -> float:
        """ Calculate total N2O emission (kgN yr-1) using Model 2
            --------------------------------------------------------
            From an overall relation derived from N2O emissions
            computed as the sum of two EF terms: N2O derived from
            denitrification, and N2O derived from Nitrification.
            This approach differs from N2OA above in that the derivation of the
            equation below included mechanisms to account for N2O saturation
            state with respect to gaseous emissions (effectively not all N2O
            produced is assumed to be evaded), and for internal consumption of
            N2O produced by denitrification, which increases as a function of
            water residence time.
        """
        n2o_emission = self.catchment.nitrogen_load() * (
            0.002277 * math.erf(1.63*self.reservoir.residence_time))
        return n2o_emission

    def __unit_n2o_emission_m2(self) -> float:
        """ Calculate unit total N2O emission in mmolN/m^2/yr using Model 2 """
        return self.total_to_unit(self.__n2o_emission_m2_n())

    def __n2o_denitrification_m2(self) -> float:
        """ Calculate N2O emission from denitrification in kgN/yr using
            Model 2
        """
        # Calculate unit N2O emission from denitfication in mmol N m-2 yr-1
        unit_n2o_denitrification = 0.7789*math.exp(-((
            self.reservoir.residence_time + 1.366)/2.751))**2 * \
            self.__unit_n2o_emission_m2()
        # Return N2O emission in kgN/yr
        return self.unit_to_total(unit_n2o_denitrification)

    def __n2o_nitrification_m2(self) -> float:
        """ Calculate N2O emission from nitrification in kgN/yr using
            Model 2
        """
        unit_n2o_nitrification = self.__unit_n2o_emission_m2() - \
            self.total_to_unit(self.__n2o_denitrification_m2())
        # Return N2O emission in kgN/yr
        return self.unit_to_total(unit_n2o_nitrification)

    # Additional methods calculating effluent nitrogen load and concentration
    # from the reservoir associated with the calculated N2O emission
    def nitrogen_downstream_load(self) -> float:
        """ Calculate downstream TN load in kg N yr-1 """
        # 1. Calculate TN burial as a factor of input TN
        tn_burial_factor = 0.51 * math.erf(
            0.4723 * self.reservoir.residence_time)
        # 2. Calculate TN denitrification as a factor of input TN
        tn_denitr_factor = 0.3833 * math.erf(
            0.4723 * self.reservoir.residence_time)
        # 3. Calculate TN loading (catchment + fixation) in kg N yr-1
        tn_loading = self.catchment.nitrogen_load() + \
            self.tn_fixation_load()
        # 4. Calculate TN burial in kg N yr-1
        tn_burial = tn_burial_factor * tn_loading
        # 5. Calculate TN denitrification in kg N yr-1
        tn_denitr = tn_denitr_factor * tn_loading
        # 6. Calculate TN downstream load in kg N yr-1
        tn_downstream_load = tn_loading - tn_burial - tn_denitr
        return tn_downstream_load

    def nitrogen_downstream_conc(self) -> float:
        """ Calculate downstream TN concentration in mg / L """
        return 1e02 * self.nitrogen_downstream_load()/(
            self.catchment.area*self.catchment.runoff)
