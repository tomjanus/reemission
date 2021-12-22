""" Module with simulation/calculation facilities for calculating GHG
    emissions for man-made reservoirs """
from dataclasses import dataclass, field
from typing import Type
from .input import Inputs
from .temperature import MonthlyTemperature
from .catchment import Catchment
from .reservoir import Reservoir
from .emissions import (CarbonDioxideEmission,
                        NitrousOxideEmission,
                        MethaneEmission)


@dataclass
class EmissionModel:
    """ Calculates emissions for a set of data provided in a dictionary
        format """
    outputs: float = field(init=False)

    def calculate(self, inputs: Type[Inputs]) -> None:
        """ Calculate emissions for a number of variables defined in config """
        for model_input in inputs.inputs:
            monthly_temp = MonthlyTemperature(model_input.monthly_temps)
            reservoir_data = model_input.reservoir_data
            catchment_data = model_input.catchment_data
            emission_factors = model_input.emission_factors
            year_vector = model_input.year_vector
            catchment = Catchment(**catchment_data)
            reservoir = Reservoir(**reservoir_data,
                                  inflow_rate=catchment.discharge)
            if "co2" in emission_factors:
                em_co2 = CarbonDioxideEmission(
                    catchment=catchment, reservoir=reservoir,
                    eff_temp=monthly_temp.eff_temp(),
                    p_calc_method='g-res')
                co2_emission_profile = em_co2.profile(years=year_vector)
                co2_emission_factor = em_co2.factor(
                    number_of_years=year_vector[-1])
                print("\n")
                print(model_input.name)
                print("CO2 emissions:")
                print("---------------------------------------------")
                print("CO2 emission profile: ",
                      ["%.2f" % emission for emission in co2_emission_profile])
                print("CO2 emission factor: %.2f" % co2_emission_factor)
                print("\n")
            if "n2o" in emission_factors:
                em_n2o = NitrousOxideEmission(
                    catchment=catchment, reservoir=reservoir, model='model 1')
                n2o_emission_profile = em_n2o.profile(years=year_vector)
                n2o_emission_factor = em_n2o.factor()
                print(model_input.name)
                print("N2O emissions:")
                print("---------------------------------------------")
                print("N2O emission profile: ", ["%.2f" % emission for emission in
                                                 n2o_emission_profile])
                print("N2O emission factor: %.2f" % n2o_emission_factor)
                print("\n")
                print('Calculate N concentration downstream of the reservoir')
                print("---------------------------------------------")
                tn_downstream_load = em_n2o.nitrogen_downstream_load()
                tn_downstream_conc = em_n2o.nitrogen_downstream_conc()
                print('TN downstream load (kg N yr-1): ', "%.1f" %
                      tn_downstream_load)
                print('TN downstream concentration (mg / L): ', "%.4f" %
                      tn_downstream_conc)
                print("\n")
            if "ch4" in emission_factors:
                em_ch4 = MethaneEmission(
                    catchment=catchment, reservoir=reservoir,
                    monthly_temp=monthly_temp, mean_ir=4.46)
                ch4_emission_factor = em_ch4.factor()
                ch4_emission_profile = em_ch4.profile(years=year_vector)
                print(model_input.name)
                print("CH4 emissions:")
                print("---------------------------------------------")
                print("CH4 emission profile: ",
                      ["%.2f" % emission for emission in
                       ch4_emission_profile])
                print("CH4 emission factor: %.4f" % ch4_emission_factor)
                print("\n")
