""" Main caller of the toolbox functions """
from types import SimpleNamespace
from src.emissions.temperature import MonthlyTemperature
from src.emissions.utils import read_table
from src.emissions.catchment import Catchment
from src.emissions.reservoir import Reservoir
from src.emissions.emissions import (CarbonDioxideEmission,
                                     NitrousOxideEmission,
                                     MethaneEmission)
from src.emissions.constants import (Climate, SoilType, Biome, TreatmentFactor,
                                     LanduseIntensity)
from src.emissions.biogenic import BiogenicFactors


def test_read_yaml():
    table = read_table('./data/emissions/McDowell/landscape_TP_export.yaml')
    table_ns = SimpleNamespace(**table)
    print(table_ns)


def test_emissions():
    """ Calculate emission factors and profiles for all three gases using
        dummy reservoir, catchment and emission input data """
    # 1. DEFINE INPUT DATA
    # Monthly Temperature Profile
    mt = MonthlyTemperature([10.56, 11.99, 15.46, 18.29, 20.79, 22.09, 22.46,
                             22.66, 21.93, 19.33, 15.03, 11.66])
    # Categorical properties
    biogenic_factors = BiogenicFactors(biome=Biome.TROPICALMOISTBROADLEAF,
                                       climate=Climate.TROPICAL,
                                       soil_type=SoilType.MINERAL,
                                       treatment_factor=TreatmentFactor.NONE,
                                       landuse_intensity=LanduseIntensity.LOW)
    # Area fractions
    catchment_area_fractions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01092, 0.11996,
                                0.867257]
    reservoir_area_fractions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # Catchment inputs
    catchment_inputs = {'runoff': 1685.5619, 'area': 78203.0,
                        'population': 8463,
                        'area_fractions': catchment_area_fractions,
                        'slope': 8.0, 'precip': 2000.0, 'etransp': 400.0,
                        'soil_wetness': 140.0,
                        'biogenic_factors': biogenic_factors}
    # Reservoir inputs
    reservoir_inputs = {'volume': 7663812, 'area': 0.56470, 'max_depth': 32.0,
                        'mean_depth': 13.6,
                        'area_fractions': reservoir_area_fractions,
                        'soil_carbon': 10.228}
    # Years vector for calculating emission profiles
    year_profile = (1, 5, 10, 20, 30, 40, 50, 65, 80, 100)
    # 2. INITIALIZE CATCHMENT AND RESERVOIR OBJECTS
    # Initialize objects
    catchment_1 = Catchment(**catchment_inputs)
    reservoir_1 = Reservoir(**reservoir_inputs,
                            inflow_rate=catchment_1.discharge)
    # 3. CALCULATE CO2 EMISSIONS
    em_co2 = CarbonDioxideEmission(catchment=catchment_1,
                                   reservoir=reservoir_1,
                                   eff_temp=mt.eff_temp(),
                                   p_calc_method='g-res')
    co2_emission_profile = em_co2.profile(years=year_profile)
    co2_emission_factor = em_co2.factor(number_of_years=year_profile[-1])
    print("\n")
    print("CO2 emissions:")
    print("---------------------------------------------")
    print("CO2 emission profile: ", ["%.2f" % emission for emission in
                                     co2_emission_profile])
    print("CO2 emission factor: %.2f" % co2_emission_factor)
    print("\n")
    # 4. CALCULATE N2O EMISSIONS
    em_n2o = NitrousOxideEmission(catchment=catchment_1, reservoir=reservoir_1,
                                  model='model 1')
    n2o_emission_profile = em_n2o.profile(years=year_profile)
    n2o_emission_factor = em_n2o.factor()
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
    print('TN downstream load (kg N yr-1): ', "%.1f" % tn_downstream_load)
    print('TN downstream concentration (mg / L): ', "%.4f" % tn_downstream_conc)
    print("\n")
    # 5. CALCULATE CH4 EMISSIONS
    em_ch4 = MethaneEmission(catchment=catchment_1, reservoir=reservoir_1,
                             monthly_temp=mt, mean_ir=4.46)
    ch4_emission_factor = em_ch4.factor()
    ch4_emission_profile = em_ch4.profile(years=year_profile)
    print("CH4 emissions:")
    print("---------------------------------------------")
    print("CH4 emission profile: ", ["%.2f" % emission for emission in
                                     ch4_emission_profile])
    print("CH4 emission factor: %.4f" % ch4_emission_factor)
    print("\n")


if __name__ == '__main__':
    # Run test functions
    test_emissions()
