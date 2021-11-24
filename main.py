""" Main caller of the toolbox functions """
from types import SimpleNamespace
from src.emissions.temperature import MonthlyTemperature
from src.emissions.utils import read_table
from src.emissions.catchment import Catchment
from src.emissions.reservoir import Reservoir
from src.emissions.emissions import CarbonDioxideEmission, NitrousOxideEmission
from src.emissions.constants import (Climate, SoilType, Biome, TreatmentFactor,
                                     LanduseIntensity)
from src.emissions.biogenic import BiogenicFactors


def test_read_yaml():
    table = read_table('./data/emissions/McDowell/landscape_TP_export.yaml')
    table_ns = SimpleNamespace(**table)
    print(table_ns)


def test_emissions():
    # Monthly Temperature Profile
    mt = MonthlyTemperature([10.56, 11.99, 15.46, 18.29, 20.79, 22.09, 22.46,
                             22.66, 21.93, 19.33, 15.03, 11.66])
    # Categorical properties
    biogenic_factors = BiogenicFactors(
        biome=Biome.TROPICALMOISTBROADLEAF,
        climate=Climate.TROPICAL,
        soil_type=SoilType.MINERAL,
        treatment_factor=TreatmentFactor.NONE,
        landuse_intensity=LanduseIntensity.LOW)
    # Area fractions
    catchment_area_fractions = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.01092, 0.11996, 0.867257]
    reservoir_area_fractions = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # Catchment inputs
    catchment_inputs = {'runoff': 1685.5619, 'area': 78203.0,
                        'population': 8463,
                        'area_fractions': catchment_area_fractions,
                        'slope': 8.0, 'precip': 2000.0,
                        'etransp': 400.0, 'soil_wetness': 140.0,
                        'biogenic_factors': biogenic_factors}
    # Reservoir inputs
    reservoir_inputs = {'volume': 7663812, 'area': 0.56470,
                        'area_fractions': reservoir_area_fractions,
                        'soil_carbon': 10.228}
    # Initialize objects
    catchment_1 = Catchment(**catchment_inputs)
    reservoir_1 = Reservoir(**reservoir_inputs,
                            inflow_rate=catchment_1.discharge)
    # Calculate CO2 emissions
    em_co2 = CarbonDioxideEmission(catchment=catchment_1,
                                   reservoir=reservoir_1,
                                   eff_temp=mt.eff_temp(),
                                   p_calc_method='g-res')
    # Calculate CO2 emission profile and CO2 emmision factor
    year_profile = (1, 5, 10, 20, 30, 40, 50, 100)

    co2_emission_profile = em_co2.profile(years=year_profile)
    co2_emission_factor = em_co2.factor(number_of_years=year_profile[-1])
    em_no2 = NitrousOxideEmission(catchment=catchment_1, reservoir=reservoir_1,
                                  model='model 1')
    n2o_emission_factor = em_no2.factor()
    print("CO2 emission profile: ", co2_emission_profile)
    print("CO2 emission factor: ", co2_emission_factor)
    print("N2O emission factor: ", n2o_emission_factor)
    print('Calculate N concentration downstream of the reservoir')
    print(em_no2.nitrogen_downstream_conc())


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    test_emissions()
