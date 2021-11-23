""" Module providing data and calculations relating to reservoirs """
import logging
from dataclasses import dataclass
from typing import List
from .constants import Landuse

# Set up module logger
log = logging.getLogger(__name__)
# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = 0.01


@dataclass
class Reservoir:
    """ Model of a generic reservoir """
    volume: float  # m3
    inflow_rate: float  # m3/year
    area: float  # Inundated area of the reservoir in km2
    soil_carbon: float  # Mass of C in inundated area in kg/m2
    area_fractions: List[float]  # Fractions of inundated area allocated
    # to specific landuse types given in Landue Enum type

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

    @property
    def residence_time(self) -> float:
        """ Calculate water residence time in years from inflow rate and
            reservoir volume where inflow rate in m3/year is calculated
            on the catchment level """
        return self.volume / self.inflow_rate

    @property
    def discharge(self) -> float:
        """ Discharge from the reservervoir over a long enough time-scale
            in comparison to residence time, is equal to the inflow rate """
        return self.inflow_rate

    @property
    def retention_coeff(self) -> float:
        """ Return empirical retention coefficient for solutes from regression
            with residence time in years being a regressor """
        return 1-1/(1+(0.801*self.residence_time))

    def reservoir_tp(self, inflow_conc: float) -> float:
        """ Calculate reservoir TP concentration in micrograms/L """
        return inflow_conc * (1.0 - self.retention_coeff)
