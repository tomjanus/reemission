""" Module providing data and calculations relating to reservoirs """
import logging
import math
from enum import Enum
from dataclasses import dataclass
from typing import List
from reemission.constants import Landuse, TrophicStatus
from reemission.exceptions import (
    WrongAreaFractionsException,
    WrongSumOfAreasException)

# Set up module logger
log = logging.getLogger(__name__)
# Margin for error by which the sum of landuse fractions can differ from 1.0
EPS = 0.01


@dataclass
class Reservoir:
    """Model of a generic reservoir.

    Atttributes:
        volume: Reservoir volume, m3
        max_depth: Maximum reservoir depth, m
        mean_depth: Mean reservoir depth, m
        inflow_rate: flow of water entering the reservoir, m3/year
        area: Inundated area of the reservoir, km2
        soil_carbon: Mass of C in inundated area, kg/m2
        area_fractions: List of fractions of inundated area allocated
            to specific landuse types given in Landue Enum type, -

    Notes:
        Sum of area fractions should be equal 1 +/- EPS
    """
    volume: float
    max_depth: float
    mean_depth: float
    inflow_rate: float
    area: float
    soil_carbon: float
    area_fractions: List[float]

    def __post_init__(self):
        """Check if the provided list of landuse fractions has the same
        length as the list of landuses.

        Raises:
            WrongAreaFractionsException if number of area fractions in the list
                not equal to the number of land uses.
            WrongSumAreasException if area fractions do not sum to 1 +/-
                acurracy coefficient EPS.
        """
        try:
            assert len(self.area_fractions) == len(Landuse)
        except AssertionError as err:
            raise WrongAreaFractionsException(
                number_of_fractions=len(self.area_fractions),
                number_of_landuses=len(Landuse)) from err
        try:
            assert 1 - EPS <= sum(self.area_fractions) <= 1 + EPS
        except AssertionError as err:
            raise WrongSumOfAreasException(
                fractions=self.area_fractions,
                accuracy=EPS) from err

    @property
    def residence_time(self) -> float:
        """Calculate water residence time from inflow rate and reservoir
        volume where inflow rate is calculated on a catchment level.
        Residence time (or Water Residence Time, WRT) represents the average
        amount of time that a molecule of water spends in a reservoir or lake.

        With assumed units of m3 for volume and m3/year for inflow_rate
        the residence time is given in years."""
        return self.volume / self.inflow_rate

    @property
    def discharge(self) -> float:
        """Return discharge from the reservervoir.

        Over a long enough time-scale in comparison to residence time,
        discharge is equal to the inflow rate. Assumed unit: m3/year."""
        return self.inflow_rate

    @property
    def retention_coeff_emp(self) -> float:
        """Empirical retention coefficient (-) for solutes from
        regression with residence time in years being a regressor."""
        return 1.0 - 1.0 / (1.0 + (0.801 * self.residence_time))

    @property
    def retention_coeff_larsen(self) -> float:
        """Retention coefficient (-) using the model of Larsen and Mercier,
        (1976) for Phosphorus retention. Assumes residence time in years.
        """
        return 1.0 / (1.0 + 1.0 / math.sqrt(self.residence_time))

    def retention_coeff(self, method: str = 'larsen') -> float:
        """Return retention coefficient using the chosen calculation method.

        Args:
            method: Retention coefficient calculation method.
        """
        if method == 'larsen':
            return self.retention_coeff_larsen
        if method in ['emp', 'empirical']:
            return self.retention_coeff_emp
        # Otherwise, use the Larsen and Mercier model
        log.warning('Residence time calculation method %s unknown. ' +
                    'Using the Larsen and Mercier model', method)
        return self.retention_coeff_larsen

    def reservoir_tp(self, inflow_conc: float) -> float:
        """Calculate reservoir TP concentration in micrograms/L.

        Args:
            inflow_conc: TP concentration in the inflow, micrograms/L.
        """
        return inflow_conc * (1.0 - self.retention_coeff)

    def trophic_status(self, inflow_conc: float) -> Enum:
        """Return reservoirs trophic status depending on the influent
        TP concentration.

        Args:
            inflow_conc: TP concentration in the inflow, micrograms/L.
        """
        reservoir_tp = self.reservoir_tp(inflow_conc)
        if reservoir_tp < 10.0:
            return TrophicStatus.OLIGOTROPHIC
        if reservoir_tp < 30.0:
            return TrophicStatus.MESOTROPHIC
        if reservoir_tp < 100.0:
            return TrophicStatus.EUTROPHIC
        # If concentration >= 100.0
        return TrophicStatus.HYPER_EUTROPHIC
