""" Project-wide collection of constants."""
from enum import Enum, unique

# Atomic mass (g/mol)
C_MOLAR = 12  # gC/mol
O_MOLAR = 16  # gO/mol
H_MOLAR = 1  # gH/mol
P_MOLAR = 30.97  # gP / mol
N_MOLAR = 14.0  # gN / mol

# Molar mass (g/mol)
CO2_MOLAR = 44
CH4_MOLAR = 16

# Global warming potentials (1 default for CO2) - Source: IPCC 2013
# For a 100-year time horizon.
N2O_GWP100 = 298  # Global Warming Potential of N2O (265–298) over 100 years
# Value of 273 for N2O_GWP100 quoted by epa.gov
CH4_GWP100 = 34  # Global Warming Potential of CH4 (28–36) over 100 years
# For a 20-year time horizon (scaled).
N2O_GWP20 = 500  # CHECK THIS VALUE!
CH4_GWP20 = 86  # Global Warming Potential of CH4 over 20 years


@unique
class Climate(Enum):
    """Enumeration class with climate types"""
    BOREAL = "Boreal"
    SUBTROPICAL = "Subtropical"
    TEMPERATE = "Temperate"
    TROPICAL = "Tropical"


@unique
class Biome(Enum):
    """Enumeration class with biome types"""
    DESERTS = 1
    MEDFORESTS = 2
    MONTANEGRASSLANDS = 3
    TEMPERATEBROADLEAFANDMIXED = 4
    TEMPERATECONIFER = 5
    TEMPERATEGRASSLANDS = 6
    TROPICALDRYBROADFLEAF = 7
    TROPICALGRASSLANDS = 8
    TROPICALMOISTBROADLEAF = 9
    TUNDRA = 10


@unique
class Landuse(Enum):
    """Enumeration class with landuse types"""
    BARE = "bare"
    SNOW_ICE = "snow_ice"
    URBAN = "urban"
    WATER = "water"
    WETLANDS = "wetlands"
    CROPS = "crops"
    SHRUBS = "shrubs"
    FOREST = "forest"


@unique
class LanduseIntensity(Enum):
    """Enumeration class for landuse intensities for calculating
    land cover export coefficients in kg P ha-1 yr-1"""
    LOW = "low_intensity"
    HIGH = "high_intensity"


@unique
class SoilType(Enum):
    """Enumeration class with soil types"""
    MINERAL = "mineral"
    ORGANIC = "organic"


@unique
class TreatmentFactor(Enum):
    """Enumeration type with wastewater treatment classifiations"""
    NONE = "none"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
