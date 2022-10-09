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
    """Climate types."""
    BOREAL = "Boreal"
    SUBTROPICAL = "Subtropical"
    TEMPERATE = "Temperate"
    TROPICAL = "Tropical"


@unique
class Biome(Enum):
    """Biome types."""
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
    """Catchment landuse types."""
    BARE = "bare"
    SNOW_ICE = "snow_ice"
    URBAN = "urban"
    WATER = "water"
    WETLANDS = "wetlands"
    CROPS = "crops"
    SHRUBS = "shrubs"
    FOREST = "forest"
    NODATA = "no data"


@unique
class LanduseIntensity(Enum):
    """Landuse intensities for calculating land cover export coefficients
    in kgP/ha/yr."""
    LOW = "low_intensity"
    HIGH = "high_intensity"


@unique
class SoilType(Enum):
    """Soil types."""
    MINERAL = "mineral"
    ORGANIC = "organic"
    NODATA = "no-data"


@unique
class TreatmentFactor(Enum):
    """Wastewater treatment classifiations."""
    NONE = "no treatment"
    PRIMARY = "primary (mechanical)"
    SECONDARY = "secondary biological treatment"
    TERTIARY = "tertiary"  # E.g. P - stripping


@unique
class TrophicStatus(Enum):
    """Reservoir/Lake trophic status classifications."""
    OLIGOTROPHIC = "oligotrophic"
    MESOTROPHIC = "mesotrophic"
    EUTROPHIC = "eutrophic"
    HYPER_EUTROPHIC = "hyper-eutrophic"
