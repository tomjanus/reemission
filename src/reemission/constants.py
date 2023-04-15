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
class Climate(str, Enum):
    """Climate types."""
    BOREAL = "boreal"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    TROPICAL = "tropical"


@unique
class BuildStatus(str, Enum):
    """Build status of a dam"""
    EXISTING = "existing"
    PLANNED = "planned"


@unique
class Biome(str, Enum):
    """Biome types."""
    DESERTS = "deserts"
    MEDFORESTS = "mediterreanan forests"
    MONTANEGRASSLANDS = "montane grasslands"
    TEMPERATEBROADLEAFANDMIXED = "temperate broadleaf and mixed"
    TEMPERATECONIFER = "temperate coniferous"
    TEMPERATEGRASSLANDS = "temperate grasslands"
    TROPICALDRYBROADFLEAF = "tropical dry broadleaf"
    TROPICALGRASSLANDS = "tropical grasslands"
    TROPICALMOISTBROADLEAF = "tropical moist broadleaf"
    TUNDRA = "tundra"


@unique
class Landuse(str, Enum):
    """Catchment landuse types."""
    BARE = "bare"
    SNOW_ICE = "snow and ice"
    URBAN = "urban"
    WATER = "water"
    WETLANDS = "wetlands"
    CROPS = "crops"
    SHRUBS = "shrubs"
    FOREST = "forest"
    NODATA = "no data"


@unique
class LanduseIntensity(str, Enum):
    """Landuse intensities for calculating land cover export coefficients
    in kgP/ha/yr."""
    LOW = "low intensity"
    HIGH = "high intensity"


@unique
class SoilType(str, Enum):
    """Soil types."""
    MINERAL = "mineral"
    ORGANIC = "organic"
    NODATA = "no data"


@unique
class TreatmentFactor(str, Enum):
    """Wastewater treatment classifiations."""
    NONE = "no treatment"
    PRIMARY = "primary (mechanical)"
    SECONDARY = "secondary biological treatment"
    TERTIARY = "tertiary"  # E.g. P - stripping


@unique
class TrophicStatus(str, Enum):
    """Reservoir/Lake trophic status classifications."""
    OLIGOTROPHIC = "oligotrophic"
    MESOTROPHIC = "mesotrophic"
    EUTROPHIC = "eutrophic"
    HYPER_EUTROPHIC = "hyper eutrophic"


@unique
class AssetConstructionStage(Enum):
    """Construction stages of a reservoir"""
    EXISTING = auto()
    FUTURE = auto()
