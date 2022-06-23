"""Collection of auxiliary functions not related to any classes or file/data
management."""
import math


def water_density(temp: float) -> float:
    """Calculate water density in kg/m3 as a function of temperature
    in deg C. Eq. A.3 in Praire2021."""
    density = 1000 * (1 - (
        (temp + 288.9414) / (508_929.2 * (temp + 68.12963))) *
        (temp - 3.9863) ** 2)
    return density


def air_density(mean_temp: float) -> float:
    """Calculate air density in kg/m3. Eq. A.1 in Praire2021.

    Args:
        mean_temp: mean temperature in degC of the 4 warmest months in
            a year.
    """
    return 101325.0 / (287.05 * (mean_temp + 273.15))


def cd_factor(wind_speed: float) -> float:
    """Calculate CD coefficient, (-). Eq. A.6 in Praire2021.
    Args:
        wind_speed: reservoir mean wind speed, m/s.
    """
    return 0.001 if wind_speed < 5.0 else 0.000015


def scale_windspeed(wind_speed: float, wind_height: float,
                    new_height: float) -> float:
    """ Calculate wind speed at desired height base on the original wind speed
    at a known height.

    Args:
        wind_speed: known wind speed, m/s
        wind_height: known wind heigt, m
        new_height: new height for which wind speed is calculated.

    Note:
        Default wind_height it 50m.

    Returns:
        Wind speed at new_height, in m/s
    """
    return wind_speed / (1 - math.sqrt(cd_function(wind_speed))/0.4 *
                         math.log10(new_height / wind_height))
