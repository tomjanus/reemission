"""Collection of auxiliary functions.

The functions are not related to any classes or file/data management.
"""
from typing import Optional, Any, List
import math


def water_density(temp: float) -> float:
    """Calculate water density in kg/m3 as a function of temperature
    in deg C. Eq. A.3 in Praire2021.
    """
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
    return 1.3E-3 if wind_speed < 5.0 else 1.5E-5


def scale_windspeed(wind_speed: float, wind_height: float,
                    new_height: float) -> float:
    """Calculate wind speed at desired height base on the original wind speed
    at a known height.

    Parameters:
        wind_speed: known wind speed, m/s
        wind_height: known wind heigt, m
        new_height: new height for which wind speed is calculated.

    Note:
        Default wind_height it 50m.

    Returns:
        Wind speed at new_height, in m/s
    """
    return wind_speed / (1 - math.sqrt(cd_factor(wind_speed))/0.4 *
                         math.log10(new_height / wind_height))


def rollout_nested_list(
        nested_list: List[Any],
        out_list: Optional[List] = None) -> List[Any]:
    """
    This function takes a nested list and returns a flattened version of that list.

    Parameters:
        nested_list: A list containing other lists and/or non-list elements.
        out_list (optional): An empty list to store the flattened output.
    Returns:
        A flattened version of the nested_list.

    Example Usage:

    nested_list = [1, [2, 3], 4, [[5, 6], 7]]
    flattened_list = rollout_nested_list(nested_list)
    print(flattened_list) # prints [1, 2, 3, 4, 5, 6, 7]
    """
    if out_list is None:
        out_list = []
    while nested_list:
        item = nested_list.pop(0)
        if not isinstance(item, List):
            out_list.append(item)
        else:
            rollout_nested_list(item, out_list)
    return out_list
