"""
Collection of auxiliary functions used in emission calculations.

.. _G-Res Technical Documentation: https://www.hydropower.org/publications/the-ghg-reservoir-tool-g-res-technical-documentation
.. _Praire2021: https://www.sciencedirect.com/science/article/pii/S1364815221001602
.. _G-Res: https://www.grestool.org/

"""
from typing import Optional, Any, List
import math


def water_density(temp: float) -> float:
    """Calculate water density as a function of temperature.

    This function calculates the water density in kg/m$^3$ as a function of 
    temperature in degrees Celsius using the equation A.3 from Praire2021_.

    Args:
        temp (float): Temperature in degrees Celsius.

    Returns:
        float: Water density, kg/m$^3$.
    """
    density = 1000 * (1 - (
        (temp + 288.9414) / (508_929.2 * (temp + 68.12963))) *
        (temp - 3.9863) ** 2)
    return density


def air_density(mean_temp: float) -> float:
    """Calculate air density.

    This function calculates the air density in kg/m$^3$ using the equation A.1 
    from Praire2021_.

    Args:
        mean_temp (float): Mean temperature in degrees Celsius of the four 
                           warmest months in a year.

    Returns:
        float: Air density, kg/m$^3$.
    """
    return 101325.0 / (287.05 * (mean_temp + 273.15))


def cd_factor(wind_speed: float) -> float:
    """Calculate the CD coefficient.

    This function calculates the CD coefficient using equation A.6 from 
    Praire2021_.

    Args:
        wind_speed (float): Reservoir mean wind speed, m/s.

    Returns:
        float: CD coefficient, --.
    """
    return 1.3E-3 if wind_speed < 5.0 else 1.5E-5


def scale_windspeed(wind_speed: float, wind_height: float,
                    new_height: float) -> float:
    """Scale wind speed to a new height.

    This function calculates the wind speed at a desired height based on the 
    original wind speed at a known height.

    Args:
        wind_speed (float): Known wind speed, m/s.
        wind_height (float): Known wind height, m.
        new_height (float): New height for which wind speed is calculated, m.

    Note:
        Default wind height is 50m.

    Returns:
        float: Wind speed at new_height, m/s.
    """
    return wind_speed / (1 - math.sqrt(cd_factor(wind_speed))/0.4 *
                         math.log10(new_height / wind_height))


def rollout_nested_list(
        nested_list: List[Any],
        out_list: Optional[List] = None) -> List[Any]:
    """Flatten a nested list.

    This function takes a nested list and returns a flattened version of that list.

    Args:
        nested_list (List[Any]): A list containing other lists and/or non-list elements.
        out_list (Optional[List], optional): An empty list to store the flattened output.

    Returns:
        List[Any]: A flattened version of the nested_list.

    Example:
        .. code-block:: Python

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
