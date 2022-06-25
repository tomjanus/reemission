"""Module containing custom exceptions."""
from typing import Tuple, Union


class WrongN2OModelError(Exception):
    """Exception raised if attempting to calculate with N2O emission model
       of unknown name/type.

    Attributes:
        permitted_models: tuple with names of recognized N2O emission models.
        message: explanation of the error
    """
    def __init__(
            self,
            permitted_models: Tuple[str, ...],
            message: str = "Model not recognized."):
        self.permitted_models = permitted_models
        self.message = message + \
            f"Permitted models: {self.permitted_models}"
        super().__init__(self.message)


class GasNotRecognizedException(Exception):
    """Exception raised if attempting to select gas of unknown name/type.

    Attributes:
        permitted_gases: tuple with names of recognized emission gases.
        message: explanation of the error
    """
    def __init__(
            self,
            permitted_gases: Tuple[str, ...],
            message: str = "Gas not recognized."):
        self.permitted_gases = permitted_gases
        self.message = message + f"Permitted gases: {self.permitted_gases}"
        super().__init__(self.message)


class WrongAreaFractionsException(Exception):
    """Exception raised if the number of area fractions does not match
    the number of land uses.

    Attributes:
        number_of_fractions: number of area fractions in a list
        number_of_landuses: number of pre-define land uses.
    """
    def __init__(
            self,
            number_of_fractions: int,
            number_of_landuses: int):
        self.message = f"Number of area fractions: {number_of_fractions} " + \
            f"not equal to the number of land uses: {number_of_landuses}."
        super().__init__(self.message)


class WrongSumOfAreasException(Exception):
    """Exception raised if area fractions in a supplied list do not sum to 1.

    Attributes:
        fractions: list of area fractions
        accuracy: acceptable error in the sum of area fractions.
    """
    def __init__(self, fractions: list, accuracy: float):
        self.message = f"Fractions sum up to: {sum(fractions)} " + \
            f"and are not within 1 +/-: {accuracy}."
        super().__init__(self.message)


class TableNotReadException(Exception):
    """Exception raised if a configuration table not found.

    Atrributes:
        table: name of the table or list of names of tables that are not found.
    """
    def __init__(self, table: Union[str, list]):
        if isinstance(table, str):
            self.message = f"Table: {table} not found."
        elif isinstance(table, list):
            self.message = f"Tables: {', '.join(table)} not found."
        else:
            self.message = ""
        super().__init__(self.message)
