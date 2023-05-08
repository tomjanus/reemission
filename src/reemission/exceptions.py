"""Module containing custom exceptions."""
from typing import Tuple, Union, Iterable


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
            number_of_landuses: int,
            message: str = ""):
        self.message = f"Number of area fractions: {number_of_fractions} " + \
            f"not equal to the number of land uses: {number_of_landuses}."
        super().__init__(" ".join([message, self.message]))


class WrongSumOfAreasException(Exception):
    """Exception raised if area fractions in a supplied list do not sum to 1.

    Attributes:
        fractions: list of area fractions
        accuracy: acceptable error in the sum of area fractions.
    """
    def __init__(self, fractions: list, accuracy: float, message: str = ""):
        self.message = f"Fractions sum up to: {sum(fractions)} " + \
            f"and are not within 1 +/-: {accuracy}."
        super().__init__(" ".join([message, self.message]))


class TableNotReadException(Exception):
    """Exception raised if a configuration table not found.

    Atrributes:
        table: name of the table or list of names of tables that are not found.
    """
    def __init__(self, table: Union[str, list]):
        if isinstance(table, str):
            self.message = f"Table: {table} could not be read."
        elif isinstance(table, list):
            self.message = f"Tables: {', '.join(table)} could not be read."
        else:
            self.message = ""
        super().__init__(self.message)


class ConversionMethodUnknownException(Exception):
    """Exception raised if a conversion of an object was not possible because
    the conversion method supplied is unknown.

    Atrributes:
        conversion_method: name of the conversion method
    """
    def __init__(self, conversion_method: str, available_methods: Iterable[str]):
        self.message = f"Conversion method {conversion_method} not recognized."
        available_methods_str = ", ".join(available_methods)
        self.message += f"\nAvailable methods: {available_methods_str}"
        super().__init__(self.message)


def append_message(exc: Exception, message: str):
    """
    Appends `message` to the message text of the exception `e`.

    Parameters
    ----------
    exc: Exception
        An exception instance whose `args` will be modified to include `message`.

    message: str
        The string to append to the message.
    """
    # Write message to arguments
    if exc.args:
        # Exception was raised with arguments
        exc.args = (str(exc.args[0]) + message,) + exc.args[1:]
    else:
        exc.args = (message,)
    # Additionally create a separate message field with exception message or
    # add message to the existing text if message field already exists
    try:
        exc.message = exc.message + message
    except AttributeError:
        exc.message = message


def replace_message(exc: Exception, message: str):
    """
    Replaces the exception message with `message`.

    Parameters
    ----------
    exc: Exception
        An exception instance whose `args` will be modified to be `message`.

    message: str
        The string to replace the exception message with.
    """
    if exc.args:
        exc.args = (message,) + exc.args[1:]
    else:
        exc.args = (message,)
    # Additionally create a separate message field with exception message
    exc.message = message
