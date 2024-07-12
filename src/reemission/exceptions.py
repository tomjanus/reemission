"""
Module containing custom exceptions.
"""
from typing import Tuple, Union, Iterable


class WrongN2OModelError(Exception):
    """Exception raised if attempting to calculate with N$_2$O emission model
    of unknown name/type.

    Attributes:
        permitted_models (Tuple[str, ...]): Names of recognized N$_2$O emission models.
        message (str): Explanation of the error.
    """
    def __init__(
            self,
            permitted_models: Tuple[str, ...],
            message: str = "Model not recognized."):
        """
        Initializes the WrongN2OModelError with permitted models and an optional message.

        Args:
            permitted_models (Tuple[str, ...]): Permitted N$_2$O emission models.
            message (str, optional): Additional message to include in the exception. Defaults to "Model not recognized.".
        """
        self.permitted_models = permitted_models
        self.message = message + \
            f"Permitted models: {self.permitted_models}"
        super().__init__(self.message)


class GasNotRecognizedException(Exception):
    """Exception raised if attempting to select gas of unknown name/type.

    Attributes:
        permitted_gases (Tuple[str, ...]): Names of recognized emission gases.
        message (str): Explanation of the error.
    """
    def __init__(
            self,
            permitted_gases: Tuple[str, ...],
            message: str = "Gas not recognized."):
        """
        Initializes the GasNotRecognizedException with permitted gases and an optional message.

        Args:
            permitted_gases (Tuple[str, ...]): Permitted emission gases.
            message (str, optional): Additional message to include in the exception. Defaults to "Gas not recognized.".
        """
        self.permitted_gases = permitted_gases
        self.message = message + f"Permitted gases: {self.permitted_gases}"
        super().__init__(self.message)


class WrongAreaFractionsException(Exception):
    """Exception raised if the number of area fractions does not match the number of land uses.

    Attributes:
        number_of_fractions (int): Number of area fractions in a list.
        number_of_landuses (int): Number of pre-defined land uses.
    """
    def __init__(
            self,
            number_of_fractions: int,
            number_of_landuses: int,
            message: str = ""):
        """
        Initializes the WrongAreaFractionsException with the number of fractions and land uses.

        Args:
            number_of_fractions (int): Number of area fractions.
            number_of_landuses (int): Number of land uses.
            message (str, optional): Additional message to include in the exception. Defaults to "".
        """
        self.message = f"Number of area fractions: {number_of_fractions} " + \
            f"not equal to the number of land uses: {number_of_landuses}."
        super().__init__(" ".join([message, self.message]))


class WrongSumOfAreasException(Exception):
    """Exception raised if area fractions in a supplied list do not sum to 1.

    Attributes:
        fractions (list): List of area fractions.
        accuracy (float): Acceptable error in the sum of area fractions.
    """
    def __init__(self, fractions: list, accuracy: float, message: str = ""):
        """
        Initializes the WrongSumOfAreasException with the fractions and accuracy.

        Args:
            fractions (list): List of area fractions.
            accuracy (float): Acceptable error in the sum of area fractions.
            message (str, optional): Additional message to include in the exception. Defaults to "".
        """
        self.message = f"Fractions sum up to: {sum(fractions)} " + \
            f"and are not within 1 +/-: {accuracy}."
        super().__init__(" ".join([message, self.message]))


class TableNotReadException(Exception):
    """Exception raised if a configuration table is not found.

    Attributes:
        table (Union[str, list]): Name of the table or list of names of tables that are not found.
    """
    def __init__(self, table: Union[str, list]):
        """
        Initializes the TableNotReadException with the table name or list of table names.

        Args:
            table (Union[str, list]): The table(s) that could not be read.
        """
        if isinstance(table, str):
            self.message = f"Table: {table} could not be read."
        elif isinstance(table, list):
            self.message = f"Tables: {', '.join(table)} could not be read."
        else:
            self.message = ""
        super().__init__(self.message)


class ConversionMethodUnknownException(Exception):
    """Exception raised if a conversion of an object was not possible because the conversion method supplied is unknown.

    Attributes:
        conversion_method (str): Name of the conversion method.
    """
    def __init__(self, conversion_method: str, available_methods: Iterable[str]):
        """
        Initializes the ConversionMethodUnknownException with the conversion method and available methods.

        Args:
            conversion_method (str): The conversion method that was not recognized.
            available_methods (Iterable[str]): A list of available conversion methods.
        """
        self.message = f"Conversion method {conversion_method} not recognized."
        available_methods_str = ", ".join(available_methods)
        self.message += f"\nAvailable methods: {available_methods_str}"
        super().__init__(self.message)


def append_message(exc: Exception, message: str) -> None:
    """
    Appends a message to the message text of the given exception.

    Args:
        exc (Exception): The exception instance whose message will be appended.
        message (str): The string to append to the exception's message.
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


def replace_message(exc: Exception, message: str) -> None:
    """
    Replaces the exception message with the given message.

    Args:
        exc (Exception): The exception instance whose message will be replaced.
        message (str): The string to replace the exception's message with.
    """
    if exc.args:
        exc.args = (message,) + exc.args[1:]
    else:
        exc.args = (message,)
    # Additionally create a separate message field with exception message
    exc.message = message
