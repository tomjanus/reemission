"""Module containing custom exceptions."""
from typing import Tuple


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
        self.message = message + \
            f"Permitted gases: {self.permitted_gases}"
        super().__init__(self.message)
