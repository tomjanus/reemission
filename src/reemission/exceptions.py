""" Module containing custom exceptions """
from typing import Tuple


class WrongN2OModelError(Exception):
    """Exception raised if attempting to calculate with N2O emission model
       of unknown name/type.

    Attributes:
        available_models: tuple with names of recognized N2O emission models.
        message: explanation of the error
    """
    def __init__(
            self,
            permitted_models: Tuple[str, ...],
            message: str = "Model not recognized."):
        self.permitted_models = permitted_models
        super().__init__(self.message)
