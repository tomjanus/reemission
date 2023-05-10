""" """
from typing import Iterable, Optional


class ColumnsNotFoundError(Exception):
    """Custom exception raised if some columns could not be found in the dataframe"""
    def __init__(self, missing_columns: Iterable[str]):
        missing_columns_str = ", ".join(missing_columns)
        message = f"Missing columns {missing_columns_str}"
        self.message = message
        super().__init__(message)


class FileDoesNotExistError(Exception):
    """Exception raised if file does not exist"""
    def __init__(self, file_name: str, message: Optional[str] = None):
        self.file_name = file_name
        self.message = message or f"File '{file_name}' does not exist."
        super().__init__(self.message)


class ConfigNotFoundException(Exception):
    """Exception raised if config file has not been instantiated or incorrectly
    loaded.

    Attributes:
        message: explanation of the error
    """
    def __init__(
            self,
            message="Config data not present. \n Possible reasons: config file" +
            "not loaded properly."):
        self.message = message
        super().__init__(self.message)


class CompositeModelValidationException(Exception):
    """ """
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(self.msg)