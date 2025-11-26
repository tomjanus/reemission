#    This file is part of Re-Emission.
#
#    Re-Emission is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Re-Emission is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Re-Emission.  If not, see <http://www.gnu.org/licenses/>.

""" """
from typing import Iterable, Optional


class ColumnsNotFoundError(Exception):
    """Custom exception raised if some columns could not be found in the dataframe"""
    def __init__(self, missing_columns: Iterable[str]):
        missing_columns_str = ", ".join(missing_columns)
        message = f"Missing columns: {missing_columns_str}"
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
