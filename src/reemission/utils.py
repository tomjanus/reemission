"""Package-wide utility functions."""
import sys
import configparser
from distutils.spawn import find_executable
import pathlib
from typing import Optional, Callable
from enum import Enum, EnumMeta
from packaging import version
import yaml
import reemission
from reemission.exceptions import TableNotReadException


def load_yaml(file_path: pathlib.Path) -> dict:
    """Conditional yaml file loader depending on the installed yaml package
    version.

    Args:
        file: path to the yaml file.
    Returns:
        A dictionary representation of the yaml file.
    """
    with open(file_path, encoding='utf-8') as file_handle:
        if version.parse(yaml.__version__) < version.parse("5.1"):
            loaded_yaml = yaml.load(file_handle)
        else:
            loaded_yaml = yaml.load(file_handle, Loader=yaml.FullLoader)
    return loaded_yaml


def get_package_file(*folders: str) -> pathlib.PosixPath:
    """Imports package data using importlib functionality.

    Args:
        *folers: comma-separated strings representing path to the packaged data
            file.
    Returns:
        A os-indepenent posix path of the data file.
    """
    # Import the package based on Python's version
    if sys.version_info < (3, 9):
        # importlib.resources either doesn't exist or lacks the files()
        # function, so use the PyPI version:
        import importlib_resources
    else:
        # importlib.resources has files(), so use that:
        import importlib.resources as importlib_resources

    pkg = importlib_resources.files("reemission")
    # Append folders to package-wide posix path
    pkg = pathlib.Path.joinpath(pkg, '/'.join(folders))
    return pkg


def read_config(file_path: pathlib.Path) -> configparser.ConfigParser:
    """ Reads the `.ini` file with global configuration parameters and return
    the parsed config object.

    Args:
        file_path: path to the .ini file.

    Returns:
        configparser.ConfigParser object of the .ini file.
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def read_table(file_path: pathlib.Path) -> dict:
    """Reads yaml table from the given YAML file.

    Args:
        file_path: path to the YAML file.
    Returns:
        Dictionary representation of the yaml file if the file exists and no
            errors occured while parsing the file.
    Raises:
        TableNotReadException.
    """
    try:
        stream = open(file_path, "r", encoding="utf-8")
        return yaml.safe_load(stream)
    except FileNotFoundError as exc:
        print(f"File in {file_path} not found.")
        raise TableNotReadException(table=file_path.as_posix()) from exc
    except yaml.YAMLError as exc:
        print(f"File in {file_path} cannot be parsed.")
        raise TableNotReadException(table=file_path.as_posix()) from exc
    finally:
        stream.close()


def find_enum_index(enum: EnumMeta, to_find: Enum) -> int:
    """ Finds index of an item in an Enum class corresponding to an item
    given in to_find.

    Args:
        enum: enum object in which the item to find is stored.
        to_find: key in the enum object.

    Returns:
        Index of the item to find if the item exists in enum.
        Otherwise, returns None.

    Raises:
        KeyError if index could not be found.
    """
    item: str
    for index, item in enumerate(enum):
        if item == to_find:
            return index
    raise KeyError(f"Index {to_find} not found.")


def is_latex_installed() -> bool:
    """Checks if LaTeX is available as a command.

    Returns:
        True if LaTeX is installed and False otherwise.
    """
    if find_executable('latex'):
        return True
    return False


def add_version(fun: Callable) -> Callable:
    """Adds version of the tool to the help heading.

    Params:
        fun: Function to decorate
    Returns:
        Decorated function
    """
    doc = fun.__doc__
    fun.__doc__ = "Package " + reemission.__name__ + " v" + \
        reemission.__version__ + "\n\n" + doc
    return fun
