""" Utility functions for multiple modules """
import sys
import configparser
from pathlib import Path, PosixPath
from typing import Optional, Type
from enum import Enum, EnumMeta
import yaml
import reemission


def load_packaged_data(*folders: str) -> PosixPath:
    """
    Imports package data using importlib functionality
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
    pkg = Path.joinpath(pkg, '/'.join(folders))
    return pkg


def read_config(file_path: Path) -> dict:
    """
    Read the .ini file with global configuration parameters and return
    the parsed config object
    :param file_path: path to the .ini file
    :return config: parsed .ini file
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def read_table(file_path: Path) -> Optional[dict]:
    """Read yaml table given in file_path"""
    try:
        stream = open(file_path, "r")
        return yaml.safe_load(stream)
    except FileNotFoundError as exc:
        print(exc)
        print(f"File in {file_path} not found.")
    except yaml.YAMLError as exc:
        print(exc)
        print(f"File in {file_path} cannot be parsed.")
    finally:
        stream.close()


def find_enum_index(enum: Type[EnumMeta], to_find: Type[Enum]) -> Optional[int]:
    """
    Finds index of an item in an Enum class corresponding to an item
    given in to_find
    """
    for index, item in enumerate(enum):
        if item == to_find:
            return index
    return None


def add_version(fun):
    """
    Add the version of the tool to the help heading.
    :param fun: function to decorate
    :return: decorated function
    """
    doc = fun.__doc__
    fun.__doc__ = "Package " + reemission.__name__ + " v" + \
        reemission.__version__ + "\n\n" + doc
    return fun
