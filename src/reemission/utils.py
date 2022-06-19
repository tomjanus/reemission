""" Utility functions for multiple modules """
import sys
import configparser
from packaging import version
from pathlib import Path, PosixPath
from typing import Optional
from enum import Enum, EnumMeta
import yaml
import reemission


def load_yaml(file: str) -> str:
    """
    Conditional yaml file loader depending on the installed yaml package
    version.
    """
    with open(file, encoding='utf-8') as file_handle:
        if version.parse(yaml.__version__) < version.parse("5.1"):
            loaded_yaml = yaml.load(file_handle)
        else:
            loaded_yaml = yaml.load(file_handle, Loader=yaml.FullLoader)
    return loaded_yaml


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


def read_config(file_path: Path) -> configparser.ConfigParser:
    """
    Read the .ini file with global configuration parameters and return
    the parsed config object
    :param file_path: path to the .ini file
    :return config: configparser.ConfigParser object of the .ini file
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def read_table(file_path: Path) -> Optional[dict]:
    """Read yaml table given in file_path"""
    try:
        stream = open(file_path, "r", encoding="utf-8")
        return yaml.safe_load(stream)
    except FileNotFoundError as exc:
        print(exc)
        print(f"File in {file_path} not found.")
        return None
    except yaml.YAMLError as exc:
        print(exc)
        print(f"File in {file_path} cannot be parsed.")
        return None
    finally:
        stream.close()


def find_enum_index(
        enum: EnumMeta,
        to_find: Enum) -> Optional[int]:
    """
    Finds index of an item in an Enum class corresponding to an item
    given in to_find
    """
    item: str
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
