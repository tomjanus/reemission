""" Utility functions for multiple modules """
import configparser
from pathlib import Path
from typing import Optional, Type
from enum import Enum, EnumMeta
import yaml


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
    """ Read yaml table given in file_path """
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


def find_enum_index(enum: Type[EnumMeta],
                    to_find: Type[Enum]) -> Optional[int]:
    """ Finds index of an item in an Enum class corresponding to an item
        given in to_find """
    for index, item in enumerate(enum):
        if item == to_find:
            return index
    return None
