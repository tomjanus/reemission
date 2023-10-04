"""Package-wide utility functions."""
from typing import Union
import sys
import os
import configparser
import pathlib
from functools import wraps
from distutils.spawn import find_executable
import pathlib
import hashlib
import time
from typing import Callable, Union, Tuple, Optional, Dict, Any
from enum import Enum, EnumMeta
from packaging import version
import yaml
import toml
import json
import pandas as pd
import geopandas as gpd
import logging
import jsonschema
import reemission
from reemission.exceptions import TableNotReadException


# Create a logger
logger = logging.getLogger(__name__)


APPLICATION_NAME = "reemission"


SplitPath = Tuple[pathlib.Path, Optional[pathlib.Path]]


def validate(data: Any, schema: Dict) -> None:
    """Validate data in a dictionary format against schema with jsonschema"""
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        print("Validation failed. Data does not conform to the schema.")
        raise jsonschema.exceptions.ValidationError from e


def load_yaml(
        file_path: pathlib.Path, 
        schema_file: Optional[pathlib.Path] = None) -> Dict:
    """Conditional yaml file loader depending on the installed yaml package
    version.

    Args:
        file_path: path to the yaml file
        schema_file: path to json 'jsonschema' file
    Returns:
        A dictionary representation of the yaml file.
    """
    with open(file_path, encoding='utf-8') as file_handle:
        if version.parse(yaml.__version__) < version.parse("5.1"):
            loaded_yaml = yaml.load(file_handle)
        else:
            loaded_yaml = yaml.load(file_handle, Loader=yaml.FullLoader)

    if schema_file:
        schema = load_json(schema_file)
        validate(loaded_yaml, schema)

    return loaded_yaml


def load_json(file_path: pathlib.Path) -> Dict:
    """Load json file"""
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def load_toml(file_path: pathlib.Path) -> Dict:
    """Load toml file"""
    return toml.load(file_path)


def load_shape(path: pathlib.Path) -> gpd.GeoDataFrame:
    """ Opens a shape file using geopandas and returns a GeoDataFrame."""
    return gpd.read_file(path)


def load_geojson(path: pathlib.Path) -> gpd.GeoDataFrame:
    """Opens a geojson file using geopandas and returns a GeoDataFrame."""
    return gpd.read_file(path)


def load_csv(path: pathlib.Path) -> pd.DataFrame:
    """Opens a csv file with tabular data and loads it into pandas
    dataframe."""
    return pd.read_csv(path)


def get_package_file(*folders: str) -> pathlib.Path:
    """Imports package data using importlib functionality.

    Args:
        *folers: comma-separated strings representing path to the packaged data
            file.
    Returns:
        A os-indepenent path of the data file.
    """
    # Import the package based on Python's version
    if sys.version_info < (3, 9):
        # importlib.resources either doesn't exist or lacks the files()
        # function, so use the PyPI version:
        import importlib_resources
    else:
        # importlib.resources has files(), so use that:
        import importlib.resources as importlib_resources

    pkg = importlib_resources.files(APPLICATION_NAME)
    # Append folders to package-wide posix path
    pkg = pathlib.Path.joinpath(pkg, '/'.join(folders))
    return pkg


def get_folder_size(folder_path: Union[pathlib.Path, str]) -> float:
    """Calculates size in bytes for all files in a given folder"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size


def clean_folder(folder_path: Union[pathlib.Path, str]) -> None:
    """Removes all subfolders and files in a given folder"""
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.rmdir(dir_path)


def read_config(
        file_path: Union[str, pathlib.Path]) -> configparser.ConfigParser:
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


def read_table(
            file_path: pathlib.Path, 
            schema_file: Optional[pathlib.Path] = None) -> Dict:
    """Reads yaml table from the given YAML file.

    Args:
        file_path: path to the YAML file.
        schema_file: path to json 'jsonschema' file
    Returns:
        Dictionary representation of the yaml file if the file exists and no
            errors occured while parsing the file.
    Raises:
        TableNotReadException.
    """
    try:
        stream = open(file_path, "r", encoding="utf-8")
        loaded_yaml = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        print(f"File in {file_path} not found.")
        raise TableNotReadException(table=file_path.as_posix()) from exc
    except yaml.YAMLError as exc:
        print(f"File in {file_path} cannot be parsed.")
        raise TableNotReadException(table=file_path.as_posix()) from exc
    finally:
        stream.close()

    if schema_file:
        schema = load_json(schema_file)
        validate(loaded_yaml, schema)

    return loaded_yaml


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


def md5(file_name: Union[str, pathlib.Path], chunk_size: int = 4) -> str:
    """ Generate MD5 checksum of a file
    Args:
        file_name: path to the file for which MD5 sum needs to be calculated
        chunk_size: size of the file chunk to be read (in KB)
    """
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f_handle:
        for chunk in iter(lambda: f_handle.read(chunk_size * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def split_path(input_path: Union[str, pathlib.Path]) -> SplitPath:
    """Split path into the parent directory tree and the file name."""
    if isinstance(input_path, str):
        input_path = pathlib.Path(input_path)
    if not input_path.suffix:
        # Provided input path is a path of directories (assuming each file 
        # needs an extension - not True in Unix but we make this assumption here
        return (input_path, None)
    return (input_path.parent, input_path.name)


def is_directory(input_path: pathlib.Path) -> bool:
    """Check if the path is a directory by checking if split_path returns file
    """
    return (split_path(input_path)[1] is None)


def timeit(func):
    """Wrapper for timing executions of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        total_time = toc - tic
        print(
            f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return wrapper


def save_to_json(
        output_path: Union[str, pathlib.Path], input_dict: Dict) -> int:
    """ Save dictionary to JSON. If folder tree does not exist, create
    one before saving the file. Returns an exit status."""
    # Create output path if output path does not exist
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)
    pathlib.Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    if output_path.suffix:
        if output_path.suffix.lower() == '.json':
            with open(output_path, "w", encoding='utf8') as write_file:
                json.dump(input_dict, write_file, indent=4)
            logger.info("Data saved to %s", output_path.as_posix())
            return os.EX_OK
        logger.error(
            "Output file requires `.json` extension. Data could not be saved.")
        return os.EX_CANTCREAT
    logger.warning(
        "Output path %s does not specify a file. Data could not be saved.", 
        output_path.as_posix())
    return os.EX_CANTCREAT


def strip_double_quotes(input: str) -> str:
    """ """
    return input.replace('"','')


def save_return(output: Dict, save_output: bool=True):
    """Decorator that saves the output of the decorated method to an output dict 
    (usually a global var)
    
    Used for saving internal variables to a global shared variable internals in
    `shared_intern.py`
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Callable:
            # Get the method name
            method_name = func.__name__
            # Call the target method and store its output if save_output is True
            result = func(*args, **kwargs)
            if save_output:
                output[method_name] = result
            # Return the result as usual
            return result
        return wrapper
    return decorator
    
    
def debug_on_exception(func: Callable):
    """Wrapper for debugging functions that raise exception
    Enters debug mode if (some) generic exception is raised
    """
    def wrapper(*args, **kwargs) -> Callable:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import pdb
            print(f"OOpsie daisey, we've raised an exception: {e}")
            pdb.set_trace()

    return wrapper
