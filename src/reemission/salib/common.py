""" Utility functions for reading/writing files.
Includes:
- `load_yaml`: Function to read a YAML file and return its contents as a dictionary.
               This function is used for loading configuration files in Re-Emission.
"""
import pathlib
from typing import Dict, Any
import yaml


def load_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    """Read the params.yaml into a Python dict."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix != '.yaml':
        raise ValueError(f"Expected a YAML file, got: {path.suffix}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data