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
