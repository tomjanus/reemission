""" """
import yaml
import toml
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from reemission.utils import (
    get_package_file, load_yaml, read_config_dict, load_toml)


class ConfigLoader:
    """Singleton class for managing application configuration files.

    Supports multiple configuration file formats (YAML, TOML, INI) and
    provides lazy loading, meaning files are only loaded when accessed.

    This class allows registering configuration files for future access
    and ensures that configurations are only loaded once across the application.
    """
    _instance = None

    def __new__(cls):
        """Creates a singleton instance of ConfigLoader."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._configs: Dict[str, Dict] = {}
            cls._instance._loaders: Dict[str, Callable] = {}
        return cls._instance

    def register(self, name: str, file_path: Path, schema_file: Optional[Path] = None) -> None:
        """Registers a configuration file to be lazily loaded later.

        Args:
            name: A unique identifier for the configuration.
            file_path: The path to the configuration file.
        """
        self._loaders[name] = lambda: self._load_from_file(file_path, schema_file)
        
    def override(self, name: str, file_path: Path, schema_file: Optional[Path] = None) -> None:
        """Overrides an existing configuration with a new file path.

        If the configuration has already been loaded, it is reloaded
        from the new file and replaces the previous contents.

        Args:
            name: The name of the configuration to override.
            file_path: Path to the new configuration file.
            schema_file (Optional[Path]): Path to a JSON Schema file for validation (only used for YAML files).

        Raises:
            ValueError: If the file format is not supported.
        """
        config_data = self._load_from_file(file_path, schema_file)
        self._configs[name] = config_data
        self._loaders[name] = lambda: self._load_from_file(file_path)

    def _load_from_file(self, file_path: Path, schema_file: Optional[Path] = None) -> Dict[str, Any]:
        """Loads a configuration file based on its extension.

        Args:
            file_path: Path to the configuration file.
            schema_file (Optional[Path]): Path to a JSON Schema file for validation (only used for YAML files).

        Returns:
            A dictionary representing the parsed configuration.

        Raises:
            ValueError: If the file format is not supported.
        """
        ext = file_path.suffix.lower()

        if ext in ['.yaml', '.yml']:
            # try read_table instead of load_yaml
            return load_yaml(file_path, schema_file)
        elif ext == '.toml':
            return load_toml(file_path)
        elif ext == '.ini':
            parser = configparser.ConfigParser()
            parser.optionxform = str  # preserve case in keys
            parser.read(file_path)
            return {s: dict(parser.items(s)) for s in parser.sections()}
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a loaded configuration.

        If the configuration was registered but not yet loaded,
        it is loaded at this point.

        Args:
            name: The name of the registered configuration.

        Returns:
            The configuration dictionary.

        Raises:
            KeyError: If the configuration has not been registered or loaded.
        """
        if name not in self._configs:
            if name in self._loaders:
                self._configs[name] = self._loaders[name]()  # Lazy load now
            else:
                raise KeyError(f"No configuration registered or loaded with name '{name}'")
        return self._configs[name]

    def update(self, name: str, updates: Dict[str, Any]) -> None:
        """Updates the contents of an already loaded configuration.

        If the configuration was registered but not loaded, it will be loaded now.

        Args:
            name: The name of the configuration to update.
            updates: A dictionary of values to update the configuration with.

        Raises:
            KeyError: If the configuration is not registered or cannot be loaded.
        """
        config = self.get(name)  # This triggers lazy loading if needed
        config.update(updates)

if __name__ == "__main__":
    ...
