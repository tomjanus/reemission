""" """
from functools import reduce
import operator
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, TypeAlias, Tuple
import pprint
from reemission.utils import load_yaml, load_toml, get_package_file

Configurations: TypeAlias = Dict[str, Dict[str, Any]]
Loaders: TypeAlias = Dict[str, Callable]
# Define custom type aliases for trails and key-value pairs representing config
# dictionaries with keys representing variables and values as their values.
Trail: TypeAlias = Tuple[str, ...]
DataDict: TypeAlias = Dict[str, Any]
ConfigData: TypeAlias = Dict[Trail, DataDict]

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
            cls._instance._configs: Configurations = {} # type: ignore
            cls._instance._loaders: Loaders = {} # type: ignore
        return cls._instance
        
    @property
    def config_names(self) -> List[str]:
        """Returns names of config loaders, i.e. supported configuration names.
        Does not equal to self._configs as the _config field only lists registered
        configs. Since configs are lazily loaded, they become available upon first access only -
        otherwise they are not loaded."""
        return list(config._loaders.keys())
    
    def is_registered(self, config_name: str) -> bool:
        """Checks if a configuration with the given name is registered.

        Args:
            name: The name of the configuration to check.

        Returns:
            True if the configuration is registered, False otherwise.
        """
        return config_name in self._loaders
        
    def is_loaded(self, config_name: str) -> bool:
        """Checks if a configuration with the given name is loaded.

        Args:
            name: The name of the configuration to check.

        Returns:
            True if the configuration is loaded, False otherwise.
        """
        return config_name in self._configs
        
    @property
    def config_names(self) -> List[str]:
        """ Returns a list of all registered configuration names.
        This allows users to see which configurations are available
        without loading them.
        Returns:
            A list of configuration names.
        Raises:
            ValueError: If no configurations have been registered.
        """
        if not self._loaders:
            raise ValueError("No configurations have been registered.")
        return [name for name in self._loaders]

    def register(
            self, 
            name: str, 
            file_path: Path,
            schema_file: Optional[Path] = None) -> None:
        """Registers a configuration file to be lazily loaded later.

        Args:
            name: A unique identifier for the configuration.
            file_path: The path to the configuration file.
        """
        self._loaders[name] = lambda: self._load_from_file(file_path, schema_file)
        
    def override(
            self, 
            name: str, 
            file_path: str | Path, 
            schema_file: Optional[Path] = None) -> None:
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
        file_path = Path(file_path)  # Ensure file_path is a Path object
        config_data = self._load_from_file(file_path, schema_file)
        self._configs[name] = config_data
        self._loaders[name] = lambda: self._load_from_file(file_path, schema_file)

    def _load_from_file(
            self, 
            file_path: Path, 
            schema_file: Optional[Path] = None) -> Dict[str, Any]:
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

    def update(
            self, 
            name: str, 
            new_data: Optional[ConfigData] = None, 
            allow_new: bool = False) -> None:
        """Updates the contents of an already loaded configuration.

        If the configuration was registered but not loaded, it will be loaded now.

        Args:
            name: The name of the configuration to update.
            updates: A dictionary of values to update the configuration with.
            allow_new: If True, new keys in updates are added to the configuration.
                       If False, only existing keys are updated.

        Raises:
            KeyError: If the configuration is not registered or cannot be loaded.
            ValueError: If allow_new is False and updates contain new keys.
        """
        #global config
        config = self.get(name)  # This triggers lazy loading if needed
        if new_data is None:
            return
        for trail, key_value_pairs in new_data.items():
            if not isinstance(trail, tuple):
                raise ValueError("Trail must be a tuple of keys.")
            if not isinstance(key_value_pairs, dict):
                raise ValueError("Pairs must be a dictionary of key-value pairs.")
            # Navigate to the correct sub-dictionary using the trail           
            sub_config = reduce(operator.getitem, trail, config)
            common_keys = set(sub_config.keys()) & set(key_value_pairs.keys())
            new_keys = set(key_value_pairs.keys()) - set(sub_config.keys())
            if not isinstance(sub_config, dict):
                return
            for key, value in key_value_pairs.items():
                if key in common_keys or (allow_new and key in new_keys):
                    sub_config[key] = value

if __name__ == "__main__":
    """ """
    custom_config = ConfigLoader()
    custom_config.register(
        "model_config", 
        file_path=get_package_file("config/config.ini"))
    print("Config names: ", custom_config.config_names)
    print("Is `model_config` loaded?: ", custom_config.is_loaded("model_config"))
    print("Is `model_config` registered?: ", custom_config.is_registered("model_config"))
    print('Loading `model_config` by attempting to acces its data...')
    model_config = custom_config.get("model_config")
    print("Is `model_config` loaded?: ", custom_config.is_loaded("model_config"))
    print("Printing the model_config data for CO2 emissions:")
    pprint.pprint(model_config['CARBON_DIOXIDE'])
    print('Accessing fields in the config:')
    updated_params = {("CARBON_DIOXIDE",): { "k1_diff": 0.5, "new_kay": 42}}
    print("k1_diff: ", model_config["CARBON_DIOXIDE"]["k1_diff"])
    print("Updating `model_config` with new values...")
    custom_config.update("model_config", updated_params)
    print("Updated k1_diff: ", model_config["CARBON_DIOXIDE"]["k1_diff"])
    print("Printing updated model_config data for CO2 emissions:")
    pprint.pprint(model_config["CARBON_DIOXIDE"])
