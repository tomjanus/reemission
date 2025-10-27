""" """
from typing import NamedTuple, Optional, Union
import pathlib
from rich.console import Console
from rich.text import Text

_has_registered = False

class ConfigEntry(NamedTuple):
    """Represents a single configuration entry within the reemission package."""
    config_subdir: str
    config_filename: str
    schema_subdir: Optional[str] = None
    schema_filename: Optional[str] = None

# -------------------------------------------------------------------------
# Mapping of configuration names to file locations and schemas
# -------------------------------------------------------------------------
CONFIG_MAPPING: dict[str, ConfigEntry] = {
    # Core configurations
    "model_config": ConfigEntry("config", "config.ini"),
    "app_config": ConfigEntry("config", "app_config.yaml"),

    # Report configurations
    "report_internal": ConfigEntry("config", "internal_vars.yaml"),
    "report_inputs": ConfigEntry("config", "inputs.yaml"),
    "report_outputs": ConfigEntry("config", "outputs.yaml"),
    "report_parameters": ConfigEntry("config", "parameters.yaml"),

    # Parameter tables
    "co2_preimpoundment": ConfigEntry(
        "parameters/Carbon_Dioxide", "pre-impoundment.yaml"
    ),
    "ch4_preimpoundment": ConfigEntry(
        "parameters/Methane", "pre-impoundment.yaml"
    ),
    "mcdowell_n_exports": ConfigEntry(
        "parameters/McDowell", "landscape_TN_export.yaml",
        "schemas", "landscape_TN_export_schema.json"
    ),
    "mcdowell_p_exports": ConfigEntry(
        "parameters/McDowell", "landscape_TP_export.yaml",
        "schemas", "landscape_TP_export_schema.json"
    ),
    "gres_p_exports": ConfigEntry(
        "parameters", "phosphorus_exports.yaml",
        "schemas", "phosphorus_exports_schema.json"
    ),
    "gres_p_loads": ConfigEntry(
        "parameters", "phosphorus_loads.yaml",
        "schemas", "phosphorus_loads_schema.json"
    ),
}


def find_config_name_by_filename(filename: str | pathlib.Path) -> Optional[str]:
    """Discover a config name from a given config filename.

    Args:
        filename (str): The filename (with or without path) of the config file.

    Returns:
        Optional[str]: The corresponding config name if found, otherwise None.
    """
    filename = pathlib.Path(filename).name  # Extract just the final filename component
    for config_name, entry in CONFIG_MAPPING.items():
        if entry.config_filename == filename:
            return config_name
    return None


def find_config_schema_by_filename(filename: str) -> Optional[str]:
    """Find the schema file path associated with a given configuration filename.

    This function searches the global ``CONFIG_MAPPING`` for the configuration
    entry that corresponds to the provided configuration filename, and returns
    the name (str) of the associated schema file, if any.

    Args:
        filename (str): The configuration filename. Can include a path or just the
            base filename; only the name component is used for matching.

    Returns:
        Optional[str]: The name of the schema file if one exists,
        otherwise ``None``.
    """
    filename = pathlib.Path(filename).name  # normalize to base name
    entry = next(
        (entry for entry in CONFIG_MAPPING.values() if entry.config_filename == filename),
        None,
    )
    if entry and entry.schema_filename:
        return entry.schema_filename
    return None


def discover_and_reset_configs(
        folder: Union[pathlib.Path, str],
        verbose: bool = False) -> None:
    """Discover and reset all known configurations found in a given folder.

    This function scans the provided folder for configuration files whose
    names match entries in the global ``CONFIG_MAPPING``. For each matching
    file, the corresponding configuration registry entry is updated using
    :func:`reset`. If a matching schema file is present in the same folder,
    it is automatically associated and used for validation.

    Args:
        folder (pathlib.Path | str): Path to the folder containing configuration
            and (optionally) schema files.
        verbose (bool): If True, prints detailed info about discovered and
            reset configurations.

    Returns:
        None
    """
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {folder}")
    # Iterate through files in the folder
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        # Try to identify a known config file
        config_name = find_config_name_by_filename(file_path.name)
        if config_name is None:
            if verbose:
                console.print(f"[yellow]Skipping unknown file:[/yellow] {file_path.name}")
            continue
        # Try to identify a matching schema file
        schema_name = find_config_schema_by_filename(file_path.name)
        schema_path = folder / schema_name if schema_name and (folder / schema_name).exists() else None
        # Update the registry
        reset(config_name=config_name, file_path=file_path, schema_file=schema_path)
        if verbose:
            console = Console()
            msg = Text.assemble(
                ("Resetting config: ", "bold green"),
                (config_name, "bold white"),
                (" from file: ", "green"),
                (str(file_path), "bold white"),
            )
            if schema_path:
                msg.append((" using schema: ", "cyan"))
                msg.append((str(schema_path), "bold cyan"))
            console.print(msg)


def register_configs() -> None:
    """Registers all configuration files used in the reemission package.
    
    Uses the global variable CONFIG_MAPPING which maps config names to
    ConfigEntry objects containing information about:
    - config file name
    - config file path
    - config schema file name
    - config schema file path
    """
    
    global _has_registered
    if _has_registered:
        return
    _has_registered = True
    
    from reemission.utils import get_package_file
    from reemission.registry import config as reemission_config
    
    for name, entry in CONFIG_MAPPING.items():
        file_path = get_package_file(f"{entry.config_subdir}/{entry.config_filename}")
        schema_file = None
        if entry.schema_subdir and entry.schema_filename:
            schema_file = get_package_file(f"{entry.schema_subdir}/{entry.schema_filename}")
        reemission_config.register(name, file_path=file_path, schema_file=schema_file)

    
def reset_all() -> None:
    """Resets all configurations to their default state."""
    from reemission.registry import config as reemission_config
    global _has_registered                   # reset the guard
    _has_registered = False
    # Clear all registered configurations
    reemission_config._configs.clear()
    reemission_config._loaders.clear()
    # Re-register the default configurations
    register_configs()


def reset(config_name: str, file_path: pathlib.Path, schema_file: Optional[pathlib.Path] = None) -> None:
    """Reset a registered configuration to use a different config file.

    This function replaces the current configuration entry in the registry
    with the contents of the provided configuration and schema files. It is
    typically used to restore or re-initialize configurations to their
    default versions after modifications during runtime.

    Args:
        config_name (str): The unique name of the configuration entry to reset.
        file_path (pathlib.Path): Path to the configuration file used to reset the entry.
        schema_file (Optional[pathlib.Path]): Optional path to a schema file
            used for validation of the configuration.

    Returns:
        None
    """
    from reemission.registry import config as reemission_config
    reemission_config.override(name=config_name, file_path=file_path, schema_file=schema_file)


if __name__ == "__main__":
    # If this script is run directly, register the configs
    register_configs()
    print(reemission_config.config_names)
