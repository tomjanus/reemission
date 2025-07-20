""" """
from typing import Optional
import pathlib

_has_registered = False

def register_configs() -> None:
    """Registers all configuration files used in the reemission package."""
    
    global _has_registered
    if _has_registered:
        return
    _has_registered = True
    
    from reemission.utils import get_package_file
    from reemission.registry import config as reemission_config
    
    # Register model config
    reemission_config.register(
        "model_config", 
        file_path=get_package_file("config/config.ini"))
    reemission_config.register(
        "app_config",
        file_path=get_package_file("config/app_config.yaml"))
    # Register presenter config
    reemission_config.register(
        "report_internal", 
        file_path=get_package_file("config/internal_vars.yaml"))
    reemission_config.register(
        "report_inputs", 
        file_path=get_package_file("config/inputs.yaml"))
    reemission_config.register(
        "report_outputs", 
        file_path=get_package_file("config/outputs.yaml"))
    reemission_config.register(
        "report_parameters", 
        file_path=get_package_file("config/parameters.yaml"))
    # Register tables
    reemission_config.register(
        "co2_preimpoundment", 
        file_path=get_package_file("parameters/Carbon_Dioxide/pre-impoundment.yaml"),
        schema_file=None)
    reemission_config.register(
        "ch4_preimpoundment", 
        file_path=get_package_file("parameters/Methane/pre-impoundment.yaml"),
        schema_file=None)
    reemission_config.register(
        "mcdowell_n_exports", 
        file_path=get_package_file("parameters/McDowell/landscape_TN_export.yaml"),
        schema_file=get_package_file('schemas/landscape_TN_export_schema.json'))
    reemission_config.register(
        "mcdowell_p_exports", 
        file_path=get_package_file("parameters/McDowell/landscape_TP_export.yaml"),
        schema_file=get_package_file('schemas/landscape_TP_export_schema.json'))
    reemission_config.register(
        "gres_p_exports", 
        file_path=get_package_file("parameters/phosphorus_exports.yaml"),
        schema_file=get_package_file('schemas/phosphorus_exports_schema.json'))
    reemission_config.register(
        "gres_p_loads", 
        file_path=get_package_file("parameters/phosphorus_loads.yaml"),
        schema_file=get_package_file('schemas/phosphorus_loads_schema.json'))

    
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
    """Reset config to its default state"""
    config.override(name=config_name, file_path=file_path, schema_file=schema_file)


if __name__ == "__main__":
    # If this script is run directly, register the configs
    register_configs()
    print(reemission_config.config_names)
