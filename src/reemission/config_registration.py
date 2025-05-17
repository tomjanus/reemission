""" """
_has_registered = False

def register_configs() -> None:
    """Registers all configuration files used in the reemission package."""
    
    global _has_registered
    if _has_registered:
        return
    _has_registered = True
    
    from reemission.utils import get_package_file
    from reemission import registry
    
    # Register model config
    registry.main_config.register(
        "model_config", 
        file_path=get_package_file("config/config.ini"))
    registry.main_config.register(
        "app_config",
        file_path=get_package_file("config/app_config.yaml"))
    # Register presenter config
    registry.presenter_config.register(
        "report_internal", 
        file_path=get_package_file("config/internal_vars.yaml"))
    registry.presenter_config.register(
        "report_inputs", 
        file_path=get_package_file("config/inputs.yaml"))
    registry.presenter_config.register(
        "report_outputs", 
        file_path=get_package_file("config/outputs.yaml"))
    registry.presenter_config.register(
        "report_parameters", 
        file_path=get_package_file("config/parameters.yaml"))
    # Register tables
    registry.tables.register(
        "co2_preimpoundment", 
        file_path=get_package_file("parameters/Carbon_Dioxide/pre-impoundment.yaml"),
        schema_file=None)
    registry.tables.register(
        "ch4_preimpoundment", 
        file_path=get_package_file("parameters/Methane/pre-impoundment.yaml"),
        schema_file=None)
    registry.tables.register(
        "mcdowell_n_exports", 
        file_path=get_package_file("parameters/McDowell/landscape_TN_export.yaml"),
        schema_file=get_package_file('schemas/landscape_TN_export_schema.json'))
    registry.tables.register(
        "mcdowell_p_exports", 
        file_path=get_package_file("parameters/McDowell/landscape_TP_export.yaml"),
        schema_file=get_package_file('schemas/landscape_TP_export_schema.json'))
    registry.tables.register(
        "gres_p_exports", 
        file_path=get_package_file("parameters/phosphorus_exports.yaml"),
        schema_file=get_package_file('schemas/phosphorus_exports_schema.json'))
    registry.tables.register(
        "gres_p_loads", 
        file_path=get_package_file("parameters/phosphorus_loads.yaml"),
        schema_file=get_package_file('schemas/phosphorus_loads_schema.json'))
