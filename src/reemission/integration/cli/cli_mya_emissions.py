""" """
#import mya_emissions
#from mya_emissions.scripts.fetch_external import fetch_inputs
#from mya_emissions.scripts.log_setup import create_logger
#from mya_emissions.scripts.file_loaders import (
#    get_package_file, load_toml, load_csv)
#from mya_emissions.scripts.file_savers import save_to_json
#from mya_emissions.scripts.data_processing import (
#    concatenate_shp_files, process_tabular_data, create_reemission_input,
#    match_tabular_to_shp, copy_shape_data)
#from mya_emissions.scripts.exceptions import ConfigNotFoundException

# Read toml configuration file
#config = load_toml(get_package_file('config/config.toml'))

@click.command()
@click.option("-i", "--input-file", type=click.Path())
@click.option("-o", "--output-file", type=click.Path())
def create_input(input_file: str, output_file: str) -> None:
    """ """


@click.command()
@click.option("-i", "--input-file", type=click.Path())
@click.option("-o", "--output-file", type=click.Path())
def tab_to_json(input_file: str, output_file: str) -> None:
    """ """
    data: dict = load_csv(pathlib.Path(input_file)).to_dict(orient='index')
    # Generate reemission input dictionary for all reservoirs
    reemission_inputs: dict = {}
    for dam_id in data:
        reemission_inputs.update(create_reemission_input(
            input_dict=data[dam_id]))
    # Save results to JSON
    save_to_json(
        output_path=output_file, input_dict=reemission_inputs)


@click.command()
@click.option("-tf", "--tab-file", type=click.Path())
@click.option("-sf", "--shp-file", type=click.Path())
@click.option("-tid", "--tab-id", type=click.STRING)
@click.option("-sid", "--shp-id", type=click.STRING)
def match_tab2shp(
        tab_file: str, shp_file: str, tab_id: str, shp_id: str) -> None:
    """ """
    missing_keys = match_tabular_to_shp(
        tab_data_file=pathlib.Path(tab_file),
        shp_file=pathlib.Path(shp_file),
        keys=(tab_id, shp_id))
    if len(missing_keys) > 0:
        ids = ", ".join(list(map(str, missing_keys)))
        logger.info(
            "The following keys in tab data are missing in the shp file: %s",
            ids)
    else:
        logger.info("All keys in tab data are present in the shp file.")


@click.command()
@click.option("-sf", "--source-file", type=click.Path())
@click.option("-tf", "--target-file", type=click.Path())
@click.option("-sk", "--source-key", type=click.STRING)
@click.option("-tk", "--target-key", type=click.STRING)
@click.option('--fields', callback=lambda _, __, x: x.split(',') if x else [])
@click.option('-s', '--save', is_flag=True, default=True,
              help="Save and overwrite target file.")
def copy_shp_data(source_file: str, target_file: str, source_key: str,
                  target_key: str, fields: List[str], save: bool) -> None:
    """ """
    fields = [field.strip() for field in fields]
    source_shape = pathlib.Path(source_file)
    target_shape = pathlib.Path(target_file)
    keys = (source_key, target_key)
    logger.info(
        "Copying keys %s from source file %s to target file %s",
        ", ".join(fields), source_file, target_file)
    shp_df = copy_shape_data(
        source_shape, target_shape, keys, fields, save)
    logger.info("Done.")


main.add_command(tab_to_json)
main.add_command(match_tab2shp)
main.add_command(copy_shp_data)

