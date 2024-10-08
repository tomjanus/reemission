"""
Module that contains the command line application.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m mya_emissisons` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``mya_emissions.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``mya_emissions.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
from typing import Callable, List, Optional, Tuple, Any
import sys
import pathlib
import rich_click as click
from reemission.app_logger import create_logger
from reemission.integration.geocaret.geocaret_tab_parser import GeoCaretOutputReader
from reemission.integration.geocaret.geocaret_shp_parser import ShpConcatenator
from reemission.integration.geocaret.geocaret_tab_to_json import (
    TabToJSONConverter, LegacySavingStrategy)
from reemission.utils import load_toml, get_package_file

# Create a logger
logger = create_logger(logger_name=__name__)
FIGLET = True
click.rich_click.USE_MARKDOWN = True

def add_description(fun: Callable) -> Callable:
    """Adds desciption of the GeoCARET integration CLI.

    Params:
        fun: Function to decorate
    Returns:
        Decorated function
    """
    doc = fun.__doc__
    fun.__doc__ = "Integration of RE-Emission with GeoCARET outputs \n\n" + doc
    return fun


@click.group()
@add_description
def geocaret_integrate() -> None:
    """--------------- RE-EMISSION GeoCARET OUTPUT PROCESSING  --------------------

You are now using the Command line interface for an interface package designed
for processing outputs obtained from GeoCARET and creating input files for
RE-EMISSION. \n
GeoCARET is a general catchment and reservoir evaluation tool that uses GIS data.\n
RE-EMISSION is a collection of methods for estimating GHG emissions from reservoirs

Full documentation at : https://tomjanus.github.io/reemission/index.html
"""

@click.option("-i", "--input-files", type=click.Path(), multiple=True)
@click.option("-c", "--config-file", type=click.Path(), default=None)
@click.option("-o", "--output-file", type=click.Path(), 
              default="input_data/all_geocaret_outputs.csv")
@click.option("-id", "--id-field", type=click.STRING, default="id",
              help="Field on which duplicates are looked for")
@click.option('-rd', '--remove-dups', is_flag=True, default=True,
              help="Remove duplicate dam IDs.")
@click.option('-cv', '--col-value-pair', nargs=2, type=click.Tuple([str, str]), multiple=True,
              help="Column / default value pairs for supplying missing tabular information.")
@click.command()
def process_tab_outputs(
        input_files: List[str], config_file: Optional[str], 
        output_file: str, id_field: str, remove_dups: bool,
        col_value_pair: Optional[List[Tuple[str, str]]]=None) -> None:
    """Read tabular output data from GeoCARET. Remove rows with
    duplicate dam ids, selects dam ids from the list or required dam ids.
    
    TODO: Handle existing reservoirs for which volume, max_depth and mean_depth
    are unknown."""
    if not input_files:
        raise ValueError("No input files. Provide input files preceded with -i tags.")
    output_reader = GeoCaretOutputReader(file_paths=input_files)
    geocaret_output = output_reader.read_files()
    if remove_dups:
        geocaret_output.remove_duplicates(on_column=id_field)
    if config_file is None:
        tab_data_config = load_toml(
            get_package_file("./config/geocaret.toml"))['tab_data']
    else:
        tab_data_config = load_toml(pathlib.Path(config_file))['tab_data']
    # Get the list of mandatory columns from config file
    input_files_str: str = ", ".join(input_files)
    if col_value_pair is not None:
        for col_name, col_value in col_value_pair:
            click.echo(f"Adding column: {col_name} with value {col_value} to data in file(s) {input_files_str}")
            geocaret_output.add_column(column_name=col_name, default_value=col_value)
    geocaret_output.filter_columns(
        mandatory_columns=tab_data_config['mandatory_fields'],
        optional_columns=tab_data_config['alternative_fields'])
    logger.debug(f"Saving to {output_file}")
    geocaret_output.to_csv(pathlib.Path(output_file))
    logger.info("Tabular data saved to: %s", output_file)


@click.command()
@click.option("-i", "--input-folders", type=click.Path(), multiple=True)
@click.option("-o", "--output-folder", type=click.Path())
@click.option(
    "-gp", "--glob-patterns", 
    callback=lambda _, __, x: x.split(',') if x else [], 
    default="R_*.shp, MS_*.shp, PS_*.shp")
@click.option(
    "-f", "--output_files", 
    callback=lambda _, __, x: x.split(',') if x else [], 
    default="reservoirs.shp, rivers.shp, dams.shp")
@click.option('--indices', callback=lambda _, __, x: x.split(',') if x else [])
def join_shapes(
        input_folders: List[str], output_folder: str, glob_patterns: List[str], 
        output_files: List[str], indices: List[str]) -> None:
    """Join multiple shapes matching glob pattern into single shape files for each
    glob pattern, e.g. reservoirs.shp for all R_*.shp files, etc."""
    glob_patterns = [pattern.strip() for pattern in glob_patterns]
    output_files = [file.strip() for file in output_files]
    int_indices: Optional[list] = list(map(int, indices))
    if len(glob_patterns) != len(output_files):
        msg = "Number of glob patterns does not match the number of output files."
        sys.exit(msg)
    for input_folder in input_folders:
        if not pathlib.Path(input_folder).is_dir():
            sys.exit(f"Input folder {input_folder} is not a directory.")
    if pathlib.Path(output_folder).is_file():
        sys.exit(f"Output folder {output_folder} is a file.")
    for glob_pattern, output_file in zip(glob_patterns, output_files):
        shp_concat = ShpConcatenator()
        click.echo(
            f"Concatenating shape files matching pattern {glob_pattern}.")
        stats = shp_concat.find_in_folders(
            folders=input_folders, glob_pattern=glob_pattern, 
            sel_ids=int_indices)
        found_ids: List[int] = list(stats['found ids'])
        if not found_ids:
            sys.exit("No shape files found for concatenation.")
        found_ids_str = ", ".join([str(id) for id in found_ids])
        click.echo(f"Concatenating shapes for dam ids {found_ids_str}.")
        # Report on any unfound ids
        unfound_ids: List[int] = list(stats['missing ids'])
        if unfound_ids:
            unfound_ids_str = ", ".join([str(id) for id in unfound_ids])
            click.echo(click.style(
                f'The following ids could not be found {unfound_ids_str}',
                fg='red'))
        shp_geodata = shp_concat.concatenate()
        shp_geodata.save(
            folder=pathlib.Path(output_folder), file_name=output_file)
        file_path = pathlib.Path(output_folder) / output_file
        click.echo(f"Data saved to shape file {file_path}.")


@click.command()
@click.option("-i", "--input-file", type=click.Path())
@click.option("-o", "--output_file", type=click.Path())
def tab_to_json(input_file: str, output_file: str) -> None:
    """Convert tabular data in CSV format into input JSON file in the
    RE-Emissions input format.
    
    Args:
        
    
        input_file: CSV file with input data in tabular format
        
    
        output_file: JSON file with input data in JSON format that can be
            read and processed by RE-Emission
    """
    #TODO: Add tests for input and output file extensions and create tree folder
    # structure for the output file if the folder does not exist
    saving_strategy = LegacySavingStrategy()
    converter = TabToJSONConverter(pathlib.Path(input_file), saving_strategy)
    converter.to_json(pathlib.Path(output_file))


geocaret_integrate.add_command(process_tab_outputs)
geocaret_integrate.add_command(join_shapes)
geocaret_integrate.add_command(tab_to_json)
