#!/usr/bin/python3
"""
Module that contains the command line application.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m reemission` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``reemission.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``reemission.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import click
import logging
import pathlib
import pyfiglet
import reemission
import reemission.presenter
from reemission.utils import add_version, get_package_file
from reemission.model import EmissionModel
from reemission.input import Inputs


# Update this section if new writers are added to the package
ext_writer_dict = {
    '.json': reemission.presenter.JSONWriter,
    '.tex': reemission.presenter.LatexWriter,
    '.pdf': reemission.presenter.LatexWriter,
    '.xls': reemission.presenter.ExcelWriter,
    '.xlsx': reemission.presenter.ExcelWriter
}

# Set up module logger
log = logging.getLogger(__name__)

FIGLET = True


@click.group()
@add_version
def main(figlet: bool = FIGLET) -> None:
    """------------------------ RE-EMISSION  ------------------------

You are now using the Command line interface of RE-Emission, a Python3
toolbox for calculating greenhouse gas emissions from man-made reservoirs,
created at the University of Manchester, UK
(https://www.manchester.ac.uk).

This is a python package currently installed in your python environement.
See the full documentation at : https://reemisison.readthedocs.io/en/latest/.
"""
    if figlet:
        result = pyfiglet.figlet_format("RE-Emission")
        click.echo(click.style(result, fg='blue'))


@click.command()
@click.argument("input-file", nargs=1, type=click.Path(exists=True))
@click.option("-o", "--output-files", type=click.Path(), multiple=True,
              default=None,
              help="files the outputs are written to.")
@click.option("-c", "--output-config", type=click.Path(exists=True),
              default=None,
              help="RE-Emission output configuration file.")
def calculate(input_file, output_files, output_config):
    """
    Calculates emissions based on the data in the JSON INPUT_FILE.
    Saves the results to output file(s) defined in option '--output-files'.
    Two types of output files are available: '.json' and 'tex/pdf'.
    'pdf' files are written using latex intermediary. Latex source files are
    saved alongside 'pdf' files.

    \f
    input_file: JSON file with information about catchment and reservoir
        related inputs.
    output_files: paths of outputs files.
    output_config: YAML output configuration file.
    """
    click.echo("Loading inputs...\n")
    input_data = Inputs.fromfile(input_file)
    # Use the default config file if not provided as an argument
    if not output_config:
        output_config = get_package_file('config', 'outputs.yaml')
    model = EmissionModel(inputs=input_data, config=output_config.as_posix())
    # Format all file names by converting to unicode
    input_file_str = f"{click.format_filename(input_file)}"
    output_config_str = f"{click.format_filename(output_config)}"
    output_files_unicode = [
        f"{click.format_filename(file)}" for file in output_files]
    output_files_str = ', '.join(output_files_unicode)
    # Create a confirmation message
    msgs = [
        "About to run the program with the following inputs:\n",
        f"Input JSON file: {input_file_str}\n",
        f"Output config file: {output_config_str}\n",
        f"Output files: {output_files_str}\n"]
    click.echo("".join(msgs))
    click.echo('Continue? [yn] ', nl=False)
    c_input = click.getchar()
    click.echo()
    if c_input == 'y':
        click.echo('Ready to calculate.')
    elif c_input == 'n':
        click.echo('Aborting.')
        return
    else:
        click.echo('Invalid input. Please try again.')
        return
    click.echo(click.style(
        "Calculating...", blink=True, bg='blue', fg='white'))
    model.calculate()
    click.echo(click.style(
        "Calculation finished", blink=False, bg='green', fg='white'))

    writers = []
    for file in output_files:
        file_ext = pathlib.Path(file).suffix.lower()
        popped_writer = ext_writer_dict.pop(file_ext, None)
        if popped_writer is None:
            log.warning(
                "Unable to save file %s. Unrecognized extension." % file)
        else:
            writers.append(popped_writer)

    if writers:
        click.echo(click.style(
            "Writing outputs...", blink=True, bg='blue', fg='white'))
        model.add_presenter(
            writers=writers,
            output_files=output_files)
        model.save_results()
        click.echo(click.style(
            "Outputs written to files.", blink=False, bg='green', fg='white'))


main.add_command(calculate)
