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
import pyfiglet
import reemission
import reemission.presenter
from reemission.utils import add_version, load_packaged_data
from reemission.model import EmissionModel
from reemission.input import Inputs

# Update this section if new writers are added to the package
writers_dict = {
    'latex': reemission.presenter.LatexWriter,
    'json': reemission.presenter.JSONWriter
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
@click.option("-w", "--writers", type=str, multiple=True, default=None,
              help="output file categories definied by RE-Emission.")
@click.option("-o", "--output-files", type=click.Path(), multiple=True,
              default=None,
              help="files the outputs are written to.")
@click.option("-c", "--output-config", type=click.Path(exists=True),
              default=None,
              help="RE-Emission output configuration file.")
def calculate(input_file, writers, output_files, output_config):
    """
    Calculates emissions based on the data in the JSON INPUT_FILE.
    Saves the results to output file(s) defined in option '--output-files'.
    The types of output files are defined in option '--writers'.
    Currently, two writers are available: 'json' and 'latex'.
    'latex' writer saves output to a '.pdf' file whilst additionally writing
    an intermediate '.tex' source file.

    \f
    input_file: JSON file with information about catchment and reservoir
        related inputs.
    writers: categories out output files written by RE-Emission. Currently
        supports `latex` and `json`.
    output_files: paths of outputs files: one per writer in the same order.
    output_config: YAML output configuration file.
    """
    click.echo("Loading inputs...\n")
    input_data = Inputs.fromfile(input_file)
    # Use the default config file if not provided as an argument
    if not output_config:
        output_config = load_packaged_data('config', 'outputs.yaml')
    model = EmissionModel(inputs=input_data, config=output_config.as_posix())
    # Format all file names by converting to unicode
    input_file_str = f"{click.format_filename(input_file)}"
    output_config_str = f"{click.format_filename(output_config)}"
    writers_str = ', '.join(writers)
    output_files_unicode = [
        f"{click.format_filename(file)}" for file in output_files]
    output_files_str = ', '.join(output_files_unicode)
    # Create a confirmation message
    msgs = [
        "About to run the program with the following inputs:\n",
        f"Input JSON file: {input_file_str}\n",
        f"Output config file: {output_config_str}\n",
        f"Writers: {writers_str}\n",
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

    if writers is not None:
        _writers = []
        for writer in writers:
            popped_writer = writers_dict.pop(writer.lower(), None)
            if popped_writer:
                _writers.append(popped_writer)

    if len(_writers) == len(output_files):
        click.echo(click.style(
            "Writing outputs...", blink=True, bg='blue', fg='white'))
        model.add_presenter(
            writers=_writers,
            output_files=output_files)
        model.save_results()
        click.echo(click.style(
            "Outputs written to files.", blink=False, bg='green', fg='white'))
    else:
        log.warning('Lengths of writers and output files are not equal. ' +
                    'Results could not be saved.')


main.add_command(calculate)
