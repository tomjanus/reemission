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
from typing import Dict, List, Tuple
import os
import logging
import configparser
import pathlib
import textwrap
import rich_click as click
import pyfiglet
from fpdf import FPDF
import subprocess
import reemission
import reemission.presenter
from reemission.app_logger import create_logger
from reemission.utils import (
    add_version, get_package_file, read_config, get_folder_size, 
    clean_folder, debug_on_exception)
from reemission.model import EmissionModel
from reemission.input import Inputs
from reemission.integration.cli import cli as integration_cli
click.rich_click.USE_MARKDOWN = True

# Update this section if new writers are added to the package
ext_writer_dict = {
    '.json': reemission.presenter.JSONWriter,
    '.tex': reemission.presenter.LatexWriter,
    '.pdf': reemission.presenter.LatexWriter,
    '.xls': reemission.presenter.ExcelWriter,
    '.xlsx': reemission.presenter.ExcelWriter
}

# Set up module logger
log = create_logger(logger_name=__name__)

FIGLET: bool = True
INI_FILE: str = get_package_file('config', 'config.ini')
# Derive default calculation options from config
config: configparser.ConfigParser = read_config(INI_FILE)
p_export_cal = config.get("CALCULATIONS", "p_export_cal")
nitrous_oxide_model = config.get("CALCULATIONS", "nitrous_oxide_model")

def run_command(command, print_result: bool = False, check: bool = False):
    #log.debug("Command: {}".format(command))
    result = subprocess.run(command, shell=False, capture_output=False, check=check)
    if result.stderr:
        raise subprocess.CalledProcessError(
                returncode = result.returncode,
                cmd = result.args,
                stderr = result.stderr
                )
    if result.stdout and print_result:
        log.debug("Command Result: {}".format(result.stdout.decode('utf-8')))
    return result


@click.group()
@add_version
def main(figlet: bool = FIGLET) -> None:
    """------------------------ RE-EMISSION  ------------------------

You are now using the Command line interface of RE-Emission, a Python
toolbox for calculating greenhouse gas emissions from reservoirs..

See the full documentation at : https://reemisison.readthedocs.io/en/latest/.
"""
    if figlet:
        result = pyfiglet.figlet_format("RE-Emission")
        click.echo(click.style(result, fg='blue'))


@click.command()
@click.argument("input-file", nargs=1, type=click.Path(exists=True))
@click.option("-o", "--output-files", type=click.Path(), multiple=True,
              default=None,
              help="Files the outputs are written to.")
@click.option("-c", "--output-config", type=click.Path(exists=True),
              default=None,
              help="RE-Emission output configuration file.")
@click.option("-a", "--author", type=click.STRING, default="",
              help="Author's name")
@click.option("-t", "--title", type=click.STRING, default="Results",
              help="Report/Study title")
@click.option("-p", "--p-model", type=click.STRING, default=p_export_cal,
              help="P-calculation method for CO2 emissions: g-res/mcdowell")
@click.option("-n", "--n2o-model", type=click.STRING,
              default=nitrous_oxide_model,
              help="Model for calculating N2O emissions: model_1/model_2")
@click.option("-c", "--confirm", is_flag=True, show_default=True, default=False)
def calculate(input_file, output_files, output_config, author,
              title, p_model, n2o_model, confirm) -> None:
    """
    Calculates emissions based on the data in the JSON INPUT_FILE.
    Saves the results to output file(s) defined in option '--output-files'.
    Two types of output files are available: '.json' and 'tex/pdf'.
    'pdf' files are written using latex intermediary. Latex source files are
    saved alongside 'pdf' files.

    Args:
    input_file: JSON file with information about catchment and reservoir
        related inputs.
    output_files: Paths of outputs files.
    output_config: YAML output configuration file.
    author: Author's name.
    title: Report/Study title.
    p_model: Method for estimating phosphorus loading to reservoirs
    n2o_model: Model for estimating N2O emissions.
    """
    click.echo("Loading inputs...\n")
    input_data = Inputs.fromfile(input_file)
    # Use the default config file if not provided as an argument
    if not output_config:
        output_config = get_package_file('config', 'outputs.yaml')
    model = EmissionModel(
        inputs=input_data,
        config=output_config.as_posix(),
        author=author,
        report_title=title,
        p_model=p_model)
    # Format all file names by converting to unicode
    input_file_str = f"{click.format_filename(input_file)}"
    output_config_str = f"{click.format_filename(output_config)}"
    output_files_unicode = [
        f"{click.format_filename(file)}" for file in output_files]
    output_files_str = ', '.join(output_files_unicode)
    # Create a confirmation message
    if not confirm:
        msgs = [
            "Running reemission with the following inputs:\n",
            f"Input JSON file: {input_file_str}\n",
            f"Output config file: {output_config_str}\n",
            f"Output files: {output_files_str}\n",
            f"Phosphorus load estimation method: {p_model}\n",
            f"Model for estimating nitrous oxide emissions: {n2o_model}\n"]
    else:
        msgs = [
            "About to run reemission with the following inputs:\n",
            f"Input JSON file: {input_file_str}\n",
            f"Output config file: {output_config_str}\n",
            f"Output files: {output_files_str}\n",
            f"Phosphorus load estimation method: {p_model}\n",
            f"Model for estimating nitrous oxide emissions: {n2o_model}\n"]    
        click.echo("".join(msgs))
        click.echo('Continue? [yn] ', nl=False)
        c_input = click.getchar()
        click.echo()
        if c_input.lower() == 'y':
            click.echo('Ready to calculate.')
        elif c_input.lower() == 'n':
            click.echo('Aborting.')
            return
        else:
            click.echo(f'Input `{c_input}` not recognized. Please try again.')
            return
        click.echo("Calculating...")
    model.calculate()

    writers = []
    for file in output_files:
        file_ext = pathlib.Path(file).suffix.lower()
        popped_writer = ext_writer_dict.pop(file_ext, None)
        if popped_writer is None:
            log.warning("Unable to save file %s. Unrecognized extension.",
                        file)
        else:
            writers.append(popped_writer)

    if writers:
        click.echo("Writing outputs...")
        model.add_presenter(
            writers=writers,
            output_files=output_files)
        model.save_results()
        click.echo("Outputs written to files.")


@click.command()
def log_to_pdf() -> None:
    """Converts log in text format into a PDF"""
    def _text_to_pdf(text: str, filename: pathlib.Path) -> None:
        """Converts string into a pdf file"""
        # Set page dimensions and text width
        a4_width_mm = 210
        pt_to_mm = 0.35
        fontsize_pt = 10
        fontsize_mm = fontsize_pt * pt_to_mm
        margin_bottom_mm = 10
        character_width_mm = 7 * pt_to_mm
        width_text = a4_width_mm / character_width_mm
        # Instantiate the FPDF object
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(True, margin=margin_bottom_mm)
        pdf.add_page()
        pdf.set_font(family='Courier', size=fontsize_pt)
        # Split text into lines, wrap each line to the maximum number of characters
        # and write each line to a pdf.cell
        splitted: List[str] = text.split('\n')
        for line in splitted:
            lines = textwrap.wrap(line, width_text)
            if len(lines) == 0:
                pdf.ln()
            for wrap in lines:
                pdf.cell(0, fontsize_mm, wrap, ln=1)
        # Output the PDF file
        pdf.output(filename, 'F')

    app_config: Dict = reemission.utils.load_yaml(
        file_path=get_package_file("./config/app_config.yaml"))
    log_path = get_package_file(app_config['logging']['log_dir'])
    log_filename = app_config['logging']['log_filename']
    log_filename_no_ext = log_filename.split(".")[0]
    log_file_path = pathlib.Path.joinpath(log_path, log_filename)
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
    except FileNotFoundError:
        log.error("Log file cannot be converted to PDF")
        log.error("Log file %s not found", log_file_path.as_posix())
    else:
        log_filename_pdf = ".".join([log_filename_no_ext,"pdf"])
        pdf_log_file_path = pathlib.Path.joinpath(log_path, log_filename_pdf)
        _text_to_pdf(text_content, pdf_log_file_path)
        log.info(
            "Log file converted to pdf file %s",
            pdf_log_file_path.as_posix())


@click.command()
@click.argument("demo_folder", nargs=1, type=click.Path())
def run_demo(demo_folder: str) -> None:
    """Run a demo analysis for a set of existing and future dams.
    
    Uses a subset of dams from Myanmar case study on assessment of gas emissions
    from existing and future hydroelectric reservoirs in Myanmar
    """
    os.makedirs(demo_folder, exist_ok=True)
    # Import modules in /examples/demo in order to run the demo example
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    #demo_folder = get_package_file("demo")
    from reemission.demo.fetch_inputs import download_mya_case_study_inputs
    from reemission.demo.postprocess_results import process_mya_case_study_results

    # Define folders and remote addresses for downloading input data
    INPUT_FOLDER=os.path.join(demo_folder, "reemission_demo_delineations")
    INPUT_FOLDER_LINK="https://drive.google.com/file/d/1PYqzy4-5P2aW8tvYZPPJ-3fDSDUoHgOv/view?usp=drive_link"
    TARGET_INPUT_FOLDER_SIZE=1179364  # Specify the target size in bytes
    IFC_DB_FOLDER=os.path.join(demo_folder, "reemission_demo_dam_db")
    IFC_DB_LINK="https://drive.google.com/file/d/1OZAVdRMOQN8J-7h3bZIMeQzIfBuo5adC/view?usp=drive_link"
    IFC_DB_SIZE=51854 # Size in bytes
    
    click.echo("RUNNING CALCULATIONS FOR A SUBSET OF EXISTING AND FUTURE HYDROELECTRIC RESERVOIRS IN MYANMAR...")
    
    click.echo("\n1. Fetching the demo dabase of dams from external sources...")
    if os.path.isdir(IFC_DB_FOLDER):
        folder_size = get_folder_size(IFC_DB_FOLDER)
        if folder_size == IFC_DB_SIZE:
            click.echo(f"The DAMS database folder {IFC_DB_FOLDER} exists and has the correct size.")
            click.echo("Fetching the dam database from external sources not required.")
        else:
            click.echo(
                f"The DAMS database in {IFC_DB_FOLDER} exists but its size is not {IFC_DB_SIZE} bytes.")
            # Remove the existing contents of the IFC_DB_FOLDER
            clean_folder(IFC_DB_FOLDER)
            click.echo("Downloading the database form external sources. Please Wait...")
            download_mya_case_study_inputs(
                IFC_DB_LINK, os.path.join(IFC_DB_FOLDER,"reemission_demo_dam_db.zip"))
    else:
        click.echo(f"The DAMS database folder {IFC_DB_FOLDER} does not exist.")
        os.makedirs(IFC_DB_FOLDER, exist_ok=True)
        click.echo("Downloading the database form external sources. Please Wait...")
        download_mya_case_study_inputs(
                IFC_DB_LINK, os.path.join(IFC_DB_FOLDER,"reemission_demo_dam_db.zip"))

    click.echo("\n2. Fetching reservoir and catchment delineations from external sources...")
    if os.path.isdir(INPUT_FOLDER):
        folder_size = get_folder_size(INPUT_FOLDER)
        if folder_size == TARGET_INPUT_FOLDER_SIZE:
            click.echo(f"The DAMS database folder {INPUT_FOLDER} exists and has the correct size.")
            click.echo("Fetching the dam database from external sources not required.")
        else:
            click.echo(
                f"The DAMS database in {INPUT_FOLDER} exists but its size is not {TARGET_INPUT_FOLDER_SIZE} bytes.")
            # Remove the existing contents of the INPUT_FOLDER
            clean_folder(INPUT_FOLDER)
            click.echo("Downloading the database form external sources. Please Wait...")
            download_mya_case_study_inputs(
                INPUT_FOLDER_LINK, os.path.join(INPUT_FOLDER,"reemission_demo_delineations.zip"))
    else:
        click.echo(f"The DAMS database folder {INPUT_FOLDER} does not exist.")
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        click.echo("Downloading the database form external sources. Please Wait...")
        download_mya_case_study_inputs(
                INPUT_FOLDER_LINK, os.path.join(INPUT_FOLDER,"reemission_demo_delineations.zip"))

    OUTPUTS_FOLDER=os.path.join(demo_folder, "geocaret_outputs")
    click.echo(f"\n3. Creating the outputs folder {OUTPUTS_FOLDER} ...")
    # Create outputs folder if it does not exist already
    if not os.path.isdir(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
        click.echo(f"Folder created: {OUTPUTS_FOLDER}")
    else:
        click.echo(f"Folder already exists: {OUTPUTS_FOLDER}")

    click.echo(
        "\n4. Merging tabular data into a single CSV file and saving to "+
        f"{os.path.join(OUTPUTS_FOLDER,'geocaret_outputs.csv')} ...")
    # Find subfolders in the shp folder
    shp_folders = [f.path for f in os.scandir(INPUT_FOLDER) if f.is_dir()]
    combined_csv_file = os.path.join(OUTPUTS_FOLDER, "geocaret_outputs.csv")
    # Run CLI command as a subprocess
    command_1 = ["reemission-geocaret", "process-tab-outputs"]
    for input_folder in shp_folders:
        command_1.append("-i")
        input_file = os.path.join(input_folder, "output_parameters.csv")
        command_1.append(f"{input_file}")
    # Append missing columns to data that are required but not given in GeoCARET
    missing_col_value_pairs: List[Tuple[str, str]] = [
        ("c_treatment_factor", "primary (mechanical)"), 
        ("c_landuse_intensity", "low intensity"),
        ('type', 'unknown')]
    for col_name, col_val in missing_col_value_pairs:
        command_1.append('-cv')
        command_1.append(col_name)
        command_1.append(col_val)
    command_1.append("-o")
    command_1.append(combined_csv_file)
    #print(command_1)
    run_command(command_1)
    #res = subprocess.run(command_1, shell=True, capture_output=False)
    #print(res)

    click.echo("\n5. Merging shape files for individual reservoirs into combined shape files...")
    # Convert the csv file into a JSON input file to RE-Emission
    # Write reemission tab2json CLI function
    # Join shape files for individual reservoirs into combined shapes for each category of shapes
    command_2 = ["reemission-geocaret", "join-shapes"]
    for input_folder in shp_folders:
        command_2.append("-i")
        command_2.append(input_folder)
    command_2 += [
        "-o", OUTPUTS_FOLDER, "-gp", "R_*.shp, C_*.shp, MS_*.shp, PS_*.shp", 
        "-f", "reservoirs.shp, catchments.shp, rivers.shp, dams.shp"]
    run_command(command_2)
    #subprocess.run(command_2, capture_output=True)

    click.echo("\n6. Converting GeoCARET tabular data to the RE-Emission input JSON file")
    command_3 = ["reemission-geocaret", "tab-to-json", "-i", combined_csv_file,
                 "-o", os.path.join(OUTPUTS_FOLDER,"reemission_inputs.json")]
    #subprocess.run(command_3, capture_output=True)
    run_command(command_3)

    REEMISSION_OUTPUTS_FOLDER=os.path.join(demo_folder, "reemission_outputs")
    click.echo(f"\n7. Creating the outputs folder {REEMISSION_OUTPUTS_FOLDER} ...")
    if not os.path.isdir(REEMISSION_OUTPUTS_FOLDER):
        os.makedirs(REEMISSION_OUTPUTS_FOLDER, exist_ok=True)
        click.echo(f"Folder created: {REEMISSION_OUTPUTS_FOLDER}")
    else:
        click.echo(f"Folder already exists: {REEMISSION_OUTPUTS_FOLDER}")

    click.echo("\n8. Calculating GHG emissions with RE-EMISSION")
    # Estimate gas emissions and save output files
    command_4 = [
        "reemission", "calculate", 
        os.path.join(OUTPUTS_FOLDER, "reemission_inputs.json"),
        "-a", "Default User",
        "-t", "Demo Example Results",
        "-o", os.path.join(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs.pdf"),
        "-o", os.path.join(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs.json"),
        "-o", os.path.join(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs_.xlsx")
    ]
    run_command(command_4)
    #subprocess.run(command_4, capture_output=True)

    # Merge results into shape files and visualise on a map
    click.echo("\n9. Merging input and output data into shape files")
    process_mya_case_study_results(
        shp_folder = pathlib.Path(demo_folder)/"geocaret_outputs",
        output_json_file =pathlib.Path(demo_folder)/"reemission_outputs/demo_GHG_outputs.json",
        map_path = pathlib.Path(demo_folder)/"demo_interactive_map",
        ifc_dam_path = pathlib.Path(demo_folder)/"reemission_demo_dam_db"/"dam_db.shp")

    click.echo("\nDONE")


main.add_command(calculate)
main.add_command(log_to_pdf)
main.add_command(run_demo)
main.add_command(integration_cli.geocaret_integrate)
