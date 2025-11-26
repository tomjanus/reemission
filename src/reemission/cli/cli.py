#    This file is part of Re-Emission.
#
#    Re-Emission is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Re-Emission is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Re-Emission.  If not, see <http://www.gnu.org/licenses/>.

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
from typing import List, Tuple, Union, Optional
import warnings
import shutil
import os
import pathlib
import textwrap
import rich_click as click
import pyfiglet
from fpdf import FPDF
import subprocess
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.progress import Progress
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
from rich import box

import reemission
import reemission.presenter
from reemission.app_logger import create_logger
from reemission.utils import (
    add_version, get_package_file, get_folder_size, 
    clean_folder, debug_on_exception, deep_get)
from reemission.model import EmissionModel
from reemission.input import Inputs
from reemission.integration.cli import cli as integration_cli
from reemission import registry
from reemission.registry import config as reemission_config
from reemission.config_registration import discover_and_reset_configs

click.rich_click.USE_MARKDOWN = True


# Update this section if new writers are added to the package
ext_writer_dict = {
    '.json': reemission.presenter.JSONWriter,
    '.html': reemission.presenter.HTMLWriter,
    '.tex': reemission.presenter.LatexWriter,
    '.pdf': reemission.presenter.LatexWriter,
    '.xls': reemission.presenter.ExcelWriter,
    '.xlsx': reemission.presenter.ExcelWriter
}

# Set up module logger
log = create_logger(logger_name=__name__)
console = Console()

FIGLET: bool = True
FIGLET_FONT = "slant"
model_config = registry.config.get("model_config")
# Read default parameters from config
# p_export_cal = deep_get(model_config, "CALCULATIONS", "p_export_cal")
# nitrous_oxide_model = deep_get(model_config, "CALCULATIONS", "nitrous_oxide_model")


def run_command(
        command: Union[str, List[str]],
        *,
        print_result: bool = False,
        check: bool = True,
        capture_output: bool = True,
        text: bool = True,
        cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """
    Run a shell command with improved safety, error handling, and logging.

    This is a modern, Pythonic wrapper around :func:`subprocess.run` designed
    for CLI tools. It captures output, logs command execution, and raises
    informative errors when a command fails.

    Args:
        command:
            The command to execute. Can be provided as a string (if ``shell=True``)
            or a list of arguments (recommended for safety).
        print_result:
            Whether to print (or log) the command's stdout to the logger.
            Defaults to ``False``.
        check:
            If ``True``, raises :class:`subprocess.CalledProcessError` on
            non-zero exit status. Defaults to ``True``.
        capture_output:
            Whether to capture the command's stdout/stderr. Defaults to ``True``.
        text:
            Whether to decode bytes output into strings. Defaults to ``True``.
        cwd:
            Optional working directory in which to execute the command.

    Returns:
        A :class:`subprocess.CompletedProcess` instance with attributes:
        ``args``, ``returncode``, ``stdout``, and ``stderr``.

    Raises:
        subprocess.CalledProcessError:
            If ``check=True`` and the command exits with a non-zero status.

    Example:
        >>> run_command(["echo", "Hello, world!"], print_result=True)
        INFO: Command succeeded: echo Hello, world!
        Hello, world!
    """

    # Ensure safety for string commands
    if isinstance(command, str):
        shell = True
        log.debug(f"Running shell command: {command}")
    else:
        shell = False
        log.debug(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=capture_output,
            text=text,
            cwd=cwd,
        )

        if print_result and result.stdout:
            log.info(result.stdout.strip())

        if result.stderr:
            log.debug(f"Command stderr: {result.stderr.strip()}")

        return result

    except subprocess.CalledProcessError as e:
        log.error(f"Command failed ({e.returncode}): {e.cmd}")
        if e.stderr:
            log.error(e.stderr.strip())
        raise
    except FileNotFoundError:
        log.error(f"Command not found: {command}")
        raise


def run_rich_command(command: list[str]) -> None:
    """Run subprocess-style commands and display formatted output."""
    console.print(f"[cyan]$ {' '.join(command)}[/cyan]")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]✓ Command completed successfully[/green]")
    else:
        console.print(f"[red]✗ Command failed[/red]\n{result.stderr}")
        raise click.ClickException("Command execution failed.")


def get_folder_size_rich(folder: str) -> int:
    """Compute folder size in bytes."""
    total = 0
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def clean_folder_rich(folder: str) -> None:
    """Delete all contents of a folder, reporting issues gracefully."""
    if not os.path.exists(folder):
        console.print(f"[yellow]Warning:[/] Folder '{folder}' does not exist.")
        return

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except PermissionError:
            console.print(f"[red]Permission denied:[/] {file_path}")
        except FileNotFoundError:
            console.print(f"[yellow]File already removed:[/] {file_path}")
        except OSError as e:
            console.print(f"[red]OS error while deleting {file_path}: {e.strerror}[/red]")

    console.print(f"[green]✓ Cleaned contents of folder:[/] {folder}")


@click.group()
@add_version
def main(figlet: bool = FIGLET) -> None:
    """------------------------ RE-EMISSION  ------------------------

You are now using the Command line interface of RE-Emission, a Python
toolbox for calculating greenhouse gas emissions from reservoirs..

See the full documentation at : https://tomjanus.github.io/reemission/index.html
"""
    if figlet:
        result = pyfiglet.figlet_format("RE-Emission")
        click.echo(click.style(result, fg='blue'))

@click.command()
@click.argument("input-file", nargs=1, type=click.Path(exists=True))
@click.option("-o", "--output-files", type=click.Path(), multiple=True, default=None,
              help="Files the outputs are written to.")
@click.option("-cf", "--config-folder", type=click.Path(exists=True), default=None,
              help="Path to custom RE-Emission config files.")
@click.option("-a", "--author", type=click.STRING, default="",
              help="Author's name.")
@click.option("-t", "--title", type=click.STRING, default="Results",
              help="Report or study title.")
@click.option("-p", "--p-model", type=click.STRING, default="g-res",
              help="P-calculation method for CO₂ emissions: g-res/mcdowell.")
@click.option("-n", "--n2o-model", type=click.STRING, default="maavara_1",
              help="Model for calculating N₂O emissions: maavara_1/maavara_2.")
@click.option("-rc", "--retention-coefficient", type=click.STRING, default="larsen",
              help="Phosphorus retention coefficient model: larsen/maavara.")              
@click.option("-c", "--confirm", is_flag=True, show_default=True, default=False,
              help="Ask for confirmation before running.")
@click.option("-v", "--verbose", is_flag=True, show_default=True, default=False,
              help="Provide additional output during execution.")
def calculate(input_file, output_files, config_folder, author,
              title, p_model, n2o_model, retention_coefficient, confirm,
              verbose) -> None:
    """
    Calculates emissions based on the data in the JSON INPUT_FILE.

    Results are saved to one or more output files defined by '--output-files'.
    Supported formats include '.json', '.tex', and '.pdf'. PDF reports are
    generated using a LaTeX intermediary, with sources stored alongside them.
    """

    console.rule("[bold cyan]RE-Emission Calculation[/bold cyan]")

    # Step 1. Load inputs
    console.print("[yellow]Loading inputs...[/yellow]")
    try:
        input_data = Inputs.fromfile(input_file)
    except Exception as e:
        console.print(f"[red]Failed to load input file:[/red] {e}")
        raise click.Abort()

    # Step 2. Resolve output configuration
    if config_folder:
        if verbose:
            console.print(f"[blue]Discovering config files in folder:[/blue] [bold white]{config_folder}[/bold white] ...")
        discover_and_reset_configs(config_folder, verbose)
        
    output_config = registry.config.get("report_outputs")
        
    # Step 3. Update the P-retention coefficient dynamically via config
    if retention_coefficient not in {"larsen", "maavara"}:
        console.print(f"[red] Phosphorus retention model '{retention_coefficient}' not known. Using Larsen and Mercier.[/red]")
        retention_coefficient = "larsen"
    reemission_config.update("model_config", {("CALCULATIONS",): {"ret_coeff_method": retention_coefficient}})
    retention_coefficient_updated = deep_get(model_config, "CALCULATIONS", "ret_coeff_method")

    # Step 4. Summarize configuration in a table
    table = Table(title="\nCalculation Parameters", header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Input JSON file", click.format_filename(input_file))
    table.add_row("Config folder with custom configs",
                  click.format_filename(config_folder) if isinstance(config_folder, str)
                  else "All configs loaded from config registry")
    table.add_row("Output files",
                  ", ".join([click.format_filename(f) for f in output_files]) if output_files else "None")
    table.add_row("Author", author or "N/A")
    table.add_row("Title", title)
    table.add_row("P-load model (g-res / mc-dowell)", p_model)
    table.add_row("P-retention model (larsen / maavara)", retention_coefficient_updated)
    table.add_row("N₂O model (maavara_1 / maavara_2)", n2o_model)
    console.print(table)

    # Step 5. Confirm before running
    if confirm:
        if not Confirm.ask("[bold yellow]Continue with these settings?[/bold yellow]"):
            #console.print("[red]Aborted by user.[/red]")
            raise click.Abort()

    # Step 6. Build model
    console.print("[yellow]Initializing model...[/yellow]")
    model = EmissionModel(
        inputs=input_data,
        presenter_config=output_config,
        author=author,
        report_title=title,
        p_model=p_model,
    )

    # Step 7. Perform calculation with progress feedback
    console.print("[yellow]Starting calculations...[/yellow]")
    with Progress(transient=True) as progress:
        task = progress.add_task("Computing emissions...", total=1)
        try:
            model.calculate()
        except Exception as e:
            console.print(f"[red]Error during calculation:[/red] {e}")
            raise click.Abort()
        progress.advance(task)

    # Step 8. Prepare file writers
    writers = []
    for file in output_files or []:
        file_ext = pathlib.Path(file).suffix.lower()
        popped_writer = ext_writer_dict.pop(file_ext, None)
        if popped_writer is None:
            log.warning("Unable to save file %s. Unrecognized extension %s.",
                        file, file_ext)
            console.print(f"[red]Skipping unknown extension:[/red] {file_ext}")
        else:
            writers.append(popped_writer)

    # Step 9. Save results
    if writers:
        console.print("[yellow]Writing outputs...[/yellow]")
        try:
            model.add_presenter(writers=writers, output_files=output_files)
            model.save_results()
        except Exception as e:
            console.print(f"[red]Error writing outputs:[/red] {e}")
            raise click.Abort()
        console.print("[green]Outputs successfully written.[/green]")
    else:
        console.print("[red]No valid output writers available.[/red]")

    console.rule("[bold green]Calculation Complete[/bold green]")


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

    # -------------------------------------------------------------------------
    # Resolve paths from config
    # -------------------------------------------------------------------------
    try:
        app_config = registry.config.get("app_config")
        log_dir = get_package_file(".") / pathlib.Path(app_config['logging']['log_dir'])
        log_filename = app_config['logging']['log_filename']
    except KeyError as e:
        click.secho(f"❌ Missing logging configuration key: {e}", fg="red")
        return
        
    log_file_path = log_dir / log_filename
    log_filename_no_ext = pathlib.Path(log_filename).stem
    pdf_log_file_path = log_dir / f"{log_filename_no_ext}.pdf"

    # -------------------------------------------------------------------------
    # Read and convert log
    # -------------------------------------------------------------------------
    if not log_file_path.exists():
        click.secho(f"❌ Log file not found: {log_file_path.resolve()}", fg="red")
        return

    click.echo(f"📖 Reading log file: {log_file_path.resolve()}")
    with log_file_path.open('r', encoding='utf-8') as file:
        text_content = file.read()

    click.echo(f"📝 Converting to PDF: {pdf_log_file_path.resolve()}")
    _text_to_pdf(text_content, pdf_log_file_path)

    click.secho(f"✅ Log successfully converted to: {pdf_log_file_path.resolve()}", fg="green")


@click.command()
@click.argument("demo_folder", nargs=1, type=click.Path())
def run_demo(demo_folder: str) -> None:
    """Run a demo analysis for a set of existing and future dams in Myanmar."""

    os.makedirs(demo_folder, exist_ok=True)
    
    # Ignore normalization / truncation warnings from pyogrio
    warnings.filterwarnings(
        "ignore",
        message="Normalized/laundered field name",
        module="pyogrio.raw"
    )

    # Ignore truncation warnings for shapefile field names
    warnings.filterwarnings(
        "ignore",
        message="Column names longer than 10 characters will be truncated",
        module="geopandas"
    )

    # Ignore numeric precision issues in shapefile writing
    warnings.filterwarnings(
        "ignore",
        message="Value .* not successfully written",
        module="pyogrio.raw"
    )

    # Full-width header with spacing below
    header = Panel(
        "[bold cyan]RE-Emission Demo[/bold cyan]\n"
        "[white]Myanmar Hydroelectric Reservoir Case Study[/white]",
        subtitle="[italic dim]Automated GHG Emission Assessment Workflow[/italic dim]",
        subtitle_align="right",
        box=box.ROUNDED,
        expand=True,  # <-- makes it full width
        padding=(1, 2),  # <-- top/bottom and left/right padding inside panel
    )

    console.print(header)
    console.print()  # adds a blank line (extra spacing below)

    # Imports
    from reemission.demo.fetch_inputs import download_mya_case_study_inputs
    from reemission.demo.postprocess_results import process_mya_case_study_results

    # Paths and constants
    INPUT_FOLDER = os.path.join(demo_folder, "reemission_demo_delineations")
    INPUT_FOLDER_LINK = "https://drive.google.com/file/d/1PYqzy4-5P2aW8tvYZPPJ-3fDSDUoHgOv/view?usp=drive_link"
    TARGET_INPUT_FOLDER_SIZE = 1179364
    IFC_DB_FOLDER = os.path.join(demo_folder, "reemission_demo_dam_db")
    IFC_DB_LINK = "https://drive.google.com/file/d/1OZAVdRMOQN8J-7h3bZIMeQzIfBuo5adC/view?usp=drive_link"
    IFC_DB_SIZE = 51854

    OUTPUTS_FOLDER = os.path.join(demo_folder, "geocaret_outputs")
    REEMISSION_OUTPUTS_FOLDER = os.path.join(demo_folder, "reemission_outputs")

    console.print(Rule("[bold yellow]1. Fetching the demo database of dams[/bold yellow]"))
    if os.path.isdir(IFC_DB_FOLDER):
        folder_size = get_folder_size_rich(IFC_DB_FOLDER)
        if folder_size == IFC_DB_SIZE:
            console.print(f"[green]✓ DAM database folder exists and has the correct size.[/green]")
        else:
            console.print(f"[yellow]⚠ Existing DAM database size mismatch ({folder_size} vs {IFC_DB_SIZE})[/yellow]")
            clean_folder_rich(IFC_DB_FOLDER)
            console.print("[cyan]Downloading database... please wait[/cyan]")
            download_mya_case_study_inputs(
                IFC_DB_LINK, os.path.join(IFC_DB_FOLDER, "reemission_demo_dam_db.zip")
            )
    else:
        console.print(f"[red]DAM database folder not found.[/red] Creating and downloading...")
        os.makedirs(IFC_DB_FOLDER, exist_ok=True)
        download_mya_case_study_inputs(
            IFC_DB_LINK, os.path.join(IFC_DB_FOLDER, "reemission_demo_dam_db.zip")
        )

    console.print(Rule("[bold yellow]2. Fetching reservoir and catchment delineations[/bold yellow]"))
    if os.path.isdir(INPUT_FOLDER):
        folder_size = get_folder_size_rich(INPUT_FOLDER)
        if folder_size == TARGET_INPUT_FOLDER_SIZE:
            console.print(f"[green]✓ Delineations folder exists and has the correct size.[/green]")
        else:
            console.print(f"[yellow]⚠ Size mismatch in delineations folder ({folder_size} vs {TARGET_INPUT_FOLDER_SIZE})[/yellow]")
            clean_folder_rich(INPUT_FOLDER)
            console.print("[cyan]Downloading delineations... please wait[/cyan]")
            download_mya_case_study_inputs(
                INPUT_FOLDER_LINK, os.path.join(INPUT_FOLDER, "reemission_demo_delineations.zip")
            )
    else:
        console.print(f"[red]Delineations folder not found.[/red] Creating and downloading...")
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        download_mya_case_study_inputs(
            INPUT_FOLDER_LINK, os.path.join(INPUT_FOLDER, "reemission_demo_delineations.zip")
        )

    console.print(Rule("[bold yellow]3. Preparing outputs folder[/bold yellow]"))
    if not os.path.isdir(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
        console.print(f"[green]✓ Created folder: {OUTPUTS_FOLDER}[/green]")
    else:
        console.print(f"[cyan]Folder already exists: {OUTPUTS_FOLDER}[/cyan]")

    console.print(Rule("[bold yellow]4. Merging tabular data[/bold yellow]"))
    shp_folders = [f.path for f in os.scandir(INPUT_FOLDER) if f.is_dir()]
    combined_csv_file = os.path.join(OUTPUTS_FOLDER, "geocaret_outputs.csv")

    command_1 = ["reemission-geocaret", "process-tab-outputs"]
    for input_folder in shp_folders:
        command_1.extend(["-i", os.path.join(input_folder, "output_parameters.csv")])

    missing_col_value_pairs: List[Tuple[str, str]] = [
        ("c_treatment_factor", "primary (mechanical)"),
        ("c_landuse_intensity", "low intensity"),
        ("type", "unknown"),
    ]
    for col_name, col_val in missing_col_value_pairs:
        command_1.extend(["-cv", col_name, col_val])

    command_1.extend(["-o", combined_csv_file])

    with Progress() as progress:
        task = progress.add_task("[cyan]Merging tabular data...", total=None)
        run_rich_command(command_1)
        progress.update(task, completed=True)

    console.print(Rule("[bold yellow]5. Merging shape files[/bold yellow]"))
    command_2 = ["reemission-geocaret", "join-shapes"]
    for input_folder in shp_folders:
        command_2.extend(["-i", input_folder])
    command_2 += [
        "-o", OUTPUTS_FOLDER,
        "-gp", "R_*.shp, C_*.shp, MS_*.shp, PS_*.shp",
        "-f", "reservoirs.shp, catchments.shp, rivers.shp, dams.shp",
    ]
    run_rich_command(command_2)

    console.print(Rule("[bold yellow]6. Converting tabular data to RE-Emission input JSON[/bold yellow]"))
    command_3 = [
        "reemission-geocaret", "tab-to-json",
        "-i", combined_csv_file,
        "-o", os.path.join(OUTPUTS_FOLDER, "reemission_inputs.json"),
    ]
    run_rich_command(command_3)

    console.print(Rule("[bold yellow]7. Preparing RE-Emission outputs folder[/bold yellow]"))
    if not os.path.isdir(REEMISSION_OUTPUTS_FOLDER):
        os.makedirs(REEMISSION_OUTPUTS_FOLDER, exist_ok=True)
        console.print(f"[green]✓ Created folder: {REEMISSION_OUTPUTS_FOLDER}[/green]")
    else:
        console.print(f"[cyan]Folder already exists: {REEMISSION_OUTPUTS_FOLDER}[/cyan]")

    console.print(Rule("[bold yellow]8. Calculating GHG emissions with RE-Emission[/bold yellow]"))
    command_4 = [
        "reemission", "calculate",
        os.path.join(OUTPUTS_FOLDER, "reemission_inputs.json"),
        "-a", 'Default User',
        "-t", 'Demo Example Results',
        "-o", os.path.join(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs.html"),
        "-o", os.path.join(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs.json"),
        "-o", os.path.join(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs_.xlsx"),
    ]
    with Status("Running emission calculations...", spinner="earth"):
        run_rich_command(command_4)

    console.print(Rule("[bold yellow]9. Postprocessing and map generation[/bold yellow]"))
    process_mya_case_study_results(
        shp_folder=pathlib.Path(OUTPUTS_FOLDER),
        output_json_file=pathlib.Path(REEMISSION_OUTPUTS_FOLDER, "demo_GHG_outputs.json"),
        map_path=pathlib.Path(demo_folder, "demo_interactive_map"),
        ifc_dam_path=pathlib.Path(IFC_DB_FOLDER, "dam_db.shp"),
    )

    console.print(
        Panel.fit(
            "[bold green]🎉 Demo complete![/bold green]\n"
            f"Results saved in: [cyan]{demo_folder}[/cyan]",
            box=box.ROUNDED,
        )
    )

main.add_command(calculate)
main.add_command(log_to_pdf)
main.add_command(run_demo)
main.add_command(integration_cli.geocaret_integrate)
