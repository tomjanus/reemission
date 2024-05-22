# RE-EMISSION examples and use cases
This module contains the examples of the usage of `RE-EMISSION` as a standalone command-line (CLI) tool and as a Python API, i.e. collection of classes and methods for processing input data, estimating GHG emissions and outputting results in the form of raw output files in `JSON` format, formatted output files/reports in `Excel/CSV` and `LaTeX/PDF` formats, as well as in the form of interactive maps, combining the tabular input and output data with geospatial data, i.e. delineated reservoirs and catchments.

## Standalone applications
The use of `RE-EMISSION` as a standalone application is showcased in two examples. To run each example, navigate to the corresponding folder and execute the script file dedicated to your operating system. On Linux/Mac OS execute the `run_*.sh` file. On Windows use the `run_*.bat` script.

### 1. Simple example (`examples/simple_example` folder)
Reads input file in `JSON` format, calculates GHG emissions for a number of reservoirs and outputs the results in three file formats: `JSON`, `PDF`, and `XLSX`. 
**Note:** Saving outputs to `XLSX` format is still experimental.

### 2. Demo (`src/reemission/demo` folder)
This is a short example of how `RE-EMISSION` works in combintion with an `upstream` reservoir and catchment delineation tool `GeoCARET`. This example showcases `RE-EMISSION`'s integration capabilities with `GeoCARET` on a small example from a larger real-life case study on estimating GHG emissions from existing and planned hydroelectric reservoirs in Myanmar. 
The demo has been moved to `src/reemission/demo` i.e. to the main package installation folder.
The demo runs in the following steps:
* Merging multiple tabular data files from several batches of reservoir delineations in `GeoCARET` into a single `CSV` file.
* Merging shape files for individual reservoirs and catchments into combined shape files representing reservoirs and catchments for all items in the study.
* Converting the merged tabular data from `GeoCARET` into `RE-EMISSION` JSON input file.
* Calculating GHG emissions with RE-EMISSION taking the JSON input file generated in the previous step.
* Pasting selected `GeoCARET` tabular input data and `RE-EMISSIONS` output data into the combined shape files for reservoirs and dams and presenting the updated shape files in the form of an interactive map using `Folium`.

## Manual problem formulation / Python API (`examples/notebooks` folder)
The usage of the RE-Emission Toolbox and its various components is demonstrated in four notebooks:

<font size="3"> 1. [Manual Step-By-Step Calculations](notebooks/01-Step-By-Step-Manual-Calculations.ipynb)</font> shows how to manually construct input data structures for a hypotethical reservoir and calculate GHG emission estimates.

<font size="3"> 2. [Automatic Calculation Of Batches of Reservoirs](notebooks/02-Automatic-Calculation-Of-Emissions-For-Batches-Of-Reservoirs.ipynb)</font> demonstrates how to: read input data in <b>JSON</b> format and output configuration <b>YAML</b> file, instantiate the emission model object from input data output configuration file, calculate emissions and display model outputs.

<font size="3"> 3. [Presentation of Results in JSON Format](notebooks/03-Saving-Results-To-JSON.ipynb)</font> demonstrates how to: read input data and output configuration <b>YAML</b> file and instantiate the emission model, add <b>JSON</b> model presenter, calculate emissions and save results to <b>JSON</b> file, read and display the results saved in <b>JSON</b> format.

<font size="3"> 4. [Presentation of Results in PDF Format (via $\LaTeX$)](notebooks/04-Saving-Results-To-LaTeX.ipynb)</font> shows how to: read input data and output configuration <b>YAML</b> file and instantiate the emission mode, add $\LaTeX$ model presenter, calculate emissions and save results to $\TeX$ and <b>PDF</b> files, and open the generated <b>PDF</b> document.
