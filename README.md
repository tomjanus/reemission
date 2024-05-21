<div id="top"></div>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL-3.0 License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<p align="center">
    <img alt="reemission-logo" height="120" src="https://user-images.githubusercontent.com/8837107/228694371-1aac24c7-97a8-4e8b-98b7-f01e63410c01.png"/>
</p>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-library">About The Library</a>
      <ul>
        <li><a href="#features">Features</a></li>
      </ul>
    </li>
    <li>
      <a href="#prerequisites">Prerequisites</a>
      <ul>
        <li><a href="#latex-installation-guidelines">LaTeX installation guidelines</a></li>
        <ul>
           <li><a href="#debian-based-linux-distributions">Debian-based Linux Distributions</a></li>
           <li><a href="#mac-os">Mac OS</a></li>
           <li><a href="#windows">Windows</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
      <ul>
        <li><a href="#from-pypi">From PyPi</a></li>
        <li><a href="#from-github">From GitHub</a></li>
      </ul>
    <li><a href="#usage">Usage</a></li>
    <ul>
      <li><a href="#as-a-toolbox">As a Toolbox</a></li>
      <li><a href="#jupyter-notebook-examples">Jupyter Notebook Examples</a></li>
      <li><a href="#using-command-line-interface-(cli)">Using Command Line Interface (CLI)</a></li>
    </ul>
    <li><a href="#example-inputs">Example inputs</a></li>
    <ul>
    <li><a href="#input-json-file">Input JSON file</a></li>
    </ul>
    <li><a href="#example-outputs">Example outputs</a></li>
    <ul>
    <li><a href="#outputs-in-json-format">Outputs in JSON format</a></li>
    <li><a href="#outputs-in-a-PDF-report-format">Outputs in a PDF report format</a></li>
    </ul>
      <li><a href="#configuration">Configuration</a></li>
        <ul>
          <li><a href="#configuration-of-inputs">Configuration of inputs</a></li>
          <li><a href="#configuration-of-outputs">Configuration of outputs</a></li>
          <li><a href="#configuration-of-global-parameters">Configuration of global parameters</a></li>
          <li><a href="#model-coefficients">Model coefficients</a></li>
        </ul>    
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citing">Citing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
      <ul>
        <li><a href="#institutions">Institutions</a></li>
        <li><a href="#resources">Resources</a></li>
      </ul>
    <li><a href="#references">References</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Library
*Re-Emission* is a Python library and a command line interface (CLI) tool for estimating **CO<sub>2</sub>**, **CH<sub>4</sub>** and **N<sub>2</sub>O** emissions from reservoirs.
It calculates full life-cycle emissions as well as emission profiles over time for each of the three greenhouse gases.

### :fire: Features
* Calculates CO<sub>2</sub>, CH<sub>4</sub> and N<sub>2</sub>O emissions for a single reservoir and for batches of reservoirs.
* Two reservoir Phosphorus mass balance calculation methods in CO<sub>2</sub> emission calculations: G-Res method and McDowell method.
* Two N<sub>2</sub>O calculation methods.
* Model parameters, and presentation of outputs are fully configurable using YAML configuration files.
* Inputs can be constructed in Python using the ```Input``` class or read from JSON files.
* Outputs in tabular form can be presented in JSON, LaTeX and PDF formats and can be configured by changing settings in the ```outputs.yaml``` configuration file.
* Integrates with the upstream catchment and reservoir delineation package HEET, whcih is currently in Beta version and undergoing development.
* Combines tabular and GIS inputs from catchment delineation with gas emission outputs and visualizes the combined data in interactive maps.

### A quick demo of results from RE-Emission using input data from catchment delineation tool HEET

Preliminary results of our first case study (for presentation use only), are shown in [https://tomjanus.github.io/mya_emissions_map/](https://tomjanus.github.io/mya_emissions_map/). The case study looks into an assessment of gas emissions from existing and planned hydroelectric reservoirs in Myanmar. A snapshot of the map is presented below. 
<p align="center">
    <a href="https://tomjanus.github.io/mya_emissions_map/" target="_blank" rel="noopener noreferrer"><img alt="myanmar_hydro-map" width="650" src="https://github.com/tomjanus/reemission/assets/8837107/96a643d4-990c-451a-9d91-a83655b0be47"/></a></p>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- PREREQUISITES -->
## Prerequisites

If you would like to generate output documents in a PDF format, you will need to install LaTeX. Without LaTeX, upon an attempt to compile the generated LaTeX source code to PDF, ```pylatex``` library implemented in this software will throw ```pylatex.errors.CompilerError```. LaTeX source file with output results will still be created but it will not be able to get compiled to PostScript or PDF.

### LaTeX installation guidelines

#### Debian-based Linux Distributions
For basic LaTeX version (recommended)
```bash
sudo apt install texlive
```
`texlive` requires additional manual installation of the following two packages: `type1ec.sty` and `siunitx.sty`. These two packages can be installed by issuing the following commands in the Terminal:
```bash
sudo apt install cm-super && sudo apt install texlive-science
```
For full LaTeX version with all packages (requires around 2GB to download and 5GB free space on a local hard drive)
```bash
sudo apt install texlive-full
```


#### Mac OS
BasicTeX (100MB) - minimum install without editor
```brew
brew install --cask basictex
```
MacTeX with built-in editor (3.2GB) - uses TeXLive
```brew
brew install --cask mactex
```

#### Windows
For easy install, download and run [install-tl-windows.exe](https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe)
For more installation options, visit [https://tug.org/texlive/windows.html](https://tug.org/texlive/windows.html). Or, make your life easier by getting yourself a Linux. :smirk:

<p align="right">(<a href="#top">back to top</a>)</p>

## Installation

### From PyPi

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ```reemission```.

```bash
pip install reemission
```

Type
```bash
pip install reemission -e .
```
if you'd like to use the package in a development mode.


### From GitHub
1. Clone the repository using either:
   - HTTPS
   ```sh
   git clone https://github.com/tomjanus/reemission.git
   ```
   - SSH
   ```sh
   git clone git@github.com:tomjanus/reemission.git
   ```
2. Install from source:
   - for development
      ```sh
      pip install -r requirements.txt -e .
      ```
   - or as a build
      ```bash
      pip install build .
      ```

        or

      ```sh
      python3 -m build --sdist --wheel .
      ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

#### As a toolbox
For calculation of emissions for a number of reservoirs with input data in ```test_input.json``` file and output configuration in ```outputs.yaml``` file.
```python
import pprint
# Import reemission utils module
import reemission.utils as utils
# Import EmissionModel class from the `model` module
from reemission.model import EmissionModel
# Import Inputs class from the `input` module
from reemission.input import Inputs
# Run a simple example input file from the /examples/ suite
input_data = Inputs.fromfile(utils.get_package_file('../../examples/simple_example/test_input.json'))
output_config = utils.get_package_file('config/outputs.yaml')
model = EmissionModel(inputs=input_data, config=output_config)
model.calculate()
pprint.pprint(model.outputs)
```

#### Jupyter Notebook Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb)

#### Using Command Line Interface (CLI)
RE-Emission has two CLI interfaces: `reemission` for performing greenhouse gas emission calculations and `reemission-heet` for processing outputs obtained from an upstream reservoir and catchment delineation tool HEET and creating input files to RE-Emission.
For more information about the usage, type in Terminal/Console:
```bash
reemission --help
```
and 
```bash
reemission-heet --help
```

For more examples, please refer to the [Documentation](https://example.com)

### Example inputs
#### Input JSON file for a single reservoir

```json
{    
    "Reservoir 1": {
        "coordinates": [23.698, 97.506],
        "monthly_temps": [13.9, 16.0, 19.3, 22.8, 24.2, 24.5,
                          24.2, 24.3, 23.9, 22.1, 18.5, 14.8],
        "year_vector": [1, 5, 10, 20, 30, 40, 50, 65, 80, 100],
        "gasses": ["co2", "ch4", "n2o"],
        "catchment": {
            "runoff": 1115.0,
            "area": 12582.613,
            "riv_length": 0.0,
            "population": 1587524.0,
            "area_fractions": [0.000, 0.000, 0.003, 0.002,
                               0.001, 0.146, 0.391, 0.457, 0.000],
            "slope": 23.0,
            "precip": 1498.0,
            "etransp": 1123.0,
            "soil_wetness": 144.0,
            "mean_olsen": 5.85,
            "biogenic_factors": {
                "biome": "tropical moist broadleaf",
                "climate": "temperate",
                "soil_type": "mineral",
                "treatment_factor": "primary (mechanical)",
                "landuse_intensity": "low intensity"
            }
        },
        "reservoir": {
            "volume": 7238166.0,
            "area": 1.604,
            "max_depth": 22.0,
            "mean_depth": 4.5,
            "area_fractions": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.15, 0.4, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0],
            "soil_carbon": 6.281,
            "mean_radiance": 4.66,
            "mean_radiance_may_sept": 4.328,
            "mean_radiance_nov_mar": 4.852,
            "mean_monthly_windspeed": 1.08,
            "water_intake_depth": null
        }
    },
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Example outputs
#### `model.outputs` dictionary
```json
{"Reservoir 2": {
    "ch4_degassing": 769.76,
    "ch4_diffusion": 230.98,
    "ch4_ebullition": 210.60,
    "ch4_net": 1211.35,
    "ch4_preimp": 0.0,
    "ch4_profile": [3525.45, 3119.39, 2681.04, 1992.56, 1495.95, 1137.74, 879.35, 620.29, 461.58, 341.18],
    "ch4_total_lifetime": 5353.45,
    "ch4_total_per_year": 53534.55,
    "co2_diffusion": 994.36,
    "co2_diffusion_nonanthro": 682.41,
    "co2_minus_nonanthro": 311.95,
    "co2_net": 311.95,
    "co2_preimp": 0.00,
    "co2_profile": [2436.81, 1151.53, 776.55, 478.25,332.89, 240.93, 175.38, 104.24, 52.14, 0.00],
    "co2_total_lifetime": 1378.64,
    "co2_total_per_year": 13786.49,
    "n2o_mean": 3.610,
    "n2o_methodA": 3.61,
    "n2o_methodB": 2.24,
    "n2o_profile": [3.61, 3.61, 3.61, 3.61, 3.61, 3.61, 3.61, 3.61, 3.61, 3.61],
    "n2o_total_lifetime": 15.95,
    "n2o_total_per_year": 159.54}
}
```

#### Outputs in JSON format
This is a formatted output format containing the input and the output data including variable names and units. 
```json
{
    "Reservoir 3": {
        "inputs": {
            "coordinates": {
                "name": "Reservoir coordinates (lat/lon)",
                "unit": "deg",
                "value": [23.698,97.506]},
            "monthly_temps": {
                "name": "Monthly Temperatures",
                "unit": "deg C",
                "value": [13.9,16.0,19.3,22.8,24.2,24.5,24.2,24.3,23.9,22.1,18.5,14.8]},
            "year_profile": {
                "name": "Year vector for emission profiles",
                "unit": "yr",
                "value": [1,5,10,20,30,40,50,65,80,100]},
            "gasses": {
                "name": "Calculated gas emissions",
                "unit": "-",
                "value": ["co2","ch4","n2o"]},
            "biogenic_factors": {
                "name": "Biogenic factors",
                "biome": {
                    "name": "Biome",
                    "unit": "",
                    "value": "tropical moist broadleaf"},
                "climate": {
                    "name": "Climate",
                    "unit": "",
                    "value": "temperate"},
                "soil_type": {
                    "name": "Soil Type",
                    "unit": "",
                    "value": "mineral"},
                "treatment_factor": {
                    "name": "Treatment Factor",
                    "unit": "",
                    "value": "primary (mechanical)"},
                "landuse_intensity": {
                    "name": "Landuse Intensity",
                    "unit": "",
                    "value": "low intensity"}},
            "catchment_inputs": {
                "name": "Inputs for catchment-level process calculations",
                "runoff": {
                    "name": "Annual runoff",
                    "unit": "mm/year",
                    "value": 1115.0},
                "area": {
                    "name": "Catchment area",
                    "unit": "km2",
                    "value": 12582.613},
                "riv_length": {
                    "name": "Length of inundated river",
                    "unit": "km",
                    "value": 0.0},
                "population": {
                    "name": "Population",
                    "unit": "capita",
                    "value": 1587524.0},
                "area_fractions": {
                    "name": "Area fractions",
                    "unit": "-",
                    "value": "0.0, 0.0, 0.003, 0.002, 0.001, 0.146, 0.391, 0.457, 0.0"},
                "slope": {
                    "name": "Mean catchment slope",
                    "unit": "%",
                    "value": 23.0},
                "precip": {
                    "name": "Mean annual precipitation",
                    "unit": "mm/year",
                    "value": 1498.0},
                "etransp": {
                    "name": "Mean annual evapotranspiration",
                    "unit": "mm/year",
                    "value": 1123.0},
                "soil_wetness": {
                    "name": "Soil wetness",
                    "unit": "mm over profile",
                    "value": 144.0},
                "mean_olsen": {
                    "name": "Soil Olsen P content",
                    "unit": "kgP/ha",
                    "value": 5.85}},
            "reservoir_inputs": {
                "name": "Inputs for reservoir-level process calculations",
                "volume": {
                    "name": "Reservoir volume",
                    "unit": "m3",
                    "value": 7238166.0},
                "area": {
                    "name": "Reservoir area",
                    "unit": "km2",
                    "value": 1.604},
                "max_depth": {
                    "name": "Maximum reservoir depth",
                    "unit": "m",
                    "value": 22.0},
                "mean_depth": {
                    "name": "Mean reservoir depth",
                    "unit": "m",
                    "value": 4.5},
                "area_fractions": {
                    "name": "Inundated area fractions",
                    "unit": "-",
                    "value": "0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.15, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"},
                "soil_carbon": {
                    "name": "Soil carbon in inundated area",
                    "unit": "kgC/m2",
                    "value": 6.281},
                "mean_radiance": {
                    "name": "Mean monthly horizontal radiance",
                    "unit": "kWh/m2/d",
                    "value": 4.66},
                "mean_radiance_may_sept": {
                    "name": "Mean monthly horizontal radiance: May - Sept",
                    "unit": "kWh/m2/d",
                    "value": 4.328},
                "mean_radiance_nov_mar": {
                    "name": "Mean monthly horizontal radiance: Nov - Mar",
                    "unit": "kWh/m2/d",
                    "value": 4.852},
                "mean_monthly_windspeed": {
                    "name": "Mean monthly wind speed",
                    "unit": "m/s",
                    "value": 1.08},
                "water_intake_depth": {
                    "name": "Water intake depth below surface",
                    "unit": "m",
                    "value": null}
            }
        },
        "outputs": {
            "co2_diffusion": {
                "name": "CO2 diffusion flux",
                "gas_name": "CO2",
                "unit": "gCO2eq m-2 yr-1",
                "long_description": "Total CO2 emissions from a reservoir integrated over lifetime",
                "value": 572.82},
            "co2_diffusion_nonanthro": {
                "name": "Nonanthropogenic CO2 diffusion flux",
                "gas_name": "CO2",
                "unit": "gCO2eq m-2 yr-1",
                "long_description": "CO2 diffusion flux taken at (after) 100 years",
                "value": 393.12},
            "co2_preimp": {
                "name": "Preimpoundment CO2 emissions",
                "gas_name": "CO2",
                "unit": "gCO2eq m-2 yr-1",
                "long_description": "CO2 emission in the area covered by the reservoir prior to impoundment",
                "value": 0.0},
            "co2_minus_nonanthro": {
                "name": "CO2 emission minus non-anthropogenic",
                "gas_name": "CO2",
                "unit": "gCO2eq m-2 yr-1",
                "long_description": "CO2 emissions minus non-anthropogenic over a number of years",
                "value": 179.71},
            "co2_net": {
                "name": "Net CO2 emission",
                "gas_name": "CO2",
                "unit": "gCO2eq m-2 yr-1",
                "long_description": "Overall integrated emissions for lifetime",
                "value": 179.71},
            "co2_total_per_year": {
                "name": "Total CO2 emission per year",
                "gas_name": "CO2",
                "unit": "tCO2eq yr-1",
                "long_description": "Total CO2 emission per year integrated over lifetime",
                "value": 288.25},
            "co2_total_lifetime": {
                "name": "Total CO2 emission per lifetime",
                "gas_name": "CO2",
                "unit": "tCO2eq",
                "long_description": "Total CO2 emission integrated over lifetime",
                "value": 28.83},
            "co2_profile": {
                "name": "CO2 emission profile",
                "gas_name": "CO2",
                "unit": "gCO2eq m-2 yr-1",
                "long_description": "CO2 emission per year for a defined list of years",
                "value": [1403.78,663.36,447.35,275.51,191.77,138.8,101.04,60.05,30.04,0.0]},
            "ch4_diffusion": {
                "name": "CH4 emission via diffusion",
                "gas_name": "CH4",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "CH4 emission via diffusion integrated over a number of years.",
                "value": 222.13},
            "ch4_ebullition": {
                "name": "CH4 emission via ebullition",
                "gas_name": "CH4",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "CH4 emission via ebullition",
                "value": 321.23},
            "ch4_degassing": {
                "name": "CH4 emission via degassing",
                "gas_name": "CH4",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "CH4 emission via degassing integrated for a number of years",
                "value": 3857.24},
            "ch4_preimp": {
                "name": "Pre-impounment CH4 emission",
                "gas_name": "CH4",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "Pre-impounment CH4 emission",
                "value": 0.0},
            "ch4_net": {
                "name": "Net CH4 emission",
                "gas_name": "CH4",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "Net per area CH4 emission",
                "value": 4400.6},
            "ch4_total_per_year": {
                "name": "Total CH4 emission per year",
                "gas_name": "CH4",
                "unit": "tCO2eq yr-1",
                "long_description": "Total CH4 emission per year integrated over lifetime",
                "value": 7058.56},
            "ch4_total_lifetime": {
                "name": "Total CH4 emission per lifetime",
                "gas_name": "CH4",
                "unit": "ktCO2eq",
                "long_description": "Total CH4 emission integrated over lifetime",
                "value": 705.86},
            "ch4_profile": {
                "name": "CH4 emission profile",
                "gas_name": "CH4",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "CH4 emission per year for a defined list of years",
                "value": [13754.64,12109.16,10332.79,7542.77,5530.28,4078.62,3031.52,1981.61,1338.42,850.48]},
            "n2o_methodA": {
                "name": "N2O emission, method A",
                "gas_name": "N2O",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "N2O emission, method A",
                "value": 0.04},
            "n2o_methodB": {
                "name": "N2O emission, method B",
                "gas_name": "N2O",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "N2O emission, method B",
                "value": 0.05},
            "n2o_mean": {
                "name": "N2O emission, mean value",
                "gas_name": "N2O",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "N2O emission factor, average of two methods",
                "value": 0.04},
            "n2o_total_per_year": {
                "name": "Total N2O emission per year",
                "gas_name": "N2O",
                "unit": "tCO2eq yr-1",
                "long_description": "Total N2O emission per year integrated over lifetime",
                "value": 0.07},
            "n2o_total_lifetime": {
                "name": "Total N2O emission per lifetime",
                "gas_name": "N2O",
                "unit": "ktCO2eq",
                "long_description": "Total N2O emission integrated over lifetime",
                "value": 0.01},
            "n2o_profile": {
                "name": "N2O emission profile",
                "gas_name": "N2O",
                "unit": "g CO2eq m-2 yr-1",
                "long_description": "N2O emission per year for a defined list of years",
                "value": [0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]}
        }
    },
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

#### Outputs in a PDF report format
1. Input data in a tabular format in the output report in PDF format

<p align="center"><img style="width: 600px"; src="https://github.com/tomjanus/reemission/blob/master/graphics/inputs_table_from_pdf.png?raw=true" alt="Inputs Table PDF format"></p>

2. Output data in a tabular format in the output report in PDF format

<p align="center"><img style="width: 600px"; src="https://github.com/tomjanus/reemission/blob/master/graphics/outputs_table_from_pdf.png?raw=true" alt="Outputs Table PDF format"></p>

2. Output plots in the output report in PDF format

<p align="center"><img style="width: 600px"; src="https://github.com/tomjanus/reemission/blob/master/graphics/emission_plots_from_pdf.png?raw=true" alt="Output Plots"></p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Configuration
Coefficients of the regressions constituting the model as well as parameters of different categories of soil and land use are stored in a number of **yaml** files in ```parameters/emissions/```.

<p align="right">(<a href="#top">back to top</a>)</p>

### Configuration of inputs

Information about the names and the units of the model inputs is stored and can be configured in ```config/emissions/inputs.yaml```
e.g. for monthly temperatures which are represented in variable ```monthly_temps```:

```yaml
monthly_temps:
  include: True
  name: "Monthly Temperatures"
  long_description: ""
  unit: "deg C"
  unit_latex: "$^o$C"
```

- ```include```: (boolean): If the variable is to be included in the output files for reporting.
- ```name```: (string): Name of the variable
- ```long_description```: (string): Description of the variable
- ```unit```: (string): Unit in text format
- ```unit_latex```: (string): Unit in LaTeX format

Finally, a global flag ```print_long_descriptions``` controls whether long descriptions are included alongside the included input variables in the output files.

<p align="right">(<a href="#top">back to top</a>)</p>

### Configuration of outputs

Similarly to inputs, definitions and units of model outputs and whether they are to be output in the output files, are stored in ```config/emissions/outputs.yaml```, e.g. for pre-impoundment CO<sub>2</sub> emissions defined in variable ```co2_preimp```:
```yaml
co2_preimp:
  include: True
  name: "Preimpoundment CO2 emissions"
  gas_name: "CO2"
  name_latex: "Preimpoundment CO$_2$ emissions"
  unit: "gCO2eq m-2 yr-1"
  unit_latex: "gCO$_2$ m$^{-2}$ yr$^{-1}$"
  long_description: "CO2 emission in the area covered by the reservoir prior to impoundment"
  hint: "Negative values denote C sink (atmosphere to land flux)"
```
- ```include```: (boolean): If the variable is to be included in the output files for reporting.
- ```name```: (string): Name of the variable
- ```gas_name```: (string): Name of the gas the variable is related to
- ```name_latex```: (string): variable name in LaTeX format
- ```unit```: (string): Unit in text format
- ```unit_latex```: (string): Unit in LaTeX format
- ```long_description```: (string): Description of the variable
- ```hint```: (string): Further information about the variable

<p align="right">(<a href="#top">back to top</a>)</p>

### Configuration of global parameters
Information about global parameters such as e.g. Global Warming Potentials ```gwp100``` is stored in ```config/emissions/parameters.yaml```

```yaml
gwp100:
  include: True
  name: "Global Warming Potential for a 100-year timescale"
  name_latex: "Global Warming Potential for a 100-year timescale"
  unit: "-"
  unit_latex: "-"
  long_description: ""
```

### Model coefficients
Values of model coefficients, i.e. regressions used to estimate different gas emissions are stored in ```config/emissions/config.ini``` file. E.g. coefficients for CO<sub>2</sub> emission calculations are listed below.
```ini
[CARBON_DIOXIDE]
# Parameters reated to CO2 emissions
k1_diff = 1.8569682
age = -0.329955
temp = 0.0332459
resArea = 0.0799146
soilC = 0.015512
ResTP = 0.2263344
calc = -0.32996
# Conversion from mg~CO2-C~m-2~d-1 to g~CO2eq~m-2~yr-1
# Based on stoichiometric relationship CO2/C = 44/12 and GWP100 of 1.0
conv_coeff = 1.33833
# Global Warming Potential of CO2 over 100 years
co2_gwp100 = 1.0
```
In addition, various coefficient tables and parameters required to calculate various emission components are stored in multiple YAML files in ```parameters/emissions/```.

## :books: Documentation

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repository and create a pull request. You can also simply open an issue with the tag "*enhancement*".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

## License
[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITING -->
## Citing

If you use RE-Emission for academic research, please cite the library using the following BibTeX entry.

```
@misc{reemission2022,
 author = {Tomasz Janus, Christopher Barry, Jaise Kuriakose},
 title = {RE-Emission: Python tool for calculating greenhouse gas emissions from man-made reservoirs},
 year = {2022},
 url = {https://github.com/tomjanus/reemission},
}
```
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## :mailbox_with_mail: Contact
- Tomasz Janus - <mailto:tomasz.janus@manchester.ac.uk> , <mailto:tomasz.k.janus@gmail.com>
- Christopher Barry - <mailto:c.barry@ceh.ac.uk>
- Jaise Kuriakose - <mailto:jaise.kuriakose@manchester.ac.uk>

Project Link: [https://github.com/tomjanus/reemission](https://github.com/tomjanus/reemission)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

### Institutions
Development of this software was funded, to a large degree, by the [University of Manchester](https://www.manchester.ac.uk/) and the [FutureDams](https://www.futuredams.org/) project.
<table style="border: 0px hidden white;margin-left:auto;margin-right:auto;">
  <tr>
<td align="center"><a href="https://www.manchester.ac.uk/"><img src="https://github.com/tomjanus/reemission/blob/master/graphics/TAB_col_white_background.png?raw=true" height="100px;" alt=""/></td>
<td align="center"><a href="https://www.futuredams.org/"><img src="https://github.com/tomjanus/reemission/blob/master/graphics/futuredams-small.png?raw=true" height="90px;" alt=""/></td>
  </tr>
</table>

<p align="right">(<a href="#top">back to top</a>)</p>

### Resources
* [Best README Template](https://github.com/othneildrew/Best-README-Template)
* [Choose an Open Source License](https://choosealicense.com)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Cookiecutter template for a Python library](https://github.com/ionelmc/cookiecutter-pylibrary)
* [Cookiecutter template for a Python package](https://github.com/audreyfeldroy/cookiecutter-pypackage)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## References

<a id="1">[[1]](https://www.sciencedirect.com/science/article/pii/S0301421504001892)</a> Marco Aurelio dos Santos, Luiz Pinguelli Rosa, Bohdan Sikar, Elizabeth Sikar, Ednaldo Oliveira dos Santos. (2006). *Gross greenhouse gas fluxes from hydro-power reservoir compared to thermo-power plants*. Energy Policy, Volume 34, Issue 4, pp. 481-488, ISSN 0301-421. https://doi.org/10.1016/j.enpol.2004.06.015

<a id="2">[[2]](https://www.pnas.org/doi/10.1073/pnas.1011464108)</a>
Beaulieu, J. J., Tank, J. L., Hamilton, S. K., Wollheim, W. M., Hall, R. O., Mulholland, P. J., Dahm, C. N. (2011). *Nitrous oxide emission from denitrification in stream and river networks*. Proceedings of the National Academy of Sciences of the United States of America, 108(1),
214–219. https://doi.org/10.1073/pnas.1011464108

<a id="3">[[3]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161947)</a>
Scherer, Laura and Pfister, Stephan (2016) *Hydropower's Biogenic Carbon Footprint*. PLOS ONE, Volume 9, 1-11, https://doi.org/10.1371/journal.pone.0161947.

<a id="4">[[4]](https://www.sciencedirect.com/science/article/pii/S1364815221001602)</a>
Yves T. Prairie, Sara Mercier-Blais, John A. Harrison, Cynthia Soued, Paul del Giorgio, Atle Harby, Jukka Alm, Vincent Chanudet, Roy Nahas (2021) *A new modelling framework to assess biogenic GHG emissions from reservoirs: The G-res tool*. Environmental Modelling & Software, Volume 143, 105-117, ISSN 1364-8152, https://doi.org/10.1016/j.envsoft.2021.105117.

<a id="5">[[5]](https://g-res.hydropower.org/wp-content/uploads/2021/10/G-res-Technical-Document-v3.0.pdf)</a> Prairie YT, Alm J, Harby A, Mercier-Blais S, Nahas R. 2017. *The GHG Reservoir Tool (G-res) Technical documentation. Updated version 3.0 (2021-10-27)*. UNESCO/IHA research
project on the GHG status of freshwater reservoirs. Joint publication of the UNESCO Chair in Global Environmental Change and the International Hydropower Association. 73 pages.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributors ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/tomjanus"><img src="https://avatars.githubusercontent.com/tomjanus" width="100px;" alt=""/><br /><sub><b>Tomasz Janus</b></sub></a><br /><a href="https://github.com/tomjanus/reemission/commits?author=tomjanus" title="Code">💻</a><a href="https://github.com/tomjanus/reemission/commits?author=tomjanus" title="Tests">⚠️</a> <a href="https://github.com/tomjanus/reemission/issues/created_by/tomjanus" title="Bug reports">🐛</a><a href="#design-TJanus" title="Design">🎨</a><a href="" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/jojo0094"><img src="https://avatars.githubusercontent.com/jojo0094" width="100px;" alt=""/><br /><sub><b>Aung Kyaw Kyaw</b></sub></a><br /><a href="https://github.com/tomjanus/reemission/commits?author=jojo0094" title="Code">💻</a><a href="https://github.com/tomjanus/reemission/commits?author=jojo0094" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/cdb0101"><img src="https://avatars.githubusercontent.com/cdb0101" width="100px;" alt=""/><br /><sub><b>Chris Barry</b></sub></a><br /><a href="#content-cbarry" title="Methods">🖋</a><a href="#ideas-cbarry" title="Ideas, Planning, & Feedback">🤔</a><a href="" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Climatejaise"><img src="https://avatars.githubusercontent.com/Climatejaise" width="100px;" alt=""/><br /><sub><b>Jaise Kurkakose</b></sub></a><br /><a href="#content-jkuriakose" title="Methods">🖋</a><a href="#ideas-jkuriakose" title="Ideas, Planning, & Feedback">🤔</a><a href="" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/tomjanus/reemission.svg?style=plastic
[contributors-url]: https://github.com/tomjanus/reemission/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/tomjanus/reemission.svg?style=plastic
[forks-url]: https://github.com/tomjanus/reemission/network/members
[stars-shield]: https://img.shields.io/github/stars/tomjanus/reemission.svg?style=plastic
[stars-url]: https://github.com/tjanus/reemission/stargazers
[issues-shield]: https://img.shields.io/github/issues/tomjanus/reemission.svg?style=plastic
[issues-url]: https://github.com/tomjanus/reemission/issues
[license-shield]: https://img.shields.io/github/license/tomjanus/reemission.svg?style=plastic
[license-url]: https://github.com/tomjanus/reemission/blob/master/LICENSE.txt
