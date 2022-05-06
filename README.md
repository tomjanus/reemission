<div id="top"></div>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<p align="center">
    <img alt="reemission-logo" height="120" src="https://github.com/tomjanus/reemission/blob/master/graphics/logo-banner-bw.png?raw=true"/>
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
*Re-Emission* is a Python library and a command line interface (CLI) tool for estimating **CO<sub>2</sub>**, **CH<sub>4</sub>** and **N<sub>2</sub>O** emissions from man-made reservoirs.
It calculates full life-cycle emissions as well as emission profiles over time for each of the three greenhouse gases.

### :fire: Features
* Calculates CO<sub>2</sub>, CH<sub>4</sub> and N<sub>2</sub>O emissions for a single reservoir and for batches of reservoirs.
* Two reservoir Phosphorus mass balance calculation methods in CO<sub>2</sub> emission calculations: G-Res method and McDowell method.
* Two N<sub>2</sub>O calculation methods.
* Model parameters, and presentation of outputs are fully configurable using YAML files.
* Inputs can be constructed in Python using the ```Input``` class or read from JSON files.
* Outputs can be presented in JSON, LaTeX and PDF format and are configurable in the ```outputs.yaml``` configuration file.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- PREREQUISITES -->
## Prerequisites

If you would like to generate output documents in a PDF format, you will need to install LaTeX. Without LaTeX, upon an attempt to compile the generated LaTeX source code to PDF, ```pylatex``` library implemented in this software will throw ```pylatex.errors.CompilerError```. LaTeX source file with output results will still be created but it will not be able to get compiled to PostScript or PDF.

### LaTeX installation guidelines

#### Debian-based Linux Distributions
For basic LaTex version (recommended)
```bash
sudo apt install texlive
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
pip install re-mission
```

Type
```bash
pip install reemission -r requirements.txt -e .
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
For calculation of emissions for a number of reservoirs with input data in ```inputs.json``` file and output configuration in ```outputs.yaml``` file.
```python
import reemission
# Import from the model module
from reemission.model import EmissionModel
# Import from the input module
from reemission.input import Inputs
input_data = Inputs.fromfile('reemission/tests/test_data/inputs.json')
output_config = 'reemission/config/emissions/outputs.yaml'
model = EmissionModel(inputs=input_data, config=output_config)
model.calculate()
print(mode.outputs)
```

#### Jupyter Notebook Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomjanus/reemission/blob/master/notebooks/index.ipynb)

#### Using Command Line Interface (CLI)
In Terminal/Console
```bash
reemission [input-file] [output-file]
```

For more examples, please refer to the [Documentation](https://example.com)

### Example inputs
#### Input JSON file

```json
{
    "Reservoir 1":
    {
        "monthly_temps": [10.56,11.99,15.46,18.29,20.79,22.09,22.46,22.66,
                          21.93,19.33,15.03,11.66],
        "year_vector": [1, 5, 10, 20, 30, 40, 50, 65, 80, 100],
        "gasses": ["co2", "ch4", "n2o"],
        "catchment":
        {
            "runoff": 1685.5619,
            "area": 78203.0,
            "population": 8463,
            "area_fractions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.01092, 0.11996,
                               0.867257],
            "slope": 8.0,
            "precip": 2000.0,
            "etransp": 400.0,
            "soil_wetness": 140.0,
            "biogenic_factors":
            {
                "biome": "TROPICALMOISTBROADLEAF",
                "climate": "TROPICAL",
                "soil_type": "MINERAL",
                "treatment_factor": "NONE",
                "landuse_intensity": "LOW"
            }
        },
        "reservoir":{
            "volume": 7663812,
            "area": 0.56470,
            "max_depth": 32.0,
            "mean_depth": 13.6,
            "area_fractions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "soil_carbon": 10.228
        }
    }
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Example outputs
#### Outputs in JSON format
```json
{
    "Reservoir 1": {
        "co2_diffusion": 243.65,
        "co2_diffusion_nonanthro": 167.25,
        "co2_preimp": -140.0,
        "co2_minus_nonanthro": 76.40,
        "co2_net": 216.40,
        "co2_profile": [
            737.05,
            422.16,
            330.28,
            257.19,
            221.57,
            199.04,
            182.98,
            165.54,
            152.78,
            140.00
        ],
        "ch4_diffusion": 95.09,
        "ch4_ebullition": 83.52,
        "ch4_degassing": 361.83,
        "ch4_preimp": 0.00,
        "ch4_net": 540.44,
        "ch4_profile": [
            1585.01,
            1399.71,
            1199.89,
            886.67,
            661.33,
            499.21,
            382.58,
            266.01,
            194.88,
            141.16
        ],
        "n2o_methodA": 1.198,
        "n2o_methodB": 1.332,
        "n2o_mean": 1.265,
        "n2o_profile": [
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20
        ]
    }
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
c_1 = 1.8569682
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
[MIT](https://choosealicense.com/licenses/mit/)

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
- Tomasz Janus -tomasz.janus@manchester.ac.uk
- Christopher Barry - c.barry@ceh.ac.uk
- Jaise Kuriakose - jaise.kuriakose@manchester.ac.uk

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
