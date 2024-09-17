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
    </li>
    <li><a href="#installation">Basic Installation</a></li>
      <ul>
        <li><a href="#from-github">From GitHub</a></li>
      </ul>
    <li><a href="#usage">Usage</a></li>
    <ul>
      <li><a href="#as-a-toolbox">As a Toolbox</a></li>
      <li><a href="#jupyter-notebook-examples">Jupyter Notebook Examples</a></li>
      <li><a href="#using-command-line-interface-(cli)">Using Command Line Interface (CLI)</a></li>
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

<!-- REQUIREMENTS -->
## Requirements

### Python Version

RE-Emission requires Python 3.10 or newer.

### LaTeX Installation (Optional)

If you would like to generate output documents in a PDF format, you will need to install LaTeX. Without LaTeX, upon an attempt to compile the generated LaTeX source code to PDF, ```pylatex``` library implemented in this software will throw ```pylatex.errors.CompilerError```. LaTeX source file with output results will still be created but it will not be able to get compiled to PostScript or PDF.

LaTeX installation guidelines can be found alonside the software installation guidelines in the documentation [Documentation](https://tomjanus.github.io/reemission/install.html)

## Basic Installation

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
      pip install .
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
RE-Emission has two CLI interfaces: `reemission` for performing greenhouse gas emission calculations and `reemission-geocaret` for processing outputs obtained from an upstream reservoir and catchment delineation tool HEET and creating input files to RE-Emission.
For more information about the usage, type in Terminal/Console:
```bash
reemission --help
```
and 
```bash
reemission-geocaret --help
```

For more examples, please refer to the [Documentation](https://tomjanus.github.io/reemission/)


<p align="right">(<a href="#top">back to top</a>)</p>

## :books: Documentation

The software documentation can be accessed [here](https://tomjanus.github.io/reemission/)

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
