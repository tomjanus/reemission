Usage
=====

.. _GeoCARET: https://github.com/Reservoir-Research/geocaret

RE-Emission can be run as a *Toolbox* or from a *Command Line*.

As a Toolbox
------------

For example, to calculate emissions for a number of reservoirs for which the input data is located in ``examples/simple_example/test_input.json`` run the code below.

.. note::
   The code produces outputs that are configured in ``outputs.yaml`` configuration file.
   For more details, please refer to :doc:`configuration` section :ref:`output-and-reporting-config`.

.. code-block:: Python

   import pprint
   # Import reemission utils module
   import reemission.utils as utils
   # Import EmissionModel class from the `model` module
   from reemission.model import EmissionModel
   # Import Inputs class from the `input` module
   from reemission.input import Inputs
   # Run a simple example input file from the /examples/ suite
   input_data = Inputs.fromfile(
       utils.get_package_file('../../examples/simple_example/test_input.json')
   )
   output_config = utils.get_package_file('config/outputs.yaml')
   model = EmissionModel(inputs=input_data, config=output_config)
   model.calculate()
   pprint.pprint(model.outputs)

More examples can be found in :doc:`tutorials/tutorials`.

Command-Line Interface (CLI)
----------------------------

RE-Emission provides a command-line functionality for selected features, e.g. calculating emisions from batches of reservoirs, running a demo, and integration with GeoCARET_, e.g. joining multiple ``.shp`` files, converting GeoCARET_ output files in the tabular form in the ``.csv`` format into RE-Emission's input ``JSON`` format, etc. 

The usage of the command line interface functions can be explored by typing ``--help`` after each function. To list all functions in the ``reemission`` **CLI**, type ``reemission --help``. The following listing should be produced:

.. code-block:: bash

   ❯ reemission --help
                                                                                 
   Usage: reemission [OPTIONS] COMMAND [ARGS]...                                  
                                                                                 
   Package reemission v1.0.0                                                      
   ------------------------ RE-EMISSION  ------------------------                 
                                                                                 
   You are now using the Command line interface of RE-Emission, a Python toolbox  
   for calculating greenhouse gas emissions from reservoirs..                     
                                                                                 
   See the full documentation at : https://reemisison.readthedocs.io/en/latest/.  
                                                                                 
   ╭─ Options ────────────────────────────────────────────────────────────────────╮
   │ --help      Show this message and exit.                                      │
   ╰──────────────────────────────────────────────────────────────────────────────╯
   ╭─ Commands ───────────────────────────────────────────────────────────────────╮
   │ calculate           Calculates emissions based on the data in the JSON       │
   │                     INPUT_FILE. Saves the results to output file(s) defined  │
   │                     in option '--output-files'. Two types of output files    │
   │                     are available: '.json' and 'tex/pdf'. 'pdf' files are    │
   │                     written using latex intermediary. Latex source files are │
   │                     saved alongside 'pdf' files.                             │
   │ geocaret-integrate  Integration of RE-Emission with GeoCARET outputs         │
   │ log-to-pdf          Converts log in text format into a PDF                   │
   │ run-demo            Run a demo analysis for a set of existing and future     │
   │                     dams.                                                    │
   ╰──────────────────────────────────────────────────────────────────────────────╯

The usage for the calculate function can be explored by typing ``reemission calculate --help``. It should produce the following listing:

.. code-block:: sh

   ❯ reemission calculate --help
   ____  _____      _____           _         _             
   |  _ \| ____|    | ____|_ __ ___ (_)___ ___(_) ___  _ __  
   | |_) |  _| _____|  _| | '_ ` _ \| / __/ __| |/ _ \| '_ \ 
   |  _ <| |__|_____| |___| | | | | | \__ \__ \ | (_) | | | |
   |_| \_\_____|    |_____|_| |_| |_|_|___/___/_|\___/|_| |_|
                                                            

                                                                                                                                                                                                
   Usage: reemission calculate [OPTIONS] INPUT_FILE                                                                                                                                             
                                                                                                                                                                                                
   Calculates emissions based on the data in the JSON INPUT_FILE. Saves the results to output file(s) defined in option '--output-files'. Two types of output files are available: '.json' and  
   'tex/pdf'. 'pdf' files are written using latex intermediary. Latex source files are saved alongside 'pdf' files.                                                                             
   Args: input_file: JSON file with information about catchment and reservoir related inputs. output_files: Paths of outputs files. output_config: YAML output configuration file. author:      
   Author's name. title: Report/Study title. p_model: Method for estimating phosphorus loading to reservoirs n2o_model: Model for estimating N2O emissions.                                     
                                                                                                                                                                                                
   ╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
   │ --output-files   -o  PATH  Files the outputs are written to.                                                                                                                               │
   │ --output-config  -c  PATH  RE-Emission output configuration file.                                                                                                                          │
   │ --author         -a  TEXT  Author's name                                                                                                                                                   │
   │ --title          -t  TEXT  Report/Study title                                                                                                                                              │
   │ --p-model        -p  TEXT  P-calculation method for CO2 emissions: g-res/mcdowell                                                                                                          │
   │ --n2o-model      -n  TEXT  Model for calculating N2O emissions: model_1/model_2                                                                                                            │
   │ --help                     Show this message and exit.                                                                                                                                     │
   ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


To run a simple estimation of emissions for hypothetical reservoir input data in ``input.json`` and save the outputs in three formats (``pdf``, ``json`` and ``xlsx``) type the following:

.. code-block:: sh

    ❯ reemission calculate -o output.pdf -o output.json -output uk.xlsx --author "John Smith" --title "Reservoir Emissions Analysis" input.json
