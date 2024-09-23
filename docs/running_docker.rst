Running the ReEmission Docker Container
=======================================

Run a GeoCARET analysis
~~~~~~~~~~~~~~~~~~~~~~~

We can use Docker image of ReEmission to run analysis using ReEmission's command-line tools. First, copy your input data file to the ``examples`` sub-folder and then start ReEmission by typing the following:

.. code-block:: bash

   ❯ docker compose run --rm reemission reemission calculate ./examples/[input-file.json] -o ./outputs/[output-file.json] 

where:

* ``[input-file.json]`` is the name of the input file placed in the `examples` folder.
* ``[output-file.json]`` is the output file name.

Alternatively, you can run the analysis with the input data in one of the files in the ``tests/data`` folder, and assuming your project name is called ``test_project``, job name is called ``job01`` and data is output in the **standard** configuration.

.. code-block:: bash

   $ docker compose run --rm geocaret python heet_cli.py tests/data/dams.csv test_project job01 standard

See :doc:`../ghg_emissions/input_data` and :doc:`running_python_package` to read about input data file specification and about the usage of GeoCARET's command-line interface (CLI) arguments, respectively.

Running demo in the Docker container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also use the Docker image to run the demo as described in full in :doc:`Running the demo <demo>`. To run the demo:

.. code-block:: bash

   ❯ docker compose run --rm reemission reemission run-demo
