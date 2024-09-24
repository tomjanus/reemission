Running the ReEmission Docker Container
=======================================

Run a ReEmission analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

We can use Docker image of ReEmission to run analysis using ReEmission's command-line tools. First, copy your input data file to the ``examples`` sub-folder and then start ReEmission by typing the following:

.. code-block:: bash

   ❯ docker compose run --rm reemission reemission calculate ./examples/[input-file.json] -o ./outputs/[output-file.json] 

where:

* ``[input-file.json]`` is the name of the input file placed in the `examples` folder.
* ``[output-file.json]`` is the output file name.

You can run a test analysis with the input data in one of the files in the ``examples`` folder.

.. code-block:: bash

   $ docker compose run --rm reemission reemission calculate ./examples/test_input.json -o ./outputs/test_output.json

Running demo in the Docker container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also use the Docker image to run the demo as described in full in :doc:`Running the demo <demo>`. To run the demo:

.. code-block:: bash

   ❯ docker compose run --rm reemission reemission run-demo
