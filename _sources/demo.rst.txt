Demo
====

.. _GeoCARET: https://github.com/Reservoir-Research/geocaret

We have created a short example to demonstrate how **RE-EMISSION** works in combintion with the *upstream* reservoir and catchment delineation tool **GeoCARET**. This example showcases RE-EMISSION's integration capabilities with GeoCARET_ on a small set of reservoirs taken from a larger real-life case study on estimating **GHG** emissions from existing and planned hydroelectric
reservoirs in Myanmar. The demo runs in the following steps: 

1. Merging multiple tabular data files from several batches of reservoir delineations in GeoCARET_ into a single ``CSV`` file.
2. Merging shape files of individual reservoirs and catchments into combined shape files of reservoirs and catchments, respectively.
3. Converting the merged tabular data into a ``JSON`` input file.
4. Calculating GHG emissions with RE-EMISSION using the input data in the ``JSON`` input file.
5. Pasting the GeoCARET_ tabular input data and RE-EMISSION's output data into shape files of reservoirs and dams
6. Presenting the updated shape files using an interactive map with `Folium <https://python-visualization.github.io/folium/latest/>`_.

You can run the demo from the command-line following installation of the RE-EMISSION package - see :doc:`Installation Guidelines <install>`

.. code:: sh

   reemission run-demo [working-directory]

``working-directory`` is the folder in which all the inputs and outpus for the demo will be stored. If the folder structure does not exists it will be created. It can be any directory/folder on the system, existing or not. If the folder does not exist, it will be automatically created. The demo relies on several input datasets produced by ``GeoCARET``. If this data is not already presetn in your ``working-directory`` it will be automatically downloaded using the ``gdown`` package. 

.. attention::
    This step requires working internet connection.

You can also type:

.. code:: bash

   reemission run-demo --help
   
to get th *Usage*.

If all goes well the demo should run as follows:

.. hint::
    Click on the image to view it in full size.

|demo-22-05-24-compressed|

.. |demo-22-05-24-compressed| image:: https://github.com/tomjanus/reemission/assets/8837107/b101e9d0-ac60-4f21-bbeb-a8a8ae85522b
  :width: 2400
  :alt: Animated GIF of a demo run
  
An interactive map with results will be automatically generated and displayed in your browser. You can view the map with results created in the demo in the :doc:`Visualisation<visualisation>` section.
