Configuration
=============

.. _GeoCARET: https://github.com/Reservoir-Research/geocaret

**RE-Emission** relies on a number of external configuration text files located in ``reemission/config`` and ``reemission/parameters``. 
These configuration files allow modification of the package's parameters and options, such as types of emission models (when alternative mdels are available), model parameters such as e.g. emission coefficients, nitrogen and phosphorus exports from land, emission coefficients of different land cover types, etc.
The configuration files are provided in three different formats: ``ini``, ``yaml`` and ``toml`` depending on the usage of the configuration file. 

.. note ::
    We are working on adding a functionality that will allow the user to modify configuration files programmatically.
    For now, in order to modify the config file, the users need to find the file in the installation directory and modify it manually using a text editor.
    
Configuration files uses
~~~~~~~~~~~~~~~~~~~~~~~~

1. Application-wide configuration in ``app_config.yaml``.
2. Emission modelling configurations in ``config.ini``.
3. Output and Reporting configurations in ``inputs.yaml``, ``outputs.yaml``, ``parameters.yaml`` and ``internal_vars.yaml``.
4. Pre-impoundment emissions and landscape Nitrogen and Phosphorus exports in various ``.yaml`` files in ``reemission/parameters``.
5. Phosphorus removal efficiencies of different wastewater treatment technologies in ``phosphorus_loads.yaml``.
6. Visualisation and integration with GeoCARET_ in ``geocaret.toml`` and ``visualisation.toml``.

Listings
~~~~~~~~

1. Application-wide configuration
---------------------------------

Application-wide configuration is used set the parameters of the logger, the $\LaTeX$ compiler, and the locations of files used in the demo. 

.. literalinclude:: ../src/reemission/config/app_config.yaml
  :caption: ``app_config.yaml``
  :language: yaml
  :linenos:

2. Emission modelling parameters
--------------------------------

This configuration ``.ini`` file is used to set the values of the emission model parametsr for the ``[CARBON_DIOXIDE]``, ``[METHANE]`` and ``[NITROUS OXIDE]`` emissions. The ``[CALCULATIONS]`` section sets different simulation/calculation options.

.. literalinclude:: ../src/reemission/config/config.ini
  :caption: ``config.ini``
  :language: ini
  :linenos:
  
.. _output-and-reporting-config:
 
3. Output and Reporting configuration
-------------------------------------

These configuration files set the presentation options for model inputs, outputs, parameters and intermediate variables. These settings are used in the :doc:`Presenter <api/reemission.presenter>` class during creation of calculation outputs files - see :doc:`Reporting of Outputs <reporting>` for examples.

Information for each variable follows a similar structure, see example of the output variable ``co2_diffusion`` representing calculated $CO_2$ emission via diffusion. :

.. code:: yaml

   # CO2 emission through diffusion
   co2_diffusion:
    include: True
    name: "CO2 diffusion flux"
    gas_name: "CO2"
    name_latex: "CO$_2$ diffusion flux"
    unit: "gCO2eq m-2 yr-1"
    unit_latex: "gCO$_{2,eq}$ m$^{-2}$ yr$^{-1}$"
    long_description: "Total CO2 emissions from a reservoir integrated over lifetime"
    hint: ""

-  ``include``: (boolean): If the variable is to be included in the
   output files for reporting.
-  ``name``: (string): Name of the variable
-  ``gas_name``: Name of the gas which the (output) variable belongs to.
-  ``name_latex``: (string): Name of the variable in $\LaTeX$ format.
-  ``unit``: (string): Unit in text format.
-  ``unit_latex``: (string): Unit in $\LaTeX$ format.
-  ``long_description``: (string): Description of the variable.
-  ``hint``: (string): Additional information about the variable.

Not all categories of variables contain the same information, e.g. **inputs** do not have names in $\LaTeX$ format but the differences are small.

|

.. tabs::

   .. tab:: Inputs

      .. literalinclude:: ../src/reemission/config/inputs.yaml
        :caption: ``inputs.yaml``
        :language: yaml
        :linenos:

   .. tab:: Outputs

      .. literalinclude:: ../src/reemission/config/outputs.yaml
        :caption: ``outputs.yaml``
        :language: yaml
        :linenos:
        
   .. tab:: Intermediate Variables
   
      .. literalinclude:: ../src/reemission/config/internal_vars.yaml
        :caption: ``internal_vars.yaml``
        :language: yaml
        :linenos:

   .. tab:: Parameters

      .. literalinclude:: ../src/reemission/config/parameters.yaml
        :caption: ``parameters.yaml``
        :language: yaml
        :linenos:

4a. Pre-impoundment emissions
-----------------------------

Pre-impoundment emissions are emissions of landscape prior to reservoir creation. Pre-impoundment emissions depend on 

.. tabs::

   .. tab:: Carbon Dioxide

      .. literalinclude:: ../src/reemission/parameters/Carbon_Dioxide/pre-impoundment.yaml
        :caption: ``Carbon_Dioxide/pre-impoundment.yaml``
        :language: yaml
        :linenos:

   .. tab:: Methane

      .. literalinclude:: ../src/reemission/parameters/Methane/pre-impoundment.yaml
        :caption: ``Methane/pre-impoundment.yaml``
        :language: yaml
        :linenos:

4b. Landscape N and P exports
-----------------------------

.. tabs::

   .. tab:: Mac-Dowell P and N exports

      .. tabs::

         .. tab:: Total Nitrogen (TN)

            .. literalinclude:: ../src/reemission/parameters/McDowell/landscape_TN_export.yaml
              :caption: ``McDowell/landscape_TN_export.yaml``
              :language: yaml
              :linenos:

         .. tab:: Total Phosphorus (P)

            .. literalinclude:: ../src/reemission/parameters/McDowell/landscape_TP_export.yaml
              :caption: ``McDowell/landscape_TP_export.yaml``
              :language: yaml
              :linenos:

   .. tab:: G-Res P Exports

      .. literalinclude:: ../src/reemission/parameters/phosphorus_exports.yaml
        :caption: ``phosphorus_exports.yaml``
        :language: yaml
        :linenos:

5. Phosphorus removal efficiencies
----------------------------------

.. literalinclude:: ../src/reemission/parameters/phosphorus_loads.yaml
  :caption: ``phopshporus_loads.yaml``
  :language: yaml
  :linenos:
  
4. Visualisation and integration with GeoCARET_
-----------------------------------------------

.. note::
   These settings are only used for the purpose of running a demonstration of how RE-Emission can be run in conjuction with GeoCARET - see :doc:`Demo <demo>`.
   They are not needed for any other functionality of the software.

.. tabs::

   .. tab:: GeoCARET_ integration

      .. literalinclude:: ../src/reemission/config/geocaret.toml
        :caption: ``geocaret.toml``
        :language: toml
        :linenos:

   .. tab:: Visualisation (Maps)

      .. literalinclude:: ../src/reemission/config/visualisation.toml
        :caption: ``visualisation.toml``
        :language: toml
        :linenos:
