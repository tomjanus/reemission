Configuration
-------------

Coefficients of the regressions constituting the model as well as
parameters of different categories of soil and land use are stored in a
number of **yaml** files in ``parameters/emissions/``.

Configuration of inputs
~~~~~~~~~~~~~~~~~~~~~~~

Information about the names and the units of the model inputs is stored
and can be configured in ``config/emissions/inputs.yaml`` e.g. for
monthly temperatures which are represented in variable
``monthly_temps``:

.. code:: yaml

   monthly_temps:
     include: True
     name: "Monthly Temperatures"
     long_description: ""
     unit: "deg C"
     unit_latex: "$^o$C"

-  ``include``: (boolean): If the variable is to be included in the
   output files for reporting.
-  ``name``: (string): Name of the variable
-  ``long_description``: (string): Description of the variable
-  ``unit``: (string): Unit in text format
-  ``unit_latex``: (string): Unit in LaTeX format

Finally, a global flag ``print_long_descriptions`` controls whether long
descriptions are included alongside the included input variables in the
output files.

Configuration of outputs
~~~~~~~~~~~~~~~~~~~~~~~~

Similarly to inputs, definitions and units of model outputs and whether
they are to be output in the output files, are stored in
``config/emissions/outputs.yaml``, e.g. for pre-impoundment CO2
emissions defined in variable ``co2_preimp``:

.. code:: yaml

   co2_preimp:
     include: True
     name: "Preimpoundment CO2 emissions"
     gas_name: "CO2"
     name_latex: "Preimpoundment CO$_2$ emissions"
     unit: "gCO2eq m-2 yr-1"
     unit_latex: "gCO$_2$ m$^{-2}$ yr$^{-1}$"
     long_description: "CO2 emission in the area covered by the reservoir prior to impoundment"
     hint: "Negative values denote C sink (atmosphere to land flux)"

-  ``include``: (boolean): If the variable is to be included in the
   output files for reporting.
-  ``name``: (string): Name of the variable
-  ``gas_name``: (string): Name of the gas the variable is related to
-  ``name_latex``: (string): variable name in LaTeX format
-  ``unit``: (string): Unit in text format
-  ``unit_latex``: (string): Unit in LaTeX format
-  ``long_description``: (string): Description of the variable
-  ``hint``: (string): Further information about the variable

Configuration of global parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Information about global parameters such as e.g. Global Warming
Potentials ``gwp100`` is stored in ``config/emissions/parameters.yaml``

.. code:: yaml

   gwp100:
     include: True
     name: "Global Warming Potential for a 100-year timescale"
     name_latex: "Global Warming Potential for a 100-year timescale"
     unit: "-"
     unit_latex: "-"
     long_description: ""

Model coefficients
~~~~~~~~~~~~~~~~~~

Values of model coefficients, i.e. regressions used to estimate
different gas emissions are stored in ``config/emissions/config.ini``
file. E.g. coefficients for CO2 emission calculations are listed below.

.. code:: ini

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

In addition, various coefficient tables and parameters required to
calculate various emission components are stored in multiple YAML files
in ``parameters/emissions/``.
