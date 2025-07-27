Tutorials
=========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb

1. `Manual Step-By-Step Calculations <01-Step-By-Step-Manual-Calculations.ipynb>`_ 
----------------------------------------------------------------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb
   
**This Notebook demonstrates how to:**
 * Construct input data structures manually for a hypotethical reservoir.
 * Instantiate Catchment and Reservoir objects.
 * Calculate $CO_2$, $CH_4$ and $N_2O$ emission factors
 * Calculate $CO_2$, $CH_4$ and $N_2O$ emission profiles

2. `Automatic Calculation Of Batches of Reservoirs <02-Automatic-Calculation-Of-Emissions-For-Batches-Of-Reservoirs.ipynb>`_
----------------------------------------------------------------------------------------------------------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb
   
**This Notebook demonstrates how to:**
 * Read input data in JSON format and output configuration ``YAML`` file.
 * Instantiate the emission model object from input data output configuration file.
 * Calculate emissions.
 * Display model ouptputs.

3. `Presentation of Results in JSON Format <03-Saving-Results-To-JSON.ipynb>`_
------------------------------------------------------------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb
   
**This Notebook demonstrates how to:**
 * Read input data and output configuration ``YAML`` file and instantiate the emission model
 * Add JSON model presenter, calculate emissions and save results to ``JSON`` file.
 * Read and display the results saved in ``JSON`` format.

4. `Presentation of Results in PDF Format (via LaTeX) <04-Saving-Results-To-LaTeX.ipynb>`_
------------------------------------------------------------------------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb
   
**This Notebook demonstrates how to:**
 * Read input data and output configuration ``YAML`` file and instantiate the emission model
 * Add $\LaTeX$ model presenter, calculate emissions and save results to $\LaTeX$ and PDF files.
 * Open the generated ``PDF`` document.

5. `Modification of configuration parameters <05-Modifying-Configuration-Parameters.ipynb>`_
--------------------------------------------------------------------------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb

  **This Notebook demonstrates how to:**
  * Read a custom configuration file from a user's file-system to update model parameters
  * Update selected config variables after loading the configuration file
  * Run step-by-step manual calculations with different model parameterizations
  * Run calculations for batches of reservoirs with custom configurations

6. `Sensitivity Analysis and Probabilistic Estimation of Reservoir Emissions <06-Parametric-Uncertainty.ipynb>`_
----------------------------------------------------------------------------------------------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tomjanus/reemission/blob/master/docs/notebooks/index.ipynb

  **This Notebook demonstrates how to:**
  * Run parametric uncertainty analysis with SALib and the Sobol method
  * Visualize the parametric sensitivity / uncertainty on various plots
  * Compute the sensitivities across many scenarios (e.g. reservoirs)
  * Present emission predictions as probability density plots