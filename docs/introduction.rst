About the Library
=================

.. _RE-Emission: https://github.com/tomjanus/reemission
.. _GeoCARET: https://github.com/Reservoir-Research/geocaret
.. _G-Res: https://www.hydropower.org/publications/the-ghg-reservoir-tool-g-res-technical-documentation

.. image:: _static/images/logo-banner-bw.png
   :align: center
   :width: 100 %

RE-Emission_ is a free open-source tool for estimating, reporting and visualising greenhouse gas (GHG) emissions from reservoirs. It intends to be extendable and allow calculating emissions from multiple reservoirs at once. Therefore, it is suited for reservoir planning and assessments on regional and national scales. RE-Emission_ uses the G-Res_ [Praire2021]_ emission model for CO$_2$ and CH$_4$ emission calculations and additionally supports estimating N$_2$O emissions.

RE-Emission_ calculates GHG emissions based on input data quantifying the geomorphological properties of the landscape flooded by the reservoir and of the reservoir catchment, as well as various climatic variables driving the physical and biochemical emission pathways. These climatic and land variables can be sourced from different datasets, e.g. surveys, databases of reservoirs, etc., measured on-site, or obtained using GIS processing tools such as ``ArcGIS`` or ``qGIS`` using different geospatial datasets as the source of information about the local land and atmospheric conditions. 

Sourcing input data for GHG emission calculations can be time-consuming and requires expert knowledge, especially if reservoir contours and reservoir catchments are not known a priori. RE-Emission_ integrates with GeoCARET_ - a command line Python tool for delineating and analysing catchments and reservoirs. GeoCARET_ automates the process of retrieving input data for the emission models. Having RE-Emission_ read and parse the outputs of GeoCARET_ and convert them into inputs allows us to estimate emissions of multiple reservoirs with minimal number of manual steps.

Features
--------

* Calculates CO$_2$, CH$_4$ and N$_2$O emissions for a single reservoir as well as for batches of reservoirs.
* Two reservoir Phosphorus mass balance calculation methods: the G-Res_ method and the `McDowell` method.
* Two N$_2$O calculation methods.
* Model parameters, and presentation of outputs are fully configurable using ``YAML`` configuration files.
* Inputs can be either constructed in Python using the ``Input`` class or read from ```JSON`` files.
* Outputs in tabular form can be presented in ``JSON``, ``LaTeX`` and ``PDF`` formats and can be configured by changing settings in the `outputs.yaml` configuration file.
* Integrates with the upstream catchment and reservoir delineation package GeoCARET_.
* Combines tabular and GIS inputs from catchment delineation with gas emission outputs and visualizes the combined data in interactive maps using ``Leaflet``.

References
----------
.. [Praire2021] Yves T. Prairie and Sara Mercier-Blais and John A. Harrison and Cynthia Soued and Paul del Giorgio and Atle Harby and Jukka Alm and Vincent Chanudet and Roy Nahas, 2021, `A new modelling framework to assess biogenic GHG emissions from reservoirs: The G-res tool`, Environmental Modelling & Software 143 (pp: 105-117), https://doi.org/10.1016/j.envsoft.2021.105117
