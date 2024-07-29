Theoretical Background
======================

.. _G-Res: https://www.hydropower.org/publications/the-ghg-reservoir-tool-g-res-technical-documentation

.. _Contribution of Reservoirs to Global Emissions:

Contribution of Reservoirs to Global Emissions
----------------------------------------------

Reservoirs contribute significantly to the global emissions of carbon dioxide (CO$_2$), methane (CH$_4$), and nitrous oxide (N$_2$O). 
Recent estimates place global reservoir greenhouse gas (GHG) emissions at approximately ∼1.076 PgCO$_{2e}$/year (:cite:t:`Harrison2021`) - a third of the emissions of the whole of the European Union in 2022.
According to :cite:t:`Soued2022`, reservoirs accounted for 5.2% of CH$_4$ and 0.2% of CO$_2$ anthropogenic emissions in 2022, with a climate impact comparable to the aviation sector and about **2% of global man-made greenhouse gas emissions**. 

.. _Global Carbon Cycle:

Global Carbon Cycle
-------------------

The Global Carbon Cycle describes the pathways of greenhouse gas (GHG) fluxes between different parts of the land, water, man-made infrastructure, and the atmosphere. These fluxes determine the equilibrium concentrations of GHGs in the atmosphere, which in turn drive the Earth's climate.

Reservoirs are integral to the Global Carbon Cycle and are interlinked with other components involved in carbon production and sequestration, such as natural landscapes, including moving and standing water bodies, and landscapes transformed by human activity. The latter includes land use changes due to agriculture, farming, urbanization, and more. These interconnections make it challenging to quantify reservoir emissions, as distinguishing between emissions directly attributable to reservoir creation (:cite:t:`Prairie2018`) and the natural emissions of the landscape is intrinsicly difficult. It is also challenging to quantify the portion of emissions elevated by human activity, such as nutrient exports to aquatic systems, specifically linked to reservoir creation versus what would have occurred in unaltered water bodies, albeit over different time scales and distances (see `Displaced Emissions`_).

Understanding the natural and anthropogenic GHG fluxes of reservoirs is essential for appropriate quantification of the Global Carbon Cycle driving the climate change.

.. image:: _static/images/CarbonCycle_Revised_1200px.jpg
   :width: 100 %
   :alt: Global Carbon Cycle
   
**Global Carbon Cycle**, Source: `https://www.usgs.gov/media/images/usgs-carbon-cycle <https://www.usgs.gov/media/images/usgs-carbon-cycle>`_, Created by: Alison Mims

Why Reservoirs Emit Greenhouse Gases
------------------------------------

Reservoirs alter the carbon cycle of the natural landscape, directly leading to net greenhouse gas (GHG) emissions.

Following the impoundment, the river is transformed into a standing pool of water with a longer residence time and greater depth. This allows various biochemical and physical processes to occur within the reservoir, leading to the production of greenhouse gases.

When a landscape is flooded, the natural emissions and pre-impoundment state are replaced by a body of water containing organic matter from the flooded soil and vegetation. This organic matter decomposes more rapidly in the aquatic environment, increasing the GHG stock in the water column, which is then released into the atmosphere through various physicochemical mechanisms - see `Reservoir Emission Pathways`_.

Reservoirs often stratify into layers (strata) with different physicochemical properties, such as oxygen or nutrient concentrations and temperature levels. This stratification creates conditions that support various biochemical processes, creating an organic carbon cycle within the reservoir. The bottom layers lack molecular oxygen, leading to anoxic and anaerobic conditions that support methanogenesis, the production of methane (CH$_4$). The organic matter fueling methane production in these deeper layers comes from flooded soils and sediments entering the reservoir via rivers and surface runoff.

Methane produced in the reservoir's deeper regions may remain at lower depths or move to the upper layers as bubbles, diffuse, or be transported by plants, eventually reaching the atmosphere. 
Some CH$_4$ is converted to carbon dioxide (CO$_2$) by methane-oxidizing bacteria (MOBs), which can be aerobic (oxygen-driven) or anaerobic (nitrite- and nitrate-driven). 
Methane pools in the lower reservoir regions can persist for long periods or be released to the atmosphere during water drawdowns or when water from deeper areas is released downstream via outlets positioned at the dam's lower parts.
In the latter case, CH$_4$ emissions occur in the river sections downstream of the reservoir through a process called *degassing*.

.. _Displaced Emissions:

Carbon dioxide (CO$_2$) emissions result from the mineralization of organic matter, primarily soluble but also particulate, entering the reservoir from the upstream catchment. This process mainly occurs via diffusion and macrophytes. While the reservoir provides conditions that support greater mineralization by offering longer residence times, a large portion of this organic matter would have been mineralized in natural aquatic systems regardless of the reservoir's presence. In this case, reservoirs alter the carbon cycle by **displacing emissions** from downstream to upstream in the catchment rather than significantly increasing overall emissions.

The alterations in the carbon cycle due to impoundment lead to new (net) GHG emissions, which vary over time. These emissions are most intense in the first years following reservoir construction and decrease as the reservoir ages, with the most significant reduction occurring in the initial 5-10 years after impoundment.

The total emissions from a reservoir over its lifetime and their evolution depend on various factors, including reservoir characteristics, catchment properties, and climatic conditions, which influence the emissions of individual gases (CH$_4$, CO$_2$ and N$_2$O) and their specific emission pathways.

The transformation of landscape from *pre-impoundment* to *post-impoundment* and the major emission routes for CO$_2$ and CH$_4$ in the reservoir are visualised in the figure below.

.. image:: _static/images/reservoir_emissions.png
   :width: 100 %
   :alt: Net Emissions from Reservoirs
  
**Landscape transformation from a river to a reservoir.** Source: :cite:t:`Prairie2018`.

The G-Res Emissions Model
-------------------------

For the estimation of CO$_2$ and CH$_4$ emissions we use the G-Res_ model (:cite:t:`Prairie2017b,Prairie2021`).
G-Res_ is based on a number of statistical regression models, each estimating emission flux via a single emission pathway - see `Reservoir Emission Pathways`_.
The regressions use information about the landscape, the geomorphological properties of the reservoir and the averaged climatic conditions in the catchment.
This data can be sourced from publicly available geospatial data hence avoiding the need for field measurements at individual reservoir locations.

The unique characteristic of G-Res_ lies in its ability to discern the true *net* GHG footprint resulting from the conversion of a
river to a reservoir. For this purpose it calculates the following GHG emission mass balances:

  * Pre-impoundment GHG footprint of the landscape, i.e. the catchment, the reservoir area and the impounded river area.
  * Reservoir emissions rates for each gas and each of its emission pathways as functions of various environmental settings of the reservoir, i.e. climatic, geographic, edaphic and hydrologic.
  * The evolution of emission fluxes for each gas and each emission pathway over the lifetime of the reservoir.
  * Displaced emissions, i.e. emissions that would have occurred in other parts of the aquatic environment irrespective of the presence of the reservoir.
  * Emissions in the reservoir related to human activity in the catchment, i.e. resulting from increased emissions due to additional nutrients and organic matter entering the reservoir from athropogenic sources, aka. **Unrelated anthropogenic sources**.

The N$_2$O Emissions Model
--------------------------

.. note::
   Modelling of nitrous oxide emissions is at an experimental stage.
   We are working on establishing the state-of-the-art models and verifying that all the calculations are correct.
   We shall update this section as soon as we are confident that the predictions from our software align with the values reported in the literature.
   
.. _Reservoir Emission Pathways:

Reservoir Emission Pathways for CO$_2$ and CH$_4$
-------------------------------------------------

The figure below, included in the IPCC 2007 report, illustrates the pathways of greenhouse gas (GHG) emissions in reservoirs. 
It shows that organic matter (OM) can be either fluvial, consisting of dissolved organic matter (DOM) or particulate organic matter (POM), or originate from flooded vegetation and soils. 
The figure distinguishes five emission pathways: (1) methane (CH$_4$) bubbling, (2) carbon dioxide (CO$_2$) and CH$_4$ diffusion within the reservoir, (3) CO$_2$ and CH$_4$ flux through macrophytes, (4) CO$_2$ and CH$_4$ degassing downstream of the reservoir, and (5) CO$_2$ and CH$_4$ diffusion downstream of the reservoir. 
It highlights methanogenesis in the benthic zone and aerobic CH$_4$ oxidation in the upper aerobic layers of the reservoir.
Macrophytes facilitate CO$_2$ and CH$_4$ fluxes into the atmosphere and contribute to carbon and nutrient sources for methanogenesis when they die and decompose.

Other pathways and processes not included in this illustration also contribute to GHG emissions. 
For instance, emissions can result from the drying and rewetting of sediments in the littoral areas of the reservoir due to water abstraction, as discussed by :cite:t:`Keller2021,MARCE2019240,Kosten2018`.

However, many known emission pathways are still not well understood due to the complexities of the fundamental processes governing them, the availability of measurements for model calibration and validation and uncertainties and difficulties associated with measuring emissions and various environmental parameters in water bodies. 
Given the current limitations in scientific understanding and data availability, the G-Res_ model is currently the most comprehensive emissions model. 
It describes the following emission pathways: (1) diffusive CO$_2$ emissions, (2) diffusive CH$_4$ emissions, (3) CH$_4$ emissions due to ebullition (bubbling), and (4) CH$_4$ emissions due to degassing.

.. image:: _static/images/Pathways-of-GHG-Emissions-from-a-Reservoir-IPCC-2007_W640.jpg
   :width: 100 %
   :alt: Pathways of GHG Emissions from a Reservoir

**Pathways of GHG Emissions from a Reservoir.** Source: IPCC 2007.

Why is it important to estimate GHG emissions from reservoirs?
--------------------------------------------------------------

As highlighted in the sections `Contribution of Reservoirs to Global Emissions`_ and `Global Carbon Cycle`_ reservoir emissions are significant on a global scale and, therefore, have a non-negligible impact on the Earth's climate. 
Accurate quantification of these emissions worldwide is necessary to estimate the total anthropogenic GHG emissions, which are used to determine the current climate and its future projections.

On a national level, estimating reservoir emissions can support climate planning instruments, such as Nationally Determined Contributions (NDCs), and in the assessment of GHG budgets for entire countries, regions, and companies. 
For example, the `Greenhouse Gas Reporting Program (GGRP) <https://www.epa.gov/ghgreporting>`_ of the United States Environmental Protection Agency is one such initiative. 
These and other efforts can aid in policy and technological planning related to climate adaptation.

Finally, estimating the GHG emissions of planned reservoirs is crucial for designing carbon-efficient reservoir infrastructure investments. 
Studies, such as :cite:t:`Hansen2022`, have shown that reservoirs can emit vastly different amounts of GHGs depending on their location. 
Therefore, to avoid construction of polluting reservoirs, reservoir emissions should be considered during strategic reservoir planning alongside other economic, technical, and environmental objectives.

Literature
----------

.. bibliography:: _static/references.bib
