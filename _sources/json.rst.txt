JSON file format description
============================

.. _GeoCARET: https://github.com/Reservoir-Research/geocaret

**RE-Emission**'s native input and output data format is ``JSON``. It can read from and save to other data formats such as e.g. ``.csv`` and ``.xlsx`` (Excel) files, however internally the data is represented in ``JSON`` as described below. The ``JSON`` format was chosen to facilitate seamless data sharing between multiple software packages, e.g. GeoCARET_ and **RE-Emission** and for communicating data over the Web.

Input Format
------------

.. code-block:: json
   :caption: Example input ``JSON`` file for a single reservoir
    
    {
        "Shweli 1": {
            "id": 1,
            "type": "hydroelectric",
            "monthly_temps": [
                13.9, 16.0, 19.3, 22.8, 24.2, 24.5, 
                24.2, 24.3, 23.9, 22.1, 18.5, 14.8
            ],
            "year_vector": [1, 5, 10, 20, 30, 40, 50, 65, 80, 100],
            "gasses": ["co2", "ch4", "n2o"],
            "catchment": {
                "runoff": 1115.0,
                "area": 12582.613,
                "riv_length": 0.0,
                "population": 1587524.0,
                "area_fractions": [
                    0.0, 0.0, 0.003, 0.002, 0.001, 0.146, 0.391, 0.457, 0.0
                ],
                "slope": 23.0,
                "precip": 1498.0,
                "etransp": 1123.0,
                "soil_wetness": 144.0,
                "mean_olsen": 5.85,
                "biogenic_factors": {
                    "biome": "tropical moist broadleaf",
                    "climate": "temperate",
                    "soil_type": "mineral",
                    "treatment_factor": "primary (mechanical)"
                }
            },
            "reservoir": {
                "volume": 7238166.0,
                "area": 1.604,
                "max_depth": 22.0,
                "mean_depth": 4.5,
                "area_fractions": [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.15, 0.4, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0
                ],
                "soil_carbon": 6.281,
                "mean_radiance": 4.66,
                "mean_radiance_may_sept": 4.328,
                "mean_radiance_nov_mar": 4.852,
                "mean_monthly_windspeed": 1.08,
                "water_intake_depth": null
            }
        }
    }
    
For each reservoir (in this example we only show one reservoir called `Shweli 1`), the input data is divided into three groups: `generic inputs`, i.e. ``id``, ``type``, ``monthly_temps``, ``year_vector`` and ``gasses`` followed by ``reservoir``- and ``catchment``-wide inputs. Definitions and units of most input variables are shown in the example ``JSON`` output (see `Output Format`_). ``id`` is a unique ID of a reservoir. It can be a String or Integer. ``type`` is a categorical variable denoting the type of the reservoir, e.g. `hydroelectric`, `irrigation`, etc. ``monthly_temps`` is a vector of monthly average air temperatures in degrees Celcius at the reservoir location. ``year_vector`` is a list of years elapsed from the year of construction for which emissions will be calculated. It is used for constructing emission profiles, i.e. emission vs. elapsed time from impoundment. ``gasses`` is a vector of greenhouse gasses which will be calculated and summed up to calculate total gas emissions. We support three gas emissions: **"co2"**, **"ch4"**, and **"n2o"**.

Tha permitted values of the categorical variables, i.e. ``type``, ``biome``, ``climate``, ``treatment_factor`` can be found in :doc:`Reemission Constants <api/reemission.constants>`.

The input files can be created manually, converted from Excel/CSV files using RE-Emission and/or obtained directly from GeoCARET_ - An open-source **Geo** spatial **CA** tchment and **RE** servoir analysis **T** ool.

.. _json-output-format:

Output Format
-------------

.. note::

    **RE-Emission** can produce outputs in several file formats other than JSON using the :doc:`Presenter <api/reemission.presenter>` class.
    Currently supported formats (exluding ``JSON ``) are:

    * ``Excel`` / ``CSV``
    * ``LaTeX`` / ``PDF``

    More information about other file formats can be found in :doc:`Reporting of Outputs <reporting>`.

.. attention::
    Saving outputs to PDF requires a working $\LaTeX$ installation - see the :doc:`Installation Guidelines <install>`
    

The content and the presentation of variables and parameters in the output ``JSON`` file is customizable - please check :doc:`configuration` section :ref:`output-and-reporting-config`.

The output ``JSON`` file is divided into three sections: **inputs**, **outputs** and **intern_vars**. **inputs** and **outputs** are the model inputs and outputs, respectively, marked for presentation. **intern_vars** are the internal variables, i.e. intermediary variables or additional auxiliary variables which are not directly output by the model(s) but are marked for saving, e.g. for cross-referencing with other results, reporting, as source of data for various numerical analyses, etc. Each reported variable/parameter section follows the same structure:

-  ``name``: (string): Name of the variable
-  ``unit``: (string): Unit in text format
-  ``value``: (Any): Value of the variable/parameter - can be a number, string, or list of numbers or strings

.. code-block:: JSON
   :caption: Example output ``JSON`` file for a single reservoir
   
    {
        "Shweli 1": {
            "inputs": {
                "coordinates": {
                    "name": "Reservoir coordinates (lat/lon)",
                    "unit": "deg",
                    "value": [
                        0.0,
                        0.0
                    ]
                },
                "id": {
                    "name": "Reservoir ID",
                    "unit": "",
                    "value": 1
                },
                "type": {
                    "name": "Reservoir type",
                    "unit": "",
                    "value": "hydroelectric"
                },
                "monthly_temps": {
                    "name": "Monthly Temperatures",
                    "unit": "deg C",
                    "value": [
                        13.9, 16.0, 19.3, 22.8, 24.2, 24.5,
                        24.2, 24.3, 23.9, 22.1, 18.5, 14.8
                    ]
                },
                "year_profile": {
                    "name": "Year vector for emission profiles",
                    "unit": "yr",
                    "value": [1,5,10,20,30,40,50,65,80,100]
                },
                "gasses": {
                    "name": "Calculated gas emissions",
                    "unit": "-",
                    "value": ["co2", "ch4", "n2o"]
                },
                "biogenic_factors": {
                    "name": "Biogenic factors",
                    "biome": {
                        "name": "Biome",
                        "unit": "",
                        "value": "tropical moist broadleaf"
                    },
                    "climate": {
                        "name": "Climate",
                        "unit": "",
                        "value": "temperate"
                    },
                    "soil_type": {
                        "name": "Soil Type",
                        "unit": "",
                        "value": "mineral"
                    },
                    "treatment_factor": {
                        "name": "Treatment Factor",
                        "unit": "",
                        "value": "primary (mechanical)"
                    }
                },
                "catchment_inputs": {
                    "name": "Inputs for catchment-level process calculations",
                    "runoff": {
                        "name": "Annual runoff",
                        "unit": "mm/year",
                        "value": 1115.0
                    },
                    "area": {
                        "name": "Catchment area",
                        "unit": "km2",
                        "value": 12582.613
                    },
                    "riv_length": {
                        "name": "Length of inundated river",
                        "unit": "km",
                        "value": 0.0
                    },
                    "population": {
                        "name": "Population",
                        "unit": "capita",
                        "value": 1587524.0
                    },
                    "area_fractions": {
                        "name": "Area fractions",
                        "unit": "-",
                        "value": "0.0, 0.0, 0.003, 0.002, 0.001, 0.146, 0.391, 0.457, 0.0"
                    },
                    "slope": {
                        "name": "Mean catchment slope",
                        "unit": "%",
                        "value": 23.0
                    },
                    "precip": {
                        "name": "Mean annual precipitation",
                        "unit": "mm/year",
                        "value": 1498.0
                    },
                    "etransp": {
                        "name": "Mean annual evapotranspiration",
                        "unit": "mm/year",
                        "value": 1123.0
                    },
                    "soil_wetness": {
                        "name": "Soil wetness",
                        "unit": "mm over profile",
                        "value": 144.0
                    },
                    "mean_olsen": {
                        "name": "Soil Olsen P content",
                        "unit": "kgP/ha",
                        "value": 5.85
                    }
                },
                "reservoir_inputs": {
                    "name": "Inputs for reservoir-level process calculations",
                    "volume": {
                        "name": "Reservoir volume",
                        "unit": "m3",
                        "value": 7238166.0
                    },
                    "area": {
                        "name": "Reservoir area",
                        "unit": "km2",
                        "value": 1.604
                    },
                    "max_depth": {
                        "name": "Maximum reservoir depth",
                        "unit": "m",
                        "value": 22.0
                    },
                    "mean_depth": {
                        "name": "Mean reservoir depth",
                        "unit": "m",
                        "value": 4.5
                    },
                    "area_fractions": {
                        "name": "Inundated area fractions",
                        "unit": "-",
                        "value": "0.0, 0.0, 0.0, 0.0, 0.0, 0.45, 0.15, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
                    },
                    "soil_carbon": {
                        "name": "Soil carbon in inundated area",
                        "unit": "kgC/m2",
                        "value": 6.281
                    },
                    "mean_radiance": {
                        "name": "Mean monthly horizontal radiance",
                        "unit": "kWh/m2/d",
                        "value": 4.66
                    },
                    "mean_radiance_may_sept": {
                        "name": "Mean monthly horizontal radiance: May - Sept",
                        "unit": "kWh/m2/d",
                        "value": 4.328
                    },
                    "mean_radiance_nov_mar": {
                        "name": "Mean monthly horizontal radiance: Nov - Mar",
                        "unit": "kWh/m2/d",
                        "value": 4.852
                    },
                    "mean_monthly_windspeed": {
                        "name": "Mean monthly wind speed",
                        "unit": "m/s",
                        "value": 1.08
                    },
                    "water_intake_depth": {
                        "name": "Water intake depth below surface",
                        "unit": "m",
                        "value": null
                    }
                }
            },
            "outputs": {
                "co2_diffusion": {
                    "name": "CO2 diffusion flux",
                    "gas_name": "CO2",
                    "unit": "gCO2eq m-2 yr-1",
                    "long_description": "Total CO2 emissions from a reservoir integrated over lifetime",
                    "value": 572.8396
                },
                "co2_diffusion_nonanthro": {
                    "name": "Nonanthropogenic CO2 diffusion flux",
                    "gas_name": "CO2",
                    "unit": "gCO2eq m-2 yr-1",
                    "long_description": "CO2 diffusion flux taken at (after) 100 years",
                    "value": 393.1274
                },
                "co2_preimp": {
                    "name": "Preimpoundment CO2 emissions",
                    "gas_name": "CO2",
                    "unit": "gCO2eq m-2 yr-1",
                    "long_description": "CO2 emission in the area covered by the reservoir prior to impoundment",
                    "value": -132.0
                },
                "co2_minus_nonanthro": {
                    "name": "CO2 emission minus non-anthropogenic",
                    "gas_name": "CO2",
                    "unit": "gCO2eq m-2 yr-1",
                    "long_description": "CO2 emissions minus non-anthropogenic over a number of years",
                    "value": 179.7121
                },
                "co2_net": {
                    "name": "Net CO2 emission",
                    "gas_name": "CO2",
                    "unit": "gCO2eq m-2 yr-1",
                    "long_description": "Overall integrated emissions for lifetime",
                    "value": 311.7121
                },
                "co2_total_per_year": {
                    "name": "Total CO2 emission per year",
                    "gas_name": "CO2",
                    "unit": "tCO2eq yr-1",
                    "long_description": "Total CO2 emission per year integrated over lifetime",
                    "value": 499.9863
                },
                "co2_total_lifetime": {
                    "name": "Total CO2 emission per lifetime",
                    "gas_name": "CO2",
                    "unit": "tCO2eq",
                    "long_description": "Total CO2 emission integrated over lifetime",
                    "value": 49.9986
                },
                "co2_profile": {
                    "name": "CO2 emission profile",
                    "gas_name": "CO2",
                    "unit": "gCO2eq m-2 yr-1",
                    "long_description": "CO2 emission per year for a defined list of years",
                    "value": [
                        1535.8117,
                        795.3817,
                        579.3641,
                        407.5143,
                        323.7749,
                        270.8014,
                        233.039,
                        192.0538,
                        162.0414,
                        132.0
                    ]
                },
                "ch4_diffusion": {
                    "name": "CH4 emission via diffusion",
                    "gas_name": "CH4",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "CH4 emission via diffusion integrated over a number of years.",
                    "value": 222.1296
                },
                "ch4_ebullition": {
                    "name": "CH4 emission via ebullition",
                    "gas_name": "CH4",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "CH4 emission via ebullition",
                    "value": 321.232
                },
                "ch4_degassing": {
                    "name": "CH4 emission via degassing",
                    "gas_name": "CH4",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "CH4 emission via degassing integrated for a number of years",
                    "value": 3857.2376
                },
                "ch4_preimp": {
                    "name": "Pre-impounment CH4 emission",
                    "gas_name": "CH4",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "Pre-impounment CH4 emission",
                    "value": 0.0
                },
                "ch4_net": {
                    "name": "Net CH4 emission",
                    "gas_name": "CH4",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "Net per area CH4 emission",
                    "value": 4400.5992
                },
                "ch4_total_per_year": {
                    "name": "Total CH4 emission per year",
                    "gas_name": "CH4",
                    "unit": "tCO2eq yr-1",
                    "long_description": "Total CH4 emission per year integrated over lifetime",
                    "value": 7058.5611
                },
                "ch4_total_lifetime": {
                    "name": "Total CH4 emission per lifetime",
                    "gas_name": "CH4",
                    "unit": "ktCO2eq",
                    "long_description": "Total CH4 emission integrated over lifetime",
                    "value": 705.8561
                },
                "ch4_profile": {
                    "name": "CH4 emission profile",
                    "gas_name": "CH4",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "CH4 emission per year for a defined list of years",
                    "value": [
                        13754.635,
                        12109.1572,
                        10332.7868,
                        7542.7723,
                        5530.2776,
                        4078.6237,
                        3031.5159,
                        1981.6111,
                        1338.4165,
                        850.4765
                    ]
                },
                "n2o_methodA": {
                    "name": "N2O emission, method A",
                    "gas_name": "N2O",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "N2O emission, method A",
                    "value": 0.0433
                },
                "n2o_methodB": {
                    "name": "N2O emission, method B",
                    "gas_name": "N2O",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "N2O emission, method B",
                    "value": 0.0481
                },
                "n2o_mean": {
                    "name": "N2O emission, mean value",
                    "gas_name": "N2O",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "N2O emission factor, average of two methods",
                    "value": 0.0457
                },
                "n2o_total_per_year": {
                    "name": "Total N2O emission per year",
                    "gas_name": "N2O",
                    "unit": "tCO2eq yr-1",
                    "long_description": "Total N2O emission per year integrated over lifetime",
                    "value": 0.0694
                },
                "n2o_total_lifetime": {
                    "name": "Total N2O emission per lifetime",
                    "gas_name": "N2O",
                    "unit": "ktCO2eq",
                    "long_description": "Total N2O emission integrated over lifetime",
                    "value": 0.0069
                },
                "n2o_profile": {
                    "name": "N2O emission profile",
                    "gas_name": "N2O",
                    "unit": "g CO2eq m-2 yr-1",
                    "long_description": "N2O emission per year for a defined list of years",
                    "value": [
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433,
                        0.0433
                    ]
                }
            },
            "intern_vars": {
                "inflow_p_conc": {
                    "name": "Influent total P concentration",
                    "unit": "micrograms / L",
                    "long_description": "Median influent total phosphorus concentration in micrograms/L entering the reservoir with runoff",
                    "value": 88.8024
                },
                "retention_coeff": {
                    "name": "Retention coefficient",
                    "unit": "-",
                    "long_description": "",
                    "value": 0.0004
                },
                "trophic_status": {
                    "name": "Trophic status of the reservoir",
                    "unit": "-",
                    "long_description": "",
                    "value": "eutrophic"
                },
                "inflow_n_conc": {
                    "name": "Influent total N concentration",
                    "unit": "micrograms / L",
                    "long_description": "Median influent total nitrogen concentration in micrograms/L entering the reservoir with runoff",
                    "value": 5.4369
                },
                "reservoir_tn": {
                    "name": "Reservoir TN concentration",
                    "unit": "micrograms / L",
                    "long_description": "",
                    "value": 5.4344
                },
                "reservoir_tp": {
                    "name": "Reservoir TP concentration",
                    "unit": "micrograms / L",
                    "long_description": "",
                    "value": 88.7758
                },
                "littoral_area_frac": {
                    "name": "Percentage of reservoir's surface area that is littoral",
                    "unit": "%",
                    "long_description": "",
                    "value": 43.4545
                },
                "mean_radiance_lat": {
                    "name": "Mean radiance at the reservoir",
                    "unit": "kWh m-2 d-1",
                    "long_description": "",
                    "value": 4.66
                },
                "global_radiance": {
                    "name": "Cumulative global horizontal radiance at the reservoir",
                    "unit": "kWh m-2 d-1",
                    "long_description": "",
                    "value": 55.92
                },
                "bottom_temperature": {
                    "name": "Bottom (hypolimnion) temperature in the reservoir",
                    "unit": "deg C",
                    "long_description": "",
                    "value": 19.8254
                },
                "bottom_density": {
                    "name": "Water density at the bottom of the reservoir",
                    "unit": "kg/m3",
                    "long_description": "",
                    "value": 998.2695
                },
                "surface_temperature": {
                    "name": "Surface (epilimnion) temperature in the reservoir",
                    "unit": "deg C",
                    "long_description": "",
                    "value": 24.3
                },
                "surface_density": {
                    "name": "Water density at the surface of the reservoir",
                    "unit": "kg/m3",
                    "long_description": "",
                    "value": 997.2522
                },
                "thermocline_depth": {
                    "name": "Thermocline depth",
                    "unit": "m",
                    "long_description": "",
                    "value": 0.8992
                },
                "nitrogen_load": {
                    "name": "Influent total N load",
                    "unit": "kgN / yr-1",
                    "long_description": "",
                    "value": 76277.2771
                },
                "phosphorus_load": {
                    "name": "Influent total P load",
                    "unit": "kgP / yr-1",
                    "long_description": "",
                    "value": 1245863.4777
                },
                "nitrogen_downstream_conc": {
                    "name": "Downstream TN concentration",
                    "unit": "mg / L",
                    "long_description": "",
                    "value": 0.0054
                }
            }
        }
    }


