# Tool for calculating biogenic greenhouse gas (GHG) emissions from reservoirs

A quick run-through of a GHG emission calculation process for a hypothetical reservoir is given in the Jupyter notebook [run_CO2_emission.ipynb](./run_CO2_emission.ipynb)

## GAS EMISSIONS - GENERAL INFORMATION
All constants/parameters for the calculation of each gas can be adjusted in ./config/emissions/config.ini

#### Temperature-dependent diffusion
CO2 and CH4 emission calculations require estimation of temperature-dependent gas diffusion.
The monthly average temperatures can be obtained from Global Climate database (Hijmans et al., 2005).
Average temperatures for the period 1950- 2000 are available at: https://www.worldclim.org/data/monthlywth.html
See also:
Willmott, C.J. & Matsuura, K. (2001). http://climate.geog.udel.edu/~climate/html_pages/download.html and http://climate.geog.udel.edu/%7Eclimate/html_pages/download.html


####

## 1. Carbon Dioxide (CO2) emissions estimation

### Model description

### Constants/parameters
1. *c_1* [provide definition]
2. *age* [provide definition]
3. *temp* [provide definition]
4. *resArea* [provide definition]
5. *soilC* [provide definition]
6. *ResTP* [provide definition]
7. *calc* [provide definition]
8. *conv_coeff* [provide definition]

### Model inputs

### Model outputs

### Usage:


## 2. Methane (CH4) emissions estimation

### Model description

### Constants/parameters
1. *conv_coeff* = 16.55
#### CH4 diffusion
2. *int_diff* [provide definition]
3. *age_diff* [provide definition]
4. *littoral_diff* [provide definition]
5. *eff_temp_CH4* [provide definition]
#### CH4 ebullition
6. *int_ebull* [provide definition]
7. *littoral_ebull* [provide definition]
8. *irrad_ebull* [provide definition]
#### CH4 degassing
9. *int_degas* [provide definition]
10. *tw_degas* [provide definition]
11. *ch4_diff* [provide definition]
12. *ch_diff_age_term* [provide definition]



## 3. Nitrous Oxide (N2O) emissions estimation

Contrary to CO2 and CH4 N2O emissions are not only dependent on the characteristics of the inundated land but are more dependent on the total nitrogen loading to the system and the processes within the reservoir that depend on water residence time and Nitrogen:Phosphorus stoichiometry with regard to N fixation.

The N20 emissions can be calculated with two alternative models, referred to as: Model 1 (N2O_M1) and Model 2 (N2O_M2). Model 2 follows the approach described in Maarvara et al. 2018 [provide reference]. Both models calculate upper and lower bounds on emission estimates.

### Model description

#### Model 1
Model 1 calculates total N2O emissions in three separate steps listed below.
1. Annual denitrification is estimated as a function of the influent total nitrogen (TN) load entering the reservoir with the river inflow
2. Annual N fixation is estimated as a function of the riverine influent TN and total phosphorus (TP) loads.
3. The N2O emissions are estimated by applying a default EF [define EF] of 0.9% to each of the above processes, i.e. denitrification and N fixation, as derived by Beaulieu et al (2011) [[1]](#1). This total quantity is taken as the annual N2O emission from the reservoir.

#### Model 2
Model 2 estimates total N2O emissions as functions of riverine TN loading and water residence time in a single step using the method described in Maarvara et al. 2018 [provide reference]

This model follows the following steps:
1. Annual N fixation is estimated as a function of the riverine input load of TN and TP. It assumes the same EF = 0.9% as in Model 1.
2. EF for denitrification is derived to account for internal consumpton of N2O at longer residence times (see Eq. 10 in Maarvara et al 2018 [reference]). In other words, different value than the default value of 0.9% is used by denitrification is estimated with the same equations as in Model 1.
3. A further adjustment of emissions is made to account for N2O evasion where water N2O pressure is above atmopsheric. In other words, where water to air N2O flux occurs.

### Comparison of Model 1 and Model 2
On the basis of the differences between Model 1 and Model 2, Model 2 is expected to provide lower evasion estimates. The difference in evasion estimates between those two models is expected to increase with water residence time.
At very short residence times [provide a range of values that are considered very short] there should be little difference between the two approaches and Model 1 might even provide lower evasion estimates than Model 2.
Individual estimates of N20 associated with denitrification and nitrification pathways are calculated within the model but do not constitute the main outputs required for gross GHG esimation from reservoirs.

### Model inputs

1. Annual total nitrogen loading : This load estimate employes a regression model for predicting the median annual TN concentration of runoff to the reservoir
2.

### Model outputs

### Usage:


## References
<a id="1">[1]</a>
Beaulieu, J. J., Tank, J. L., Hamilton, S. K., Wollheim, W. M., Hall, R. O.,
Mulholland, P. J., Dahm, C. N. (2011). *Nitrous oxide emission from
denitrification in stream and river networks*. Proceedings of the
National Academy of Sciences of the United States of America, 108(1),
214–219. https://doi.org/10.1073/pnas.1011464108

<a id="2">[2]</a>
