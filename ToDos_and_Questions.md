## QUESTIONS FOR CHRIS

1. Units for pre-impoundment emissions. For CO2 the unit is t CO2eq/ha/yr, while for CH4 it's in  kg CH4/ha/yr. Double check if there's no error in one of the tables.
2. EFF_TEMP: the coefficient for CO2 is 0.05 but for CH4 it is 0.052. Is this a roundoff error or are these coefficients different (albeit only slightly)?
3. McDowell's Landscape TP export model has one less biome to the TP model used in CO2 calcs, namely tundra
4. In calculation of N2O emission from Denitrification in mmol N m-2 yr-1 from Model 2 (DS2) there is a double minus sign in the equation. Is this correct or a typo?
5. Can we add uncertainty estimates to the predictions? I assume it is not
that difficult to do using McDowell's regression since the error estimates on
the coefficients are already provided.
6. In pre-impoundment emissions for CH4 and CO2 the area used for the calculations is called catchment area. Is this the inundated area?
7. We need to also have estimates of CO2 emissions due to construction in order to calculate total CO2 emissions resulting form dam construction and operation.
8. N2O total emission does not have a time horizon. Is it assumed that it is over 100 years or does it happen as long as the reservoir is in place?
9. In calculations of wind speed at 10m, does the input wind need to be at 50 or does the relationship interpolate from other wind speeds provided that the hight of the measurement is provided as input variable x?
10. There are different versions of water density vs temperature equation in hypolimnion and epilimnion density calculations. One might have a missing bracket somewhere. Needs to be checked which one is correct.
11. Can you explain the (16/12) coefficient in CH4 emmisions, specifically integrated emission over 100 years.


## TODOs:
### 1. Package development
* WRITE TESTS
* WRITE DOCUMENTATION - sphinx?
* CREATE PACKAGING - setuptools? poetry?
* DECIDE UPON LICENSING - MIT? GPL?
* ADD TO PYPI AND TRAVIS
### 2. Calculation methodology
* CASCADED TN NITROGEN LOADING
* ADDITION OF UNERTAINTIES IN THE CALCULATIONS?

