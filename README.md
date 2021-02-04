# Energy-poverty-US

This project explores how different indicators used to measure the prevalence of energy poverty can shape the recognition and alleviation efforts regarding the issue; and bridge a gap between the quantification of energy use in the residential energy sector at high geographic resolution across New York and California.

The project aims to answer the following questions:
- How well can we use energy consumption predictions at the household level with scarce geographical information (RECS) to understand energy consumption patterns at the census tract level? 
- How do different energy expenditure indicators influence the counting/recognition of housholds as enrgy poor in the areas of study?
- What aspects of energy vulnerability cannot be understood uniquely through expenditure thresholds?
- What proxies can be used to understand "Hidden Energy Poverty"?
- How can energy efficiency proxies and expenditure metrics compement each other to undertand the multidimensional nature of energy insecurity?
- Finally, how is energy poverty distributed in California and New York? 

Data used is publicly available from the Residential Energy Consumption Survey (RECS), the American Community Survey (ACS), Public Use Microdata Sample (PUMS), and the DOE’s Low-Income Energy Affordability (LEAD) data. Data collection efforts on weather variables and utility disconnection/arrears are currently underway.

The project is structured into:
- ACS_EDA.ipynb contains the interaction with the American Community Survey's API and the cleaning steps for the out of sample observations to be used in the project
- Energy_Poverty_US.ipynb contains the summary file of the project, with model selection, and data explorations
- recs_EDA.py performs data cleaning functions on RECS dataset
- regression_recs.py test/train of RECS data using OLS/Regularization/Ensemble methods/KNN to predict energy usage 
- classification_recs.py test/train of RECS data using SVC/Ensemble methods to predict prevalence of "inability to heat the home"
- allocation_acs.py uses ACS cleaned dataset to predict energy usage (regression) and prevalence of "inability to heat the home" (classification) per census block group using best models
