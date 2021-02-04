# Energy-poverty-US

This project explores how different indicators used to measure the prevalence of energy poverty can shape the recognition and alleviation efforts regarding the issue; and bridge a gap between the quantification of energy use in the residential energy sector at high geographic resolution across New York and California.

The project aims to answer the following questions:
- How well can we use energy consumption predictions at the household level with scarce geographical information (RECS) to understand energy consumption patterns at the census tract level? 
- How do different energy expenditure indicators influence the counting/recognition of housholds as enrgy poor in the areas of study?
- What aspects of energy vulnerabilities cannot be understood uniquely through expenditure thresholds?
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


## References 
Bednar, D.J., Reames, T.G. (2020) “Recognition of and response to energy poverty in the United States”. Nature Energy 5, 432–439. https://doi.org/10.1038/s41560-020-0582-0

Bouzarovski S., Petrova S. and Sarlamanov R., (2012) “Energy poverty policies in the EU: A critical perspective”, Energy Policy, 49, (C), 76-82

Cornelis, M, (2020) "Energy Efficiency, the Overlooked Climate Emergency Solution," Economic Policy, Russian Presidential Academy of National Economy and Public Administration, Volume 2, 48-67.

EU Energy Poverty Observatory. “Indicators & Data” Accessed on Dec 2020. https:// www.energypoverty.eu/indicators-data

Harrison C. and Popke J. (2011) “ ‘Because You Got to Have Heat’: The Networked Assemblage of Energy Poverty in Eastern North Carolina”, Annals of the Association of American Geographers, 101:4, 949-961, doi: 10.1080/00045608.2011.569659

Hernández D. (2016). “Understanding 'energy insecurity' and why it matters to health”. Social science & medicine (1982), 167, 1–10. https://doi.org/10.1016/j.socscimed.2016.08.029

Reames T.G. (2016) “Targeting energy justice: Exploring spatial, racial/ethnic and socioeconomic disparities in urban residential heating energy efficiency” Energy Policy, Volume 97, 549-558, ISSN 0301-4215, https://doi.org/10.1016/j.enpol.2016.07.048.

Thomson, H., Bouzarovski, S., and Snell, C. (2017) “Rethinking the measurement of energy poverty in Europe: A critical analysis of indicators and data”. Indoor and Built Environment, 26(7), 879–901. https://doi.org/10.1177/1420326X17699260

Wenwen Zhang,Caleb Robinson, Subhrajit Guhathakurta, Venu M. Garikapati, Bistra Dilkina, Marilyn A. Brown , and Ram M. Pendyala. (2018) Estimating Residential Energy Consumption in Metropolitan Areas: A Microsimulation Approach https://doi.org/10.1016/j.energy.2018.04.161
