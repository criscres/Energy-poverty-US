# Energy-poverty-US

This repository supports the analysis presented in the article:  
**"Uncovering the dimensions and distribution of household energy insecurity in the United States"**, submitted to *Nature Energy*.

## Overview
This project introduces a multidimensional framework to measure household energy insecurity across U.S. census tracts. Using machine learning models trained on the Residential Energy Consumption Survey (RECS), we estimate ten indicators of economic, behavioral, and physical energy hardship. These models are applied to representative "typical" households in over 82,000 U.S. census tracts using American Community Survey (ACS) data. The resulting spatial and demographic insights are used to assess the adequacy of existing energy assistance and weatherization programs.

## Research Questions
- How can we move beyond traditional energy burden metrics to better identify and map energy-insecure households?
- What are the demographic and geographic patterns in economic, behavioral, and physical forms of energy insecurity?
- How well do existing assistance programs align with neighborhoods facing multidimensional energy hardship?

## Repository Structure

### `figures/`
Code for generating visualizations, including:
- National maps of energy insecurity prevalence
- Disparities by race, gender, rurality, and income
- Program eligibility coverage across risk levels

### `apply_models_ACS/`
Scripts to:
- Download and preprocess ACS data
- Match tract-level characteristics to RECS-trained model inputs
- Generate predictions for ten energy insecurity metrics
- Export census tract-level risk flags

### `launch_models_RECS/`
Scripts to:
- Clean and process RECS 2020 data
- Train and evaluate machine learning models for:
  - Regression tasks (e.g., energy consumption, energy burden)
  - Classification tasks (e.g., under-consumption, coping behaviors)
- Select models based on out-of-sample R² and AUROC
- Save models for deployment on ACS tract-level data

## Data Sources
- [Residential Energy Consumption Survey (RECS) 2020](https://www.eia.gov/consumption/residential/)
- [American Community Survey (ACS) 2016–2020](https://www.census.gov/programs-surveys/acs)
- [DOE Low-Income Energy Affordability Data (LEAD) Tool](https://www.energy.gov/eere/slsc/maps/lead-tool)
- [Utility Disconnection Dashboard – Indiana University](https://energyjustice.indiana.edu/disconnection-dashboard/)

## Key Methodological References
- Hernández, D. (2016). *Understanding "energy insecurity" and why it matters to health.* Social Science & Medicine. [DOI](https://doi.org/10.1016/j.socscimed.2016.08.029)
- Cong, S., et al. (2022). *Unveiling hidden energy poverty using the energy equity gap.* Nature Communications. [DOI](https://doi.org/10.1038/s41467-022-30146-5)
- Reames, T. G. (2016). *Targeting energy justice.* Energy Policy. [DOI](https://doi.org/10.1016/j.enpol.2016.07.048)
- Scheier, E., & Kittner, N. (2022). *A measurement strategy to address disparities across household energy burdens.* Nature Communications. [DOI](https://doi.org/10.1038/s41467-021-27673-y)
- Batlle, C., et al. (2024). *US federal resource allocations are inconsistent with concentrations of energy poverty.* Science Advances. [DOI](https://doi.org/10.1126/sciadv.adp8183)

## Citation
If you use this code, methodology, or indicators, please cite our article (forthcoming in *Nature Energy*) and consider starring the repository.

---

**Maintainer**: Cristina Crespo Montañés  
**GitHub**: [@criscres](https://github.com/criscres)
