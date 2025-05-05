# COVID-19 Exploratory Data Analysis (EDA): Global and Per-Country Results

## Global Overview

### 1. Setting the Stage: The Global Pandemic in Numbers

- **Data coverage:** 429,435 rows (all countries, all dates)
- **Countries/regions:** 255
- **Columns:** 67
- **Missing values:** Most columns have some missing data; see per-country CSVs for details.

![Global New Cases & Deaths](global_new_cases_deaths.png)

**Narrative:**
COVID-19 swept across the globe in multiple waves, with each country experiencing unique surges and declines. The global time series plot above shows the aggregate daily new cases and deaths, highlighting the pandemic’s worldwide impact.

### 2. Policy Response: Did Stringency Save Lives?

![Global Stringency vs. New Cases](global_policy_vs_cases.png)

**Narrative:**
Government interventions, measured by the mean stringency index, varied over time and across countries. The data shows that stricter measures often preceded downturns in global case numbers, though the relationship is complex and influenced by many factors.

### 3. Vaccination: The Turning Point

![Global Vaccination vs. New Cases](global_vaccination_vs_cases.png)

**Narrative:**
The rollout of vaccines globally marked a turning point. As the mean number of people vaccinated per hundred increased, new cases began to decline, demonstrating the effectiveness of immunization on a global scale.

### 4. Healthcare System Strain: ICU and Hospitalization Trends

![Global ICU and Hospital Patients](global_icu_and_hospital.png)

**Narrative:**
Peaks in ICU and hospital admissions mirrored global case surges, pushing healthcare systems to their limits. These trends underscore the importance of flattening the curve to prevent overwhelming hospitals worldwide.

### 5. Data Quality and Limitations

- **Missing values:**
    - Many columns have missing data, especially for ICU/hospitalization, testing, and excess mortality.
    - See `country_eda_summary.csv` and `missing_{country}.csv` for detailed per-country missing value reports.

**Narrative:**
No dataset is perfect. Missing values and reporting inconsistencies are transparently summarized. This honesty is crucial for scientific integrity and for understanding the limitations of our conclusions.

---

## Per-Country EDA

- For each country, summary statistics are saved as `summary_{country}.csv`.
- Missing value reports are saved as `missing_{country}.csv`.
- The file `country_eda_summary.csv` provides a tabular overview of all countries, including row counts, date ranges, and missing data columns.

---

# How to Reproduce This EDA

Run the following command to generate the analysis and plots:

```bash
python covid_data.py
```

---

# Conclusion

This EDA provides a comprehensive, narrative-driven overview of the COVID-19 pandemic globally and for each country, combining statistical summaries, visualizations, and context. The story told by the data is one of challenge, adaptation, and the power of science and policy to shape outcomes.

---

*All data sourced from [Our World in Data](https://ourworldindata.org/covid-cases).*
