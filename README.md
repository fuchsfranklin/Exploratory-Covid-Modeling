# COVID-19 Demographic Analytics

## Overview
This project provides a robust framework for COVID-19 data analysis and predictive modeling using the public dataset from [Our World in Data](https://ourworldindata.org/covid-cases).

## Key Research Questions
This analysis focuses on three critical pandemic management questions:

1.  **Predicting Policy Effectiveness Lag**
    Quantify the time lag between changes in the stringency index (government interventions) and observable changes in new cases or deaths, helping to understand how quickly different countries respond to policy interventions.

2.  **Forecasting "Pandemic Fatigue"**
    Use features like stringency index, new cases, and vaccination rates to predict when a population might show reduced compliance with restrictions (e.g., rising cases despite high stringency), potentially using changes in `positive_rate` or `new_cases_per_test` as proxies.

3.  **Predicting Healthcare System Strain**
    Forecast future ICU or hospital patient counts based on current cases, deaths, policy measures, and potentially incorporating demographic and health system features (e.g., `hospital_beds_per_thousand`, `median_age`).

## Repository Structure
```
.
├── docs/                   # Documentation files
├── eda_outputs/           # Exploratory Data Analysis outputs (visualizations, summaries)
│   ├── per_country/       # Country-specific analysis results
│   └── ...                # Global analysis plots
├── outputs/               # Data processing outputs
│   └── per_country/      # Country-specific processed data files (missing data, summaries)
│       ├── missing_*.csv
│       └── summary_*.csv
├── scripts/              # Python scripts for data fetching, processing, and analysis
├── owid-covid-data.csv   # Main COVID-19 dataset
└── README.md            # Project documentation
```

## Data Source
- [Our World in Data COVID-19 Dataset (CSV)](https://covid.ourworldindata.org/data/owid-covid-data.csv)
- Updated daily, includes global and country-level data on cases, deaths, testing, vaccinations, and more.

## Features
- **Automated data pulling**: Fetches the latest COVID-19 data directly from the authoritative source (requires implementation).
- **Flexible data processing**: Filter by country, date, or metric.
- **Advanced predictive modeling**: Addresses the key research questions using time series analysis, machine learning, and statistical methods.
- **Publication-quality code**: Aims for clean, reproducible, and extensible code.

## Getting Started

### Prerequisites
- Python 3.8+
- Required Python packages (will be specified in `requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - statsmodels
  - matplotlib
  - seaborn
  - requests (for data fetching)
  - dtaidistance (optional, for DTW)

### Installation
```bash
# Clone the repository
git clone <repository_url>
cd Continuous-Glucose-Prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies (requirements.txt needs to be created)
# pip install -r requirements.txt
```

## Example Usage (Conceptual)

```python
# Example for Policy Lag Analysis
from scripts.policy_effectiveness_lag import PolicyLagAnalyzer

analyzer = PolicyLagAnalyzer(data_path='owid-covid-data.csv')
lag_results = analyzer.find_policy_effectiveness_lag(
    countries=['Germany', 'United States'],
    metric='new_cases'
)
print(lag_results)

# Similar conceptual examples for Pandemic Fatigue and Healthcare Strain
```

## Data Source Reference
- [Our World in Data COVID-19 Dataset](https://ourworldindata.org/covid-cases)
- [CSV Download Link](https://covid.ourworldindata.org/data/owid-covid-data.csv)

## License
MIT
