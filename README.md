# Exploratory COVID-19 Modeling: Because There Weren't Already Enough Predictive COVID-19 Projects

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)

*Because if there's one thing the world was missing, it was my take on COVID-19 modeling (not).*

## Project Overview

Let's be honest: by now, COVID-19 prediction models are almost their own pandemic. But while everyone and their neighbor has forecasted cases and deaths, I wanted to explore the pandemic data myself as well, maybe with questions that are slightly less straightforward than predicting cases and waves post-hoc. With demographic data, I thought it was interesting to consider how hospitals really felt the strain, when people collectively ran out of steam (a.k.a. pandemic fatigue), and why policies sometimes seemed to work on their own mysterious schedule.

## Key Findings

### Healthcare Strain Prediction
- **MAE: 4.11** ICU patients per million
- **Top Predictor:** Death rates (43.66% importance)
- **Second:** Hospital patients (11.31%)
- **Training:** 31,285 samples with 55 engineered features

### Pandemic Fatigue Detection
- **Definition:** High stringency (≥60) + cases rising >20%
- **Accuracy:** 89.1% balanced accuracy, 91.3% ROC AUC
- **Coverage:** 10 major countries analyzed

### Policy Effectiveness
- **Methods:** Cross-correlation, Granger causality, wavelet coherence
- **Typical Lag:** 7-21 days between policy and effect
- **Insight:** Early action critical due to implementation lag

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Exploratory-Covid-Modeling.git
cd Exploratory-Covid-Modeling
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### Run Analysis

```bash
python scripts/healthcare_strain.py
```

## Repository Structure

```
.
├── .streamlit/              # Streamlit deployment config
├── dashboard/               # Interactive dashboard
│   └── app.py
├── docs/                    # Documentation
├── eda_outputs/             # Visualizations
├── results/                 # Analysis results
│   ├── healthcare_strain/
│   ├── pandemic_fatigue/
│   └── policy_effectiveness/
├── scripts/                 # Core analysis scripts
│   ├── healthcare_strain.py
│   ├── pandemic_fatigue.py
│   └── policy_effectiveness_lag.py
├── example_analysis.ipynb   # Jupyter examples
├── LICENSE                  # MIT License
├── owid-covid-data.csv      # Main dataset
├── QUICKSTART.md            # Quick start guide
├── README.md                # This file
└── requirements.txt         # Dependencies
```

## Methodologies

### Healthcare Strain Analysis
- Gradient boosting regression with time-series features
- 55 predictive features (lagged indicators, policies, demographics)
- Target: ICU patients per million
- Validation: Time-series cross-validation

### Pandemic Fatigue Analysis
- Binary classification of fatigue periods
- Features: Stringency, demographics, case trends, vaccination
- Definition: Cases increase despite high restrictions
- Methods: Logistic regression, gradient boosting

### Policy Effectiveness Analysis
- Multiple time-series methods
- Cross-correlation, Granger causality, wavelet coherence
- Causal inference: Difference-in-differences, synthetic control
- Regional analysis for high-quality data subsets

## Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Healthcare Strain Report](docs/healthcare_strain_analysis_report.md)
- [Pandemic Fatigue Report](docs/pandemic_fatigue_analysis_report.md)
- [Policy Effectiveness Report](docs/policy_effectiveness_lag_analysis_report.md)
- [Integrated Findings](docs/integrated_findings_report.md)

## Dashboard Features

- **Overview:** Global statistics and interactive world map
- **Healthcare Strain:** ICU predictions with feature importance
- **Pandemic Fatigue:** Detection timeline and metrics
- **Policy Effectiveness:** Lag analysis with correlations
- **Cross-Country Comparison:** Multi-country metric comparison

## Data Sources

- [Our World in Data COVID-19 Dataset](https://ourworldindata.org/covid-cases)
- [COVID-19 Twitter Sentiment Dataset](https://github.com/thepanacealab/covid19_twitter)
- [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility/)
- [Oxford COVID-19 Government Response Tracker](https://www.bsg.ox.ac.uk/research/research-projects/covid-19-government-response-tracker)

## Research Contributions

1. **Methodological Innovation:** Data-driven pandemic fatigue definition
2. **Comparative Analysis:** Healthcare strain predictors across phases
3. **Policy Evaluation:** Multi-method effectiveness assessment
4. **Temporal Insights:** Time-varying intervention relationships
5. **Data Quality:** Critical evaluation of pandemic data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

Franklin Fuchs

## Acknowledgments

- Data provided by [Our World in Data](https://ourworldindata.org/)
- Built with [Streamlit](https://streamlit.io/)
- Analysis powered by scikit-learn, pandas, and plotly

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
