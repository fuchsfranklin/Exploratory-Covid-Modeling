# Exploratory COVID-19 Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)

An exploratory analysis of COVID-19 pandemic dynamics, focusing on three questions
that go beyond standard case/death forecasting:

1. Can we predict ICU strain from lagged indicators?
2. When do populations exhibit "pandemic fatigue" — rising cases despite strict restrictions?
3. How long does it take for policy changes to show measurable effects?

## Key Findings

### Healthcare Strain Prediction
- **MAE: ~5.5** ICU patients per million (genuine forward prediction using only lagged features)
- **Top predictor:** Deaths per million (7-day lag) — 44.7% importance
- **Model:** Gradient Boosting with 46 lagged + demographic features
- **Design:** No contemporaneous features used, enabling real 7-14 day forecasting

### Pandemic Fatigue Detection
- **Balanced accuracy:** 0.908, **ROC AUC:** 0.969
- **Definition:** Stringency ≥ 60 AND cases rising > 20% over 14 days
- **Model:** Logistic Regression with interaction and volatility features
- **Finding:** ~8% of country-days across 176 countries meet fatigue criteria

### Policy Effectiveness Lag
- **Reproduction rate:** 10/10 countries show significant lag, median ~12 days
- **Deaths:** 5/10 countries significant, median ~15 days
- **Methods:** Cross-correlation, Granger causality, wavelet coherence
- **Insight:** Policy effects on R are more consistently detectable than on raw case counts

## Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Run Analysis Scripts

```bash
python scripts/healthcare_strain.py
python scripts/pandemic_fatigue.py
python scripts/policy_effectiveness_lag.py
```

Each script produces timestamped results in `results/<analysis>/`.

## Repository Structure

```
├── dashboard/
│   └── app.py                  # Streamlit dashboard (reads from results/)
├── scripts/
│   ├── healthcare_strain.py    # ICU prediction model
│   ├── pandemic_fatigue.py     # Fatigue classification model
│   └── policy_effectiveness_lag.py  # Lag analysis (CCF, Granger, wavelet)
├── results/                    # Model outputs (auto-generated)
│   ├── healthcare_strain/
│   ├── pandemic_fatigue/
│   └── policy_effectiveness/
├── owid-covid-data.csv         # OWID dataset
├── example_analysis.ipynb      # Jupyter notebook examples
├── requirements.txt
└── LICENSE
```

## Methodology Notes

### Healthcare Strain
Uses only **lagged** versions of dynamic features (7 and 14 day lags) as predictors.
This avoids data leakage from contemporaneous indicators (e.g., using current hospital
patients to predict current ICU patients) and enables genuine forward-looking prediction.
The tradeoff is higher MAE compared to models that use contemporaneous features, but the
predictions are actually useful for planning.

### Pandemic Fatigue
A binary classification problem: is this country-day a "fatigue period"? Features include
smoothed epidemiological indicators, their interactions with stringency, rolling volatility,
and z-scores. The class imbalance (~8% positive) is handled via balanced class weights.

### Policy Effectiveness
Three complementary methods are applied per country:
- **Cross-correlation:** Identifies the lag with strongest negative correlation between
  differenced stringency and differenced outcome series
- **Granger causality:** Tests whether lagged stringency values improve prediction of outcomes
- **Wavelet coherence:** Captures time-varying, frequency-dependent relationships

Results are aggregated across countries with consensus lag estimates.

## Limitations

- Analysis is retrospective — all data is historical
- The OWID dataset has varying completeness across countries and time periods
- Fatigue definition is a simplification; real compliance behavior is more nuanced
- Policy lag analysis uses aggregate stringency index, not individual policy components
- No causal claims are made — these are observational associations

## Data Source

[Our World in Data COVID-19 Dataset](https://ourworldindata.org/covid-cases),
which incorporates the Oxford COVID-19 Government Response Tracker for policy stringency.

## License

MIT — see [LICENSE](LICENSE).

## Author

Franklin Fuchs
