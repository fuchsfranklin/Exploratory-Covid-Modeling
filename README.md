# Exploratory COVID-19 Demographic Modeling and Analysis

## Project Overview
This repository contains a directed analysis of the COVID-19 pandemic demographic trends, focusing on three critical aspects: healthcare system strain, pandemic fatigue, and policy effectiveness. We aim to provide insights into pandemic dynamics that can help shape and understand future public health strategies.s

## Executive Summarys

### Key Findings

1. **Healthcare Strain Prediction:**
   - Successfully modeled ICU utilization rates with Mean Absolute Error of 3.96 ICU patients per million
   - Death rates (43.8%) are the strongest predictor of future ICU demand
   - Models generalized well across different countries and pandemic phases

2. **Pandemic Fatigue Detection:**
   - Operationalized pandemic fatigue as periods with high restriction levels but increasing transmission
   - Achieved 89.1% balanced accuracy and 91.3% ROC AUC in detecting fatigue periods
   - Identified key predictors of pandemic fatigue across 10 major countries

3. **Policy Effectiveness Lag Analysis:**
   - Applied multiple time-series methods (Cross-Correlation, Granger Causality, Transfer Function, Wavelet Coherence)
   - Data quality challenges prevented statistically significant lag identification
   - Findings emphasize the complex relationships between policy implementation and outcomes

## Repository Structure

```
.
├── docs/                                # Documentation files
├── eda_outputs/                         # Exploratory Data Analysis visualizations
│   ├── country_eda_summary.csv          # Country-level EDA summary statistics
│   ├── global_*.png                     # Global trend visualizations
│   └── per_country/                     # Country-specific visualizations
├── models/                              # Saved model files
│   └── healthcare_strain_predictor*.pkl # Healthcare strain prediction models
├── outputs/                             # Data processing outputs
│   └── per_country/                     # Country-specific processed data
├── results/                             # Analysis results by research question
│   ├── healthcare_strain/               # Healthcare strain analysis results
│   ├── pandemic_fatigue/                # Pandemic fatigue analysis results
│   └── policy_effectiveness/            # Policy effectiveness analysis results
├── scripts/                             # Python scripts for analysis
│   ├── healthcare_strain.py             # Healthcare strain prediction
│   ├── pandemic_fatigue.py              # Pandemic fatigue detection
│   └── policy_effectiveness_lag.py      # Policy lag analysis
├── *.md                                 # Detailed analysis reports
├── owid-covid-data.csv                  # Main COVID-19 dataset
└── README.md                            # Project documentation
```

## Detailed Analysis Reports

The repository includes comprehensive reports for each analysis:

- [Healthcare Strain Analysis Report](healthcare_strain_analysis_report.md)
- [Pandemic Fatigue Analysis Report](pandemic_fatigue_analysis_report.md)
- [Policy Effectiveness Lag Analysis Report](policy_effectiveness_lag_analysis_report.md)

## Methodologies

### Healthcare Strain Analysis
- **Approach:** Gradient boosting regression with time-series feature engineering
- **Features:** 55 predictive features including lagged indicators, policy measures, and demographics
- **Target:** ICU patients per million population
- **Key Innovation:** Incorporation of both epidemiological and policy indicators with appropriate time lags

### Pandemic Fatigue Analysis
- **Approach:** Logistic regression for binary classification of fatigue periods
- **Definition:** "Pandemic fatigue" identified as periods when cases increase despite high stringency measures
- **Features:** Policy stringency, demographic factors, case/death trends, and vaccination rates
- **Key Innovation:** Data-driven operationalization of a complex socio-behavioral phenomenon

### Policy Effectiveness Lag Analysis
- **Approach:** Multiple complementary time-series methods
- **Methods:** Cross-Correlation Function Analysis, Granger Causality, Transfer Function Modeling, Wavelet Coherence
- **Countries:** 10 major countries with diverse COVID-19 responses
- **Key Innovation:** Novel application of wavelet coherence to analyze time-frequency relationships in policy effects

## Data Source
- [Our World in Data COVID-19 Dataset](https://ourworldindata.org/covid-cases) ([CSV Download](https://covid.ourworldindata.org/data/owid-covid-data.csv))
- Contains comprehensive daily data on cases, deaths, testing, vaccination, hospital utilization, and policy measures

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, PyWavelets

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Exploratory-Covid-Modeling.git
cd Exploratory-Covid-Modeling

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn pywavelets
```

### Example Usage

```python
# Healthcare Strain Analysis
from scripts.healthcare_strain import HealthcareStrainPredictor

predictor = HealthcareStrainPredictor('owid-covid-data.csv')
predictions = predictor.predict_icu_utilization(country='Germany', horizon_days=14)

# Pandemic Fatigue Analysis
from scripts.pandemic_fatigue import PandemicFatigueDetector

detector = PandemicFatigueDetector('owid-covid-data.csv')
fatigue_periods = detector.detect_fatigue(country='United States')

# Policy Effectiveness Analysis
from scripts.policy_effectiveness_lag import PolicyLagAnalyzer

analyzer = PolicyLagAnalyzer('owid-covid-data.csv')
lag_results = analyzer.analyze_policy_lags(country='France', method='ccf')
```

## Key Insights and Implications

1. **Healthcare Planning:** The healthcare strain model enables proactive resource allocation by providing accurate ICU utilization forecasts.

2. **Public Health Messaging:** Pandemic fatigue detection can inform targeted communication strategies when compliance is likely to wane.

3. **Policy Timing:** Understanding the complex relationship between interventions and outcomes emphasizes the need for early, proactive measures.

4. **Data Quality:** Significant data quality challenges highlight the importance of consistent, standardized pandemic data collection.

5. **Integrated Approach:** The combined insights from all three analyses provide a more complete understanding of pandemic dynamics.

## License
MIT License

## Contributors
- Project Lead: [Your Name]
- Data Analysis: [Contributor Names]
- Modeling: [Contributor Names]
