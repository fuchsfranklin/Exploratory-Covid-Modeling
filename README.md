# Exploratory COVID-19 Modeling: Because There Weren't Already Enough Predictive COVID-19 Projects

*Because if there’s one thing the world was missing, it was my take on COVID-19 modeling (not).*

## Project Overview

Let’s be honest: by now, COVID-19 prediction models are almost their own pandemic. But while everyone and their neighbor has forecasted cases and deaths, I wanted to explore the pandemic data myself as well, maybe with questions that are slightly less straightforward than predicting cases and waves post-hoc. With demographic data, I thought is was interesting to consider about how hospitals really felt the strain, when people collectively ran out of steam (a.k.a. pandemic fatigue), and why policies sometimes seemed to work on their own mysterious schedule.


## Executive Summary

### Key Findings

1. **Healthcare Strain Prediction:**
   - Successfully modeled ICU utilization rates with Mean Absolute Error of 3.96 ICU patients per million
   - Death rates (43.8%) are the strongest predictor of future ICU demand
   - Models generalized well across different countries and pandemic phases
   - Advanced techniques including LSTM and ensemble methods enhance predictive accuracy

2. **Pandemic Fatigue Detection:**
   - Operationalized pandemic fatigue as periods with high restriction levels but increasing transmission
   - Achieved 89.1% balanced accuracy and 91.3% ROC AUC in detecting fatigue periods
   - Identified key predictors of pandemic fatigue across 10 major countries
   - Incorporated novel data sources including sentiment analysis from social media to understand public response

3. **Policy Effectiveness Lag Analysis:**
   - Applied multiple time-series methods (Cross-Correlation, Granger Causality, Transfer Function, Wavelet Coherence)
   - Data quality challenges prevented statistically significant lag identification
   - Findings emphasize the complex relationships between policy implementation and outcomes
   - Applied causal inference techniques to high-quality data subsets to strengthen methodological rigor

## Project Innovations

### Novel Research Angles
- **Variant-Specific Analysis**: Exploring differential impacts of major COVID variants on healthcare systems
- **Public Sentiment Integration**: Combining epidemiological data with social media sentiment to understand pandemic fatigue
- **Regional Adaptation Patterns**: Analyzing how different regions adapted to similar policy interventions over time
- **Subpopulation Impact Assessment**: Examining effects across demographic groups where data quality permits

### Methodological Advancements
- **Advanced Deep Learning Models**: LSTM networks for time-series prediction of healthcare strain
- **Ensemble Methods**: Combining multiple predictive models to improve accuracy and robustness
- **Causal Inference Techniques**: Applying difference-in-differences and synthetic control methods to isolate policy effects
- **Rigorous Validation**: Comprehensive cross-validation and sensitivity analyses across multiple countries

### Interactive Exploration
- **Web Dashboard**: Interactive visualization of key findings using Streamlit
- **Time-Series Explorer**: Dynamic tool for examining temporal relationships between policies and outcomes
- **Country Comparison Tool**: Visual analysis of cross-country differences in pandemic trajectories

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
├── dashboard/                           # Interactive dashboard components
│   ├── app.py                           # Streamlit application entry point
│   └── components/                      # Dashboard UI components
├── *.md                                 # Detailed analysis reports
├── owid-covid-data.csv                  # Main COVID-19 dataset
└── README.md                            # Project documentation
```

## Detailed Analysis Reports

The repository includes comprehensive reports for each analysis:

- [Healthcare Strain Analysis Report](healthcare_strain_analysis_report.md)
- [Pandemic Fatigue Analysis Report](pandemic_fatigue_analysis_report.md)
- [Policy Effectiveness Lag Analysis Report](policy_effectiveness_lag_analysis_report.md)
- [Integrated Findings and Recommendations](integrated_findings_report.md)

## Methodologies

### Healthcare Strain Analysis
- **Approach:** Gradient boosting regression with time-series feature engineering
- **Features:** 55 predictive features including lagged indicators, policy measures, and demographics
- **Target:** ICU patients per million population
- **Key Innovation:** Incorporation of both epidemiological and policy indicators with appropriate time lags
- **Advanced Models:** LSTM networks for capturing complex temporal patterns and ensemble methods for improved robustness
- **Feature Selection:** Recursive feature elimination to identify most impactful predictors
- **Validation:** K-fold cross-validation with time-series splits to ensure temporal generalization

### Pandemic Fatigue Analysis
- **Approach:** Logistic regression for binary classification of fatigue periods
- **Definition:** "Pandemic fatigue" identified as periods when cases increase despite high stringency measures
- **Features:** Policy stringency, demographic factors, case/death trends, and vaccination rates
- **Key Innovation:** Data-driven operationalization of a complex socio-behavioral phenomenon
- **Social Media Integration:** Analysis of Twitter sentiment related to COVID policies in selected regions
- **Advanced Classification:** Gradient boosting and ensemble methods for improved prediction
- **Feature Enhancement:** Addition of mobility data and public opinion surveys where available

### Policy Effectiveness Lag Analysis
- **Approach:** Multiple complementary time-series methods
- **Methods:** Cross-Correlation Function Analysis, Granger Causality, Transfer Function Modeling, Wavelet Coherence
- **Countries:** 10 major countries with diverse COVID-19 responses
- **Key Innovation:** Novel application of wavelet coherence to analyze time-frequency relationships in policy effects
- **Causal Methods:** Difference-in-differences and synthetic control approaches for high-quality data subsets
- **Regional Focus:** Targeted analysis of US states and European regions with reliable data
- **Policy Decomposition:** Separating effects of specific interventions (masks, lockdowns, etc.) where data permits

## Data Source
- [Our World in Data COVID-19 Dataset](https://ourworldindata.org/covid-cases) ([CSV Download](https://covid.ourworldindata.org/data/owid-covid-data.csv))
- Contains comprehensive daily data on cases, deaths, testing, vaccination, hospital utilization, and policy measures
- **Additional Data Sources:**
  - [COVID-19 Twitter Sentiment Dataset](https://github.com/thepanacealab/covid19_twitter)
  - [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility/)
  - [Oxford COVID-19 Government Response Tracker](https://www.bsg.ox.ac.uk/research/research-projects/covid-19-government-response-tracker)

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, PyWavelets
- Additional packages: tensorflow, streamlit, pytorch, transformers, pycaret

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
pip install tensorflow streamlit pytorch transformers pycaret plotly
```

### Running the Interactive Dashboard
```bash
# Navigate to the dashboard directory
cd dashboard

# Launch the Streamlit app
streamlit run app.py
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

## Research Contributions & Publishable Insights

This project contributes to the COVID-19 research landscape in several ways:

1. **Methodological Innovation**: Novel data-driven definitions of pandemic fatigue that complement traditional survey-based approaches

2. **Comparative Analysis**: Systematic comparison of healthcare strain predictors across pandemic phases and geographies

3. **Policy Evaluation Framework**: Development of a multi-method approach to assess policy effectiveness despite data limitations

4. **Temporal Insights**: Identification of time-varying relationships between interventions and outcomes using wavelet coherence

5. **Data Quality Assessment**: Critical evaluation of COVID-19 data quality and its implications for research findings

Publication targets include journals focusing on public health informatics, computational epidemiology, and healthcare systems research.

## Future Directions

1. **Integration of Genomic Data**: Incorporating variant prevalence data to assess impact of mutations on healthcare strain

2. **Expanded Sentiment Analysis**: Deeper analysis of social media content to understand regional variations in pandemic response

3. **Long-term Impact Assessment**: Extending analysis to post-acute phases and healthcare system recovery periods

4. **Transfer Learning**: Applying lessons from COVID-19 to model other infectious disease outbreaks

5. **Policy Optimization Models**: Developing prescriptive models for intervention timing and intensity optimization

## Contributors
Franklin Fuchs
