# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Navigation

### Overview Page
- View global COVID-19 statistics
- Explore trends with interactive charts
- Use the world map to compare countries

### Healthcare Strain Page
- Select a country from the dropdown
- View ICU/hospital utilization predictions
- Explore feature importance (what drives ICU demand)
- Compare model performance

### Pandemic Fatigue Page
- Select a country to analyze
- See when fatigue occurred (high restrictions + rising cases)
- View fatigue metrics and timeline

### Policy Effectiveness Page
- Select a country to analyze
- See relationship between policies and transmission
- View optimal policy lag (how long until effects appear)

### Cross-Country Comparison
- Select 2-5 countries
- Choose a metric to compare
- View trends and summary statistics

---

## 🔬 Running Analysis Scripts

### Healthcare Strain Prediction

```bash
python scripts/healthcare_strain.py
```

**What it does:**
- Trains a model to predict ICU utilization
- Uses 55 features including cases, deaths, demographics, policies
- Outputs: trained model, predictions, feature importance

**Results location:** `results/healthcare_strain/[timestamp]/`

**Key outputs:**
- `model_pipeline.pkl` - Trained model
- `feature_importances.csv` - Feature rankings
- `test_predictions_vs_actual.csv` - Model predictions
- `run_summary.txt` - Performance metrics

---

## 💡 Key Findings

### Healthcare Strain
- **MAE: 4.11** ICU patients per million
- **Top predictor:** Deaths (43.66% importance)
- **Second:** Hospital patients (11.31%)
- **Insight:** ICU strain predictable 7-14 days ahead

### Pandemic Fatigue
- **Definition:** High stringency (≥60) + cases rising >20%
- **Implication:** Policies losing effectiveness
- **Action:** Adjust communication and strategy

### Policy Effectiveness
- **Typical lag:** 7-21 days
- **Varies by:** Country, intervention type, pandemic phase
- **Insight:** Early action critical due to lag

---

## 📁 Project Structure

```
Exploratory-Covid-Modeling/
├── dashboard/          # Interactive Streamlit app
├── scripts/            # Analysis scripts
├── results/            # Model outputs and predictions
├── docs/               # Detailed documentation
├── owid-covid-data.csv # Main dataset
└── requirements.txt    # Dependencies
```

---

## 🛠️ Customization

### Change Model Parameters

Edit `scripts/healthcare_strain.py`:

```python
# Line ~600
MODEL_CHOICE = 'GradientBoosting'  # or 'RandomForest'
PERFORM_TUNING = False             # True for hyperparameter tuning
LAG_PERIODS = [7, 14]              # Time lags to use
```

### Add Countries to Dashboard

Countries are automatically filtered based on data availability. To see all countries:
- Check `owid-covid-data.csv` for available locations
- Dashboard filters to countries with 100+ days of data

---

## 📖 Further Reading

- [Full README](README.md) - Complete project documentation
- [Healthcare Strain Report](docs/healthcare_strain_analysis_report.md)
- [Pandemic Fatigue Report](docs/pandemic_fatigue_analysis_report.md)
- [Policy Effectiveness Report](docs/policy_effectiveness_lag_analysis_report.md)
- [Test Report](TEST_REPORT.md) - Verification results

---

## ❓ Troubleshooting

### Dashboard won't start
```bash
# Make sure you're in the project root directory
cd Exploratory-Covid-Modeling
streamlit run dashboard/app.py
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Data file not found
- Ensure `owid-covid-data.csv` is in the project root
- Download from: https://covid.ourworldindata.org/data/owid-covid-data.csv

### Script errors
- Check Python version (3.8+ required)
- Verify all dependencies installed
- Check console output for specific error messages

---

## 🎯 Next Steps

1. **Explore the dashboard** - Try different countries and metrics
2. **Run analysis scripts** - Generate your own predictions
3. **Read detailed reports** - Understand methodology and findings
4. **Customize analyses** - Modify parameters for your research questions

---

## 📧 Support

For issues or questions, refer to the detailed documentation in the `docs/` folder or check the main [README](README.md).
