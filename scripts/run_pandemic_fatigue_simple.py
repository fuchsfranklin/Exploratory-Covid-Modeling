"""
Run the Pandemic Fatigue Analysis with enhanced features
including sentiment analysis integration and advanced classification methods.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

# Make sure the script can find the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PandemicFatiguePredictor class
from pandemic_fatigue import PandemicFatiguePredictor

# Set up the results directory
results_dir = "../results/pandemic_fatigue"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(results_dir, f"{run_id}_GradientBoosting")
os.makedirs(run_dir, exist_ok=True)

print(f"Starting pandemic fatigue analysis run {run_id}")
print(f"Results will be saved to: {run_dir}")

# Define fatigue parameters
fatigue_def_params = {
    'stringency_threshold': 60,  # High stringency threshold
    'case_increase_window': 14,  # Window to detect case increases
    'case_increase_threshold': 0.2,  # 20% increase in cases
}

# Initialize the predictor with enhanced features
predictor = PandemicFatiguePredictor(
    data_path='owid-covid-data.csv',
    model_type='GradientBoosting',  # Using gradient boosting for enhanced classification
    tune_hyperparameters=True,  # Enable hyperparameter tuning
    fatigue_def_params=fatigue_def_params,
    use_sentiment_analysis=True,  # Enable sentiment analysis integration
    sentiment_model='vader',  # Using VADER for sentiment analysis
    use_mobility_data=False,  # No mobility data available
    results_base_dir=results_dir,
    run_id=run_id
)

# Load and preprocess the data
print("Loading and preprocessing data...")
predictor.load_and_preprocess_data()

# Select countries for analysis based on data quality
countries_to_analyze = [
    'United States', 'Germany', 'United Kingdom', 'France',
    'Italy', 'Spain', 'Canada', 'Japan', 'South Korea', 'Brazil'
]

# Define fatigue periods based on our definition
print("Defining fatigue periods...")
predictor.define_fatigue_periods()
data_with_target = predictor.data

# Debug: print available columns and target variable name
print(f"Available columns: {list(data_with_target.columns)}")
print(f"Target variable name: {predictor.target_variable_name}")

# Prepare features for modeling
features = [
    'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
    'reproduction_rate', 'stringency_index', 'population_density',
    'median_age', 'gdp_per_capita', 'human_development_index',
    'positive_rate', 'people_fully_vaccinated_per_hundred'
]

# Filter for complete data
filtered_data = data_with_target.dropna(subset=[predictor.target_variable_name] + features)
print(f"Data ready for model training: {filtered_data.shape} rows")

# Train and evaluate the model
print("Training fatigue detection model...")
predictor.train_model(
    data_with_target,
    features=features,
    target=predictor.target_variable_name
)

# Print evaluation results if available
if hasattr(predictor, 'evaluation_results'):
    evaluation = predictor.evaluation_results
    print("\nFinal fatigue detection model performance:")
    print(f"Balanced Accuracy: {evaluation.get('balanced_accuracy', 'N/A')}")
    print(f"ROC AUC Score: {evaluation.get('roc_auc', 'N/A')}")
    print(f"F1 Score: {evaluation.get('f1', 'N/A')}")
else:
    print("No evaluation results found.")

print("\nAnalysis complete. Detailed results saved to:", run_dir)
