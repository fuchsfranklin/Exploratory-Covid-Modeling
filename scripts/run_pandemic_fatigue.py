"""
Run the Pandemic Fatigue Analysis with enhanced features
including sentiment analysis integration and advanced classification methods.
"""

from pandemic_fatigue import PandemicFatiguePredictor
import os
import sys

# Make sure the script can find the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define fatigue parameters
fatigue_def_params = {
    'stringency_threshold': 60,  # High stringency threshold
    'case_increase_window': 14,  # Window to detect case increases
    'case_increase_threshold': 0.2,  # 20% increase in cases
}

# Initialize the predictor with enhanced features
predictor = PandemicFatiguePredictor(
    data_path='../owid-covid-data.csv',
    model_type='GradientBoosting',  # Using gradient boosting for enhanced classification
    tune_hyperparameters=True,  # Enable hyperparameter tuning
    fatigue_def_params=fatigue_def_params,
    use_sentiment_analysis=True,  # Enable sentiment analysis integration
    sentiment_model='vader',  # Using VADER for sentiment analysis
    use_mobility_data=True  # Enable mobility data integration
)

# Load and preprocess the data
print("Loading and preprocessing data...")
predictor.load_and_preprocess_data()
print(f"Loaded data shape: {predictor.data.shape}")

# Define features
features = [
    'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
    'reproduction_rate', 'stringency_index', 'population_density',
    'median_age', 'gdp_per_capita', 'human_development_index',
    'positive_rate', 'people_fully_vaccinated_per_hundred'
]

# Define time-lag features to generate
lag_periods = [7, 14, 21]
lag_features = [
    'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
    'reproduction_rate', 'stringency_index'
]

print("Engineering features...")
engineered_data = predictor.engineer_features(
    predictor.data, 
    features=features,
    lag_periods=lag_periods,
    lag_features=lag_features
)
print(f"Engineered data shape: {engineered_data.shape}")

# Create fatigue indicator
print("Creating fatigue indicator...")
data_with_target = predictor.create_fatigue_indicator(engineered_data)
print(f"Data with fatigue indicator shape: {data_with_target.shape}")

# Define countries to include in the analysis
countries_to_analyze = [
    'United States', 'Germany', 'United Kingdom', 'France',
    'Italy', 'Spain', 'Canada', 'Japan', 'South Korea', 'Brazil'
]

# Train model for fatigue detection
print("Training fatigue detection model...")
evaluation = predictor.train_and_evaluate_model(
    data_with_target,
    features=features + predictor.get_lag_feature_names(lag_features, lag_periods),
    countries=countries_to_analyze
)

# Print final evaluation results
print("\nFinal fatigue detection model performance:")
print(f"Balanced Accuracy: {evaluation.get('balanced_accuracy', 'N/A')}")
print(f"ROC AUC Score: {evaluation.get('roc_auc', 'N/A')}")
print(f"F1 Score: {evaluation.get('f1', 'N/A')}")
print("\nDetailed performance saved in results directory.")
print(f"Results directory: {predictor.run_dir}")
