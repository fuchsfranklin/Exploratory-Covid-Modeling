"""
Healthcare System Strain Prediction

This module focuses on predicting future healthcare strain, primarily using
ICU patient counts or hospital patient counts as target variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib

os.makedirs('eda_outputs/per_country', exist_ok=True)
os.makedirs('models', exist_ok=True)

class HealthcareStrainPredictor:
    def __init__(self, feature_cols, target_col):
        # Store column names
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = GradientBoostingRegressor(random_state=42) # Added random_state for reproducibility
        self.scaler = MinMaxScaler()
        self.data = None # Will be loaded as DataFrame
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_scaled = None
        self.X_test_scaled = None

    def load_and_preprocess_data(self, csv_path):
        # Load data using pandas
        df = pd.read_csv(csv_path)

        # Select relevant columns
        all_cols = self.feature_cols + [self.target_col]
        self.data = df[all_cols].copy()

        # Basic preprocessing: Fill missing values with 0
        # More sophisticated imputation might be needed for better results
        self.data.fillna(0, inplace=True)

        # Ensure data is numeric (errors='coerce' turns non-numeric into NaN, then fillna handles them)
        for col in all_cols:
             self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data.fillna(0, inplace=True)

        # Separate features (X) and target (y)
        X = self.data[self.feature_cols]
        y = self.data[self.target_col]

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features (fit on training data, transform both train and test)
        self.X_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        # Train the model using scaled training data
        if self.X_scaled is None or self.y_train is None:
            raise ValueError("Data not preprocessed or split. Call load_and_preprocess_data first.")
        self.model.fit(self.X_scaled, self.y_train)

    def predict(self, X):
        # Implement prediction steps
        # Assume input X is unscaled, apply the fitted scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate_model(self):
        # Evaluate the model on the scaled test set
        if self.X_test_scaled is None or self.y_test is None:
             raise ValueError("Data not preprocessed or split. Call load_and_preprocess_data first.")
        predictions = self.model.predict(self.X_test_scaled) # Use the internal model
        return mean_absolute_error(self.y_test, predictions)

    def save_model(self, model_path):
        # Save the trained model to a file
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        # Load a trained model from a file
        self.model = joblib.load(model_path)

# Example usage
if __name__ == "__main__":
    # Define features and target
    # Example: Predict ICU patients based on new cases and deaths (smoothed, per million)
    features = ['new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million']
    target = 'icu_patients_per_million' # Example target

    # Initialize the predictor
    predictor = HealthcareStrainPredictor(feature_cols=features, target_col=target)

    # Load and preprocess the data (including splitting)
    try:
        predictor.load_and_preprocess_data('owid-covid-data.csv')
    except KeyError as e:
        print(f"Error: Column not found in CSV: {e}")
        print("Please ensure 'owid-covid-data.csv' contains the required columns.")
        exit() # Exit if columns are missing
    except FileNotFoundError:
        print("Error: 'owid-covid-data.csv' not found.")
        exit()

    # Train the model
    predictor.train_model()

    # Evaluate the model (uses internal test set)
    mae = predictor.evaluate_model()
    print(f"Mean Absolute Error on Test Set: {mae}")

    # Save the model
    predictor.save_model('models/healthcare_strain_predictor.pkl')

    # Load the model (optional, for demonstration)
    # predictor.load_model('models/healthcare_strain_predictor.pkl')

    # Make predictions on the test set (using the unscaled test features)
    # Note: predict method handles scaling internally
    predictions = predictor.predict(predictor.X_test)
    print("Sample Predictions on Test Set:")
    print(predictions[:10]) # Print first 10 predictions
    print("Actual Values on Test Set:")
    print(predictor.y_test[:10].values) # Print first 10 actual values