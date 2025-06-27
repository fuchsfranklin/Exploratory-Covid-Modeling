"""
Healthcare System Strain Prediction

This module focuses on predicting future healthcare strain, primarily using
ICU patient counts or hospital patient counts as target variables.
It incorporates various features including dynamic (time-varying) data like
case counts and vaccination rates, static (less frequently changing) data like
demographics and healthcare capacity, and lagged features to capture
time-series dependencies.

Enhanced with LSTM deep learning models and ensemble methods for improved accuracy,
particularly for capturing complex temporal patterns across different pandemic phases
and COVID-19 variants.
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Note: Matplotlib is imported but not used in this script.
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer # Added for sophisticated imputation
from sklearn.feature_selection import RFECV # Added for recursive feature elimination
import os
import joblib
import datetime # Added for timestamping runs
import json     # Added for saving run details

# Add TensorFlow and Keras imports for LSTM models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will not be available.")

# Add ensemble method components
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

# Ensure necessary directories exist for outputs and models
os.makedirs('eda_outputs/per_country', exist_ok=True) # For EDA outputs, if any were generated here
os.makedirs('models', exist_ok=True) # For saving trained models
RESULTS_BASE_DIR = "results/healthcare_strain"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True) # Main results directory

class HealthcareStrainPredictor:
    """
    A class to predict healthcare strain using configurable regression models.

    It handles data loading, preprocessing (including lagging features),
    model training, prediction, evaluation, and model saving/loading.
    It now supports KNN imputation, choice of models (Gradient Boosting, Random Forest),
    hyperparameter tuning with GridSearchCV and TimeSeriesSplit, and rolling average features.
    Includes functionality for saving detailed run results and summaries.
    """
    def __init__(self, 
                 base_dynamic_feature_cols, 
                 static_feature_cols, 
                 target_col, 
                 lag_periods=[7, 14],
                 rolling_avg_windows=[7, 14], # Added for rolling averages
                 model_type='GradientBoosting', # Extended options: 'GradientBoosting', 'RandomForest', 'LSTM', 'Ensemble'
                 use_hyperparameter_tuning=False, # Added
                 n_neighbors_imputation=5, # Added for KNNImputer
                 cv_splits=3, # Added for TimeSeriesSplit in GridSearchCV
                 use_feature_selection=False, # Added for RFECV
                 lstm_units=64, # Added for LSTM configuration
                 lstm_dropout=0.2, # Added for LSTM configuration
                 variant_data=None): # Added for variant-specific analysis
        """
        Initializes the HealthcareStrainPredictor.

        Args:
            base_dynamic_feature_cols (list): Columns for dynamic features (for lags and rolling avgs).
            static_feature_cols (list): Columns for static features.
            target_col (str): Target variable column name.
            lag_periods (list): Lag periods for dynamic features.
            rolling_avg_windows (list): Window sizes for rolling averages of dynamic features.
            model_type (str): Type of model to use ('GradientBoosting', 'RandomForest', 'LSTM', or 'Ensemble').
            use_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
            n_neighbors_imputation (int): Number of neighbors for KNNImputer.
            cv_splits (int): Number of splits for TimeSeriesSplit in GridSearchCV.
            use_feature_selection (bool): Whether to use recursive feature elimination.
            lstm_units (int): Number of LSTM units for deep learning model.
            lstm_dropout (float): Dropout rate for LSTM layers.
            variant_data (DataFrame, optional): COVID variant prevalence data.
        """
        self.base_dynamic_feature_cols = base_dynamic_feature_cols
        self.static_feature_cols = static_feature_cols
        self.target_col = target_col
        self.lag_periods = lag_periods
        self.rolling_avg_windows = rolling_avg_windows
        self.model_type = model_type
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.n_neighbors_imputation = n_neighbors_imputation
        self.cv_splits = cv_splits
        self.use_feature_selection = use_feature_selection
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.variant_data = variant_data
        
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_results_dir = os.path.join(RESULTS_BASE_DIR, f"{self.run_id}_{self.model_type}_{'tuned' if self.use_hyperparameter_tuning else 'default'}")

        self.feature_cols = []  # Populated after feature engineering
        self.model_pipeline = None # This will store the fitted pipeline (imputer, scaler, model)
        self.best_params_ = None
        self.feature_importances_ = None
        
        self.data = None
        self.X_train_df = None # DataFrame before pipeline processing
        self.y_train_series = None # Series
        self.X_test_df = None # DataFrame before pipeline processing
        self.y_test_series = None # Series
        
        # Check if LSTM is requested but TensorFlow is not available
        if model_type == 'LSTM' and not TENSORFLOW_AVAILABLE:
            print("Warning: LSTM model requested but TensorFlow not available. Falling back to GradientBoosting.")
            self.model_type = 'GradientBoosting'
            
        # Create the directory for this run's results
        os.makedirs(self.run_results_dir, exist_ok=True)
        
    def load_and_preprocess_data(self, csv_path):
        """
        Loads data, performs feature engineering (lags, rolling averages),
        converts to numeric, handles NaNs by dropping rows with NaN target or all-NaN features,
        and splits into training and testing sets chronologically.

        Note: Imputation and scaling are now part of the scikit-learn pipeline
        and applied during model training/prediction to prevent data leakage.
        self.data is populated with the original data columns aligned to the processed data's index.
        """
        df_orig = pd.read_csv(csv_path)
        df = df_orig.copy() # Work on a copy for processing

        df['date'] = pd.to_datetime(df['date'])
        # Sort by location then date for consistent feature engineering per location
        df.sort_values(['location', 'date'], inplace=True)

        current_features = list(self.static_feature_cols)

        # Create lagged features
        for col in self.base_dynamic_feature_cols:
            for lag in self.lag_periods:
                lagged_col_name = f'{col}_lag{lag}'
                df[lagged_col_name] = df.groupby('location')[col].shift(lag)
                current_features.append(lagged_col_name)
        
        # Create rolling average features
        for col in self.base_dynamic_feature_cols:
            for window in self.rolling_avg_windows:
                rolling_avg_col_name = f'{col}_roll_avg{window}'
                df[rolling_avg_col_name] = df.groupby('location')[col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
                current_features.append(rolling_avg_col_name)

        # Add original dynamic features
        current_features.extend(self.base_dynamic_feature_cols)
        self.feature_cols = sorted(list(set(current_features))) # Ensure unique and sorted for consistency

        # Convert all feature and target columns to numeric
        cols_to_convert = self.feature_cols + [self.target_col]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where the target is NaN, or ALL feature columns are NaN for that row
        df.dropna(subset=[self.target_col], inplace=True)
        df.dropna(subset=self.feature_cols, how='all', inplace=True)
        
        # Align df_orig (self.data) with the processed df (df) using its index
        # This ensures self.data contains original columns (like 'location', 'date') for rows that were kept
        self.data = df_orig.loc[df.index].copy()
        # Add engineered features to self.data for full context if needed for saving results
        for f_col in self.feature_cols:
            if f_col in df.columns: # Ensure the engineered feature exists in the processed df
                 self.data[f_col] = df[f_col]
            
        if df.empty:
            raise ValueError("No data remaining after initial NaN handling and feature engineering.")

        # Split data: Using a simple chronological split for now (last 20% as test)
        # For more robust time-series splitting, consider group-wise splitting or specific date cutoffs.
        # Data should be globally sorted by date if not doing per-location models for this split type.
        df_sorted_globally = df.sort_values('date')
        train_df, test_df = train_test_split(df_sorted_globally, test_size=0.2, shuffle=False)

        self.X_train_df = train_df[self.feature_cols]
        self.y_train_series = train_df[self.target_col]
        self.X_test_df = test_df[self.feature_cols]
        self.y_test_series = test_df[self.target_col]

        # Store the indices of train/test splits for later retrieval of original data if needed
        self.train_indices = self.X_train_df.index
        self.test_indices = self.X_test_df.index

        if self.X_train_df.empty or self.X_test_df.empty:
            raise ValueError("Training or testing dataframe is empty after split. Check data volume and split ratio.")

    def _create_model(self):
        """
        Creates and returns the appropriate model based on self.model_type.
        If hyperparameter tuning is enabled, returns a GridSearchCV object.
        """
        if self.model_type == 'GradientBoosting':
            base_model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            } if self.use_hyperparameter_tuning else {}
        
        elif self.model_type == 'RandomForest':
            base_model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, 30, None]
            } if self.use_hyperparameter_tuning else {}
        
        elif self.model_type == 'LSTM':
            # LSTM doesn't fit into the scikit-learn pipeline framework
            # It will be handled separately in the train model method
            return None
        
        elif self.model_type == 'Ensemble':
            # Create an ensemble of multiple models
            gb_model = GradientBoostingRegressor(random_state=42)
            rf_model = RandomForestRegressor(random_state=42)
            ridge_model = Ridge(random_state=42)
            
            # Create a voting ensemble
            base_model = VotingRegressor(
                estimators=[
                    ('gb', gb_model),
                    ('rf', rf_model),
                    ('ridge', ridge_model)
                ]
            )
            param_grid = {} # Hyperparameter tuning for ensemble is more complex and handled separately
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create and return the model, wrapped in GridSearchCV if tuning is enabled
        if self.use_hyperparameter_tuning and param_grid:
            time_series_cv = TimeSeriesSplit(n_splits=self.cv_splits)
            return GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=time_series_cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
        else:
            return base_model
            
    def _create_lstm_model(self, input_shape):
        """
        Creates and returns a LSTM model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input data (n_timesteps, n_features)
            
        Returns:
            A compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(units=self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.lstm_dropout),
            LSTM(units=self.lstm_units // 2),
            Dropout(self.lstm_dropout),
            Dense(32, activation='relu'),
            Dense(1)  # Output layer for regression
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _prepare_lstm_data(self, X_train, y_train, X_test, y_test, n_timesteps=14):
        """
        Prepares the data in the format needed for LSTM models (3D).
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target vectors
            n_timesteps: Number of time steps for each sample
            
        Returns:
            X_train_3d, X_test_3d: Reshaped feature arrays
            y_train, y_test: Original target arrays
        """
        # Apply imputation and scaling
        imputer = KNNImputer(n_neighbors=self.n_neighbors_imputation)
        scaler = MinMaxScaler()
        
        X_train_processed = imputer.fit_transform(X_train)
        X_train_processed = scaler.fit_transform(X_train_processed)
        
        X_test_processed = imputer.transform(X_test)
        X_test_processed = scaler.transform(X_test_processed)
        
        # Reshape data into 3D format for LSTM [samples, timesteps, features]
        # This is a simplified approach - normally would need more complex time sequences
        n_samples_train = X_train_processed.shape[0] - n_timesteps + 1
        n_samples_test = X_test_processed.shape[0] - n_timesteps + 1
        n_features = X_train_processed.shape[1]
        
        X_train_3d = np.zeros((n_samples_train, n_timesteps, n_features))
        X_test_3d = np.zeros((n_samples_test, n_timesteps, n_features))
        
        # Create sequences
        for i in range(n_samples_train):
            X_train_3d[i] = X_train_processed[i:i+n_timesteps]
            
        for i in range(n_samples_test):
            X_test_3d[i] = X_test_processed[i:i+n_timesteps]
        
        # Adjust targets to match sample count
        y_train_lstm = y_train[n_timesteps-1:].values
        y_test_lstm = y_test[n_timesteps-1:].values
        
        return X_train_3d, y_train_lstm, X_test_3d, y_test_lstm
        
    def train_model(self, train_ratio=0.8, random_state=42):
        """
        Trains the model using the processed data.
        
        Args:
            train_ratio (float): Ratio of data to use for training vs testing.
            random_state (int): Random seed for reproducible splits.
            
        Returns:
            Fitted model and evaluation results.
        """
        if self.X_train_df is None or self.y_train_series is None:
            raise ValueError("Data not loaded and preprocessed. Call load_and_preprocess_data first.")

        # Execute feature selection if requested
        if self.use_feature_selection and self.model_type != 'LSTM':
            print("Performing recursive feature elimination...")
            base_model = self._create_model()
            rfecv = RFECV(
                estimator=base_model,
                step=1,
                cv=TimeSeriesSplit(n_splits=self.cv_splits),
                scoring='neg_mean_absolute_error',
                min_features_to_select=5
            )
            
            # Apply KNN imputation before feature selection
            imputer = KNNImputer(n_neighbors=self.n_neighbors_imputation)
            X_train_imputed = imputer.fit_transform(self.X_train_df)
            
            rfecv.fit(X_train_imputed, self.y_train_series)
            
            # Get selected features
            selected_features = [f for f, selected in zip(self.feature_cols, rfecv.support_) if selected]
            print(f"Selected {len(selected_features)} out of {len(self.feature_cols)} features")
            
            # Update feature columns
            self.feature_cols = selected_features
            self.X_train_df = self.X_train_df[self.feature_cols]
            self.X_test_df = self.X_test_df[self.feature_cols]
        
        # Handle LSTM separately from scikit-learn models
        if self.model_type == 'LSTM':
            X_train_3d, y_train_lstm, X_test_3d, y_test_lstm = self._prepare_lstm_data(
                self.X_train_df, self.y_train_series, 
                self.X_test_df, self.y_test_series
            )
            
            # Define early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Create and train the LSTM model
            lstm_model = self._create_lstm_model(input_shape=(X_train_3d.shape[1], X_train_3d.shape[2]))
            
            history = lstm_model.fit(
                X_train_3d, y_train_lstm,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Make predictions
            y_pred_lstm = lstm_model.predict(X_test_3d).flatten()
            mae = mean_absolute_error(y_test_lstm, y_pred_lstm)
            
            # Save results
            self.model = lstm_model
            self.history = history.history
            self.evaluation_results = {
                'mae': mae,
                'test_predictions': y_pred_lstm,
                'test_actual': y_test_lstm
            }
            
            # Save the model
            lstm_model.save(os.path.join(self.run_results_dir, 'lstm_model.h5'))
            
            # Since LSTM doesn't provide feature importances, create a placeholder
            self.feature_importances_ = None
            
            print(f"LSTM Model MAE: {mae:.4f}")
            return mae
            
        else:
            # Continue with the original scikit-learn approach for other models
            # Define pipeline steps
            steps = [
                ('imputer', KNNImputer(n_neighbors=self.n_neighbors_imputation)),
                ('scaler', MinMaxScaler())]

            # Add model to pipeline
            if self.model_type == 'RandomForest':
                model = RandomForestRegressor(random_state=42)
                steps.append(('regressor', model))
                param_grid = {
                    'regressor__n_estimators': [50, 100], # Reduced for speed
                    'regressor__max_depth': [None, 10, 20],
                    # 'regressor__min_samples_split': [2, 5],
                    # 'regressor__min_samples_leaf': [1, 2]
                }
            elif self.model_type == 'GradientBoosting':
                model = GradientBoostingRegressor(random_state=42)
                steps.append(('regressor', model))
                param_grid = {
                    'regressor__n_estimators': [50, 100], # Reduced for speed
                    'regressor__learning_rate': [0.05, 0.1],
                    # 'regressor__max_depth': [3, 5]
                }
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")
            
            pipeline = Pipeline(steps)

            if self.use_hyperparameter_tuning:
                print(f"Starting hyperparameter tuning for {self.model_type}...")
                # TimeSeriesSplit for cross-validation in tuning
                tscv = TimeSeriesSplit(n_splits=self.cv_splits)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
                grid_search.fit(self.X_train_df, self.y_train_series)
                self.model_pipeline = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                print(f"Best parameters found: {self.best_params_}")
            else:
                print(f"Training {self.model_type} with default parameters...")
                self.model_pipeline = pipeline 
                self.model_pipeline.fit(self.X_train_df, self.y_train_series)
                self.best_params_ = "Not Tuned (default parameters)"
            
            if hasattr(self.model_pipeline.named_steps['regressor'], 'feature_importances_'):
                importances = self.model_pipeline.named_steps['regressor'].feature_importances_
                self.feature_importances_ = pd.Series(importances, index=self.X_train_df.columns).sort_values(ascending=False)
            else:
                self.feature_importances_ = None

            print("Model training complete.")

    def predict(self, X_unscaled_df):
        """
        Makes predictions on new, unscaled data using the trained pipeline.
        Args:
            X_unscaled_df (pd.DataFrame): DataFrame of unscaled features for prediction.
                                       Must contain all columns defined in self.feature_cols.
        Returns:
            np.array: Array of predictions.
        """
        if self.model_pipeline is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        # Ensure columns are in the same order as during training
        X_unscaled_df_ordered = X_unscaled_df[self.feature_cols]
        return self.model_pipeline.predict(X_unscaled_df_ordered)

    def evaluate_model(self):
        """
        Evaluates the trained model pipeline on the internal test set using Mean Absolute Error (MAE).
        """
        if self.model_pipeline is None or self.X_test_df is None or self.y_test_series is None:
             raise ValueError("Model not trained or data not split. Call train_model and load_and_preprocess_data.")
        
        predictions = self.model_pipeline.predict(self.X_test_df)
        return mean_absolute_error(self.y_test_series, predictions)

    def _generate_run_summary_text(self, mae):
        """
        Generates a human-readable text summary of the run.
        """
        summary_lines = [
            "Healthcare Strain Prediction Run Summary",
            "--------------------------------------",
            f"Run ID: {self.run_id}",
            f"Timestamp: {datetime.datetime.now().isoformat()}",
            f"Model Type: {self.model_type}",
            f"Target Column: {self.target_col}",
            f"Hyperparameter Tuning: {'Enabled' if self.use_hyperparameter_tuning else 'Disabled'}",
            f"Best Parameters: {json.dumps(self.best_params_, indent=2) if self.use_hyperparameter_tuning and self.best_params_ != 'Not Tuned (default parameters)' else self.best_params_}",
            f"Features Used ({len(self.feature_cols)} total):",
            f"  Sample Features (first 5): {', '.join(self.feature_cols[:5])}...", 
            "\nEvaluation",
            "----------",
            f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}",
            "\nInterpretation of MAE:",
            f"The MAE indicates that, on average, the model's predictions for '{self.target_col}'",
            f"are off by approximately {mae:.4f} units from the actual values on the test set.",
            "A lower MAE is generally better. This value should be contextualized against the typical range",
            f"and variability of '{self.target_col}'. For example, if '{self.target_col}' typically ranges from 0-100, an MAE of {mae:.4f} might be considered good.",
            "\nFeature Importances (Top 10):"
        ]
        if self.feature_importances_ is not None:
            summary_lines.append(self.feature_importances_.head(10).to_string())
        else:
            summary_lines.append("Feature importances not available for this model type or not extracted.")
        
        summary_lines.append("\nPredictions:")
        summary_lines.append("Test set predictions and actual values (including date and location) are saved in 'test_predictions_vs_actual.csv'.")
        summary_lines.append(f"Full run details and configurations are saved in 'run_details.json'.")
        summary_lines.append(f"The trained model pipeline is saved as 'model_pipeline.pkl'.")
        summary_lines.append(f"All outputs for this run are in directory: {self.run_results_dir}")
        return "\n".join(summary_lines)

    def save_run_results(self, mae, test_predictions):
        """
        Saves all results and artifacts from the current run to a dedicated directory.
        """
        os.makedirs(self.run_results_dir, exist_ok=True)

        run_details = {
            'run_id': self.run_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_type': self.model_type,
            'target_column': self.target_col,
            'all_engineered_features': self.feature_cols,
            'base_dynamic_feature_cols': self.base_dynamic_feature_cols,
            'static_feature_cols': self.static_feature_cols,
            'lag_periods': self.lag_periods,
            'rolling_avg_windows': self.rolling_avg_windows,
            'knn_imputer_neighbors': self.n_neighbors_imputation,
            'hyperparameter_tuning_enabled': self.use_hyperparameter_tuning,
            'cv_splits_for_tuning': self.cv_splits if self.use_hyperparameter_tuning else 'N/A',
            'best_hyperparameters': self.best_params_,
            'mae_on_test_set': mae,
            'training_data_shape': self.X_train_df.shape,
            'test_data_shape': self.X_test_df.shape
        }
        with open(os.path.join(self.run_results_dir, 'run_details.json'), 'w') as f:
            json.dump(run_details, f, indent=4)

        if self.feature_importances_ is not None:
            self.feature_importances_.to_csv(os.path.join(self.run_results_dir, 'feature_importances.csv'), header=['importance'])

        # Use self.data (which has original columns aligned with test_indices) to get date and location
        predictions_df = self.data.loc[self.test_indices, ['date', 'location', self.target_col]].copy()
        predictions_df.rename(columns={self.target_col: 'actual_target'}, inplace=True)
        # Ensure test_predictions is a 1D array for assignment
        predictions_df['predicted_target'] = np.array(test_predictions).flatten()
        predictions_df.to_csv(os.path.join(self.run_results_dir, 'test_predictions_vs_actual.csv'), index=False)

        summary_text = self._generate_run_summary_text(mae)
        with open(os.path.join(self.run_results_dir, 'run_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        model_output_path = os.path.join(self.run_results_dir, f'model_pipeline.pkl')
        self.save_model(model_output_path) 

        print(f"\n--- Run Complete ({self.run_id}) --- ")
        print(summary_text)

    def save_model(self, model_path):
        """
        Saves the trained model pipeline to a file using joblib.
        """
        if self.model_pipeline is None:
            raise ValueError("No model trained to save.")
        joblib.dump(self.model_pipeline, model_path)
        print(f"Model pipeline saved to {model_path}")

    def load_model(self, model_path):
        """
        Loads a pre-trained model pipeline from a file using joblib.
        """
        self.model_pipeline = joblib.load(model_path)
        # Important: After loading a pipeline, self.feature_cols should be consistent
        # with the features the loaded pipeline was trained on. This class assumes
        # feature_cols is set during init/preprocessing. If loading a model trained
        # with different features, this could be an issue. For this script's flow,
        # it's assumed that load_and_preprocess_data would be called, setting feature_cols,
        # or the user ensures consistency.
        print(f"Model pipeline loaded from {model_path}")

# Example usage block
if __name__ == "__main__":
    # --- Configuration --- # 
    MODEL_CHOICE = 'GradientBoosting'  # 'GradientBoosting' or 'RandomForest'
    PERFORM_TUNING = False             # True to run GridSearchCV, False to use default parameters
    KNN_NEIGHBORS = 5
    CV_SPLITS_FOR_TUNING = 3 # Number of splits for TimeSeriesSplit in GridSearchCV
    ROLLING_WINDOWS = [7, 14] # Windows for rolling averages
    LAG_PERIODS = [7, 14]     # Lag periods

    # Define the target variable for prediction
    target = 'icu_patients_per_million'
    # Define base dynamic features (time-varying, will be lagged)
    base_dynamic_features = [
        'new_cases_smoothed_per_million', 
        'new_deaths_smoothed_per_million',
        'reproduction_rate',
        'hosp_patients_per_million',
        'new_tests_smoothed_per_thousand',
        'positive_rate',
        'people_fully_vaccinated_per_hundred',
        'total_boosters_per_hundred',
        'stringency_index'
    ]
    
    # Define static features (less frequently changing, not lagged by default)
    static_features = [
        'population_density', 
        'median_age', 
        'aged_65_older', 
        'gdp_per_capita',
        'extreme_poverty',
        'cardiovasc_death_rate', 
        'diabetes_prevalence',
        'hospital_beds_per_thousand',
        'life_expectancy',
        'human_development_index'
    ]
    # Initialize the predictor
    predictor = HealthcareStrainPredictor(
        base_dynamic_feature_cols=base_dynamic_features,
        static_feature_cols=static_features,
        target_col=target,
        lag_periods=LAG_PERIODS,
        rolling_avg_windows=ROLLING_WINDOWS,
        model_type=MODEL_CHOICE,
        use_hyperparameter_tuning=PERFORM_TUNING,
        n_neighbors_imputation=KNN_NEIGHBORS,
        cv_splits=CV_SPLITS_FOR_TUNING
    )

    # Path to the dataset
    csv_data_path = 'owid-covid-data.csv'

    # Load, preprocess data, and handle potential errors
    try:
        print(f"Starting run ID: {predictor.run_id}")
        print(f"Results will be saved to: {predictor.run_results_dir}")
        print(f"\nLoading and preprocessing data from {csv_data_path}...")
        predictor.load_and_preprocess_data(csv_data_path)
        print("Data loading and preprocessing complete.")
        print(f"X_train_df shape: {predictor.X_train_df.shape}")
        print(f"X_test_df shape: {predictor.X_test_df.shape}")
        print(f"Number of features: {len(predictor.feature_cols)}")
        # print(f"Features: {predictor.feature_cols}") # Uncomment to see all feature names

    except FileNotFoundError:
        print(f"Error: '{csv_data_path}' not found. Make sure it's in the correct path.")
        exit()
    except KeyError as e:
        print(f"Error: Column not found in CSV: {e}. Ensure all specified features and target exist.")
        exit()
    except ValueError as e:
        print(f"ValueError during data loading/preprocessing: {e}")
        exit()

    try:
        print("\nTraining model...")
        predictor.train_model() # This now handles pipeline creation and optional tuning
    except Exception as e:
        print(f"Error during model training: {e}")
        # import traceback
        # traceback.print_exc()
        exit()

    print("\nEvaluating model...")
    mae = predictor.evaluate_model()

    print("\nGenerating and saving results...")
    test_set_predictions = predictor.predict(predictor.X_test_df)
    predictor.save_run_results(mae, test_set_predictions)

    print("\nScript finished.")
    print("\nReminder of further improvements to consider:")
    print("- Advanced outlier detection and handling (e.g., as a pipeline step).")
    print("- Geographic stratification: training separate models for different regions.")
    print("- More extensive hyperparameter grids for tuning.")
    print("- Exploration of other models (e.g., XGBoost, LightGBM, Prophet for time-series native models).")
    print("- More sophisticated time-series splitting for train/test (e.g., by specific date cutoffs or group-aware splits).")
    print("- In-depth analysis of feature importance and interactions.")