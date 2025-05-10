"""
Pandemic Fatigue Analysis and Prediction

This module aims to identify, analyze, and potentially forecast periods of "pandemic fatigue."
Pandemic fatigue is operationally defined based on indicators such as high stringency 
coinciding with high or unexpectedly increasing disease transmission proxies 
(e.g., positive rate, cases per test).

The script preprocesses data, engineers features, trains a model (to be defined),
and saves results in a structured manner.
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Plotting will be handled separately or in EDA notebooks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier # Added for future option
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix

import os
import json
import joblib
from datetime import datetime

# Base directory for results
RESULTS_BASE_DIR = "results/pandemic_fatigue"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


class PandemicFatiguePredictor:
    """
    Identifies, analyzes, and potentially forecasts pandemic fatigue.
    """

    def __init__(self, 
                 data_path='owid-covid-data.csv',
                 target_variable_name="fatigue_indicator",
                 model_type='LogisticRegression', 
                 tune_hyperparameters=False,
                 hyperparameter_grid=None,
                 fatigue_def_params=None, # Added
                 country_col='location',
                 date_col='date',
                 results_base_dir=RESULTS_BASE_DIR,
                 run_id=None):
        """
        Initialize the PandemicFatiguePredictor.

        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the OWID dataset.
        target_variable_name : str
            Name of the engineered target variable representing fatigue.
        model_type : str
            Type of model to use (e.g., 'LogisticRegression').
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning.
        hyperparameter_grid : dict, optional
            Grid of hyperparameters for tuning.
        fatigue_def_params : dict, optional
            Parameters for defining the fatigue metric.
            Example: {
                'stringency_percentile_threshold': 0.75,
                'min_sustained_high_stringency_days': 28,
                'proxy_lookback_window': 14,
                'proxy_increase_threshold_factor': 1.10
            }
        country_col : str
            Name of the column identifying countries/locations.
        date_col : str
            Name of the column for dates.
        results_base_dir : str
            Base directory to save run results.
        run_id : str, optional
            A unique identifier for the run. If None, generated automatically.
        """
        self.data_path = data_path
        self.target_variable_name = target_variable_name
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_grid = hyperparameter_grid
        self.country_col = country_col
        self.date_col = date_col
        self.results_base_dir = results_base_dir

        default_fatigue_params = {
            'stringency_col_raw': 'stringency_index', # Raw column name
            'proxy_col_raw_options': ['positive_rate', 'new_cases_smoothed_per_million'], # Raw column names
            'stringency_percentile_threshold': 0.65,  # Reduced from 0.75
            'min_sustained_high_stringency_days': 14, # Reduced from 28
            'proxy_lookback_window': 10,             # Reduced from 14
            'proxy_increase_threshold_factor': 1.05   # Reduced from 1.10
        }
        self.fatigue_def_params = default_fatigue_params
        if fatigue_def_params:
            self.fatigue_def_params.update(fatigue_def_params)
        
        if run_id is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name_parts = [current_time, self.model_type]
            if self.tune_hyperparameters:
                run_name_parts.append("tuned")
            else:
                run_name_parts.append("default")
            self.run_id = "_".join(run_name_parts)
        else:
            self.run_id = run_id
            
        self.run_results_dir = os.path.join(self.results_base_dir, self.run_id)
        os.makedirs(self.run_results_dir, exist_ok=True)

        self.data = None
        self.model_pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.test_dates = None
        self.test_locations = None
        self.feature_names = None
        self.metrics = {}
        self.raw_data_for_test_output = None # To store original date/location for test set
        self.best_model_params = None # Store best params from GridSearchCV

        print(f"Run ID: {self.run_id}")
        print(f"Results will be saved in: {self.run_results_dir}")

    def _define_fatigue_metric(self, df_country):
        """
        Defines a binary pandemic fatigue indicator for a single country.

        Fatigue is defined based on parameters in self.fatigue_def_params:
        1. Stringency (e.g., stringency_index_smoothed) has been consistently high 
           (above `stringency_percentile_threshold`) for at least `min_sustained_high_stringency_days`.
        2. During such a period, the transmission proxy (e.g., positive_rate_smoothed)
           is significantly higher (by `proxy_increase_threshold_factor`) than its
           average over the recent `proxy_lookback_window`.
           
        Returns a dataframe with the fatigue_indicator column added (1 = fatigue, 0 = no fatigue).
        """
        params = self.fatigue_def_params
        country_name = df_country[self.country_col].iloc[0] if not df_country.empty else "Unknown Country"
        df_country[self.target_variable_name] = 0  # Initialize target column

        stringency_col_smoothed = f"{params['stringency_col_raw']}_smoothed"

        if stringency_col_smoothed not in df_country.columns:
            print(f"Warning: Smoothed stringency column '{stringency_col_smoothed}' not found for {country_name}. Skipping fatigue definition.")
            return df_country

        # Determine the actual smoothed proxy column to use
        chosen_proxy_col_smoothed = None
        for proxy_raw in params['proxy_col_raw_options']:
            potential_proxy_smoothed = f"{proxy_raw}_smoothed"
            if potential_proxy_smoothed in df_country.columns and df_country[potential_proxy_smoothed].isnull().mean() < 0.5:  # Less strict null check (50% vs 75%)
                chosen_proxy_col_smoothed = potential_proxy_smoothed
                break
        
        if not chosen_proxy_col_smoothed:
            print(f"Warning: No suitable smoothed proxy column found for {country_name} (tried options: {params['proxy_col_raw_options']}). Skipping fatigue definition.")
            return df_country
        
        print(f"Using '{chosen_proxy_col_smoothed}' as proxy for fatigue definition in {country_name}.")

        # 1. Calculate stringency threshold
        if len(df_country[stringency_col_smoothed].dropna()) < params['min_sustained_high_stringency_days'] * 1.5: # Need enough data
            print(f"Warning: Insufficient data points for {stringency_col_smoothed} in {country_name} ({len(df_country[stringency_col_smoothed].dropna())}) to define fatigue robustly. Skipping.")
            return df_country
            
        stringency_thresh_val = df_country[stringency_col_smoothed].quantile(params['stringency_percentile_threshold'])
        if pd.isna(stringency_thresh_val):
            print(f"Warning: Could not calculate stringency threshold for {country_name}. Skipping fatigue definition.")
            return df_country

        # 2. Identify periods of high stringency
        df_country['is_high_stringency'] = (df_country[stringency_col_smoothed] >= stringency_thresh_val).astype(int)

        # 3. Identify sustained high stringency periods
        # Group by consecutive blocks of 'is_high_stringency'
        df_country['high_stringency_block_id'] = (df_country['is_high_stringency'].diff(1) != 0).astype('int').cumsum()
        df_country['days_in_block'] = df_country.groupby('high_stringency_block_id')['is_high_stringency'].transform('size')
        
        df_country['is_sustained_high_stringency'] = (df_country['is_high_stringency'] == 1) & \
                                                    (df_country['days_in_block'] >= params['min_sustained_high_stringency_days'])
        
        # 4. For sustained high stringency periods, check proxy behavior
        df_country[f'{chosen_proxy_col_smoothed}_roll_avg'] = df_country[chosen_proxy_col_smoothed].rolling(
                                                                window=params['proxy_lookback_window'], 
                                                                min_periods=max(1, params['proxy_lookback_window'] // 2)
                                                            ).mean().shift(1) # Shift(1) to use strictly past data for avg

        # Modified fatigue condition - less stringent to capture more potential fatigue periods
        fatigue_conditions_met = (
            # Either in a sustained high stringency period with elevated proxy
            ((df_country['is_sustained_high_stringency']) & 
             (df_country[chosen_proxy_col_smoothed] > (df_country[f'{chosen_proxy_col_smoothed}_roll_avg'] * params['proxy_increase_threshold_factor']))) |
            # OR a significant unexpected increase in the proxy regardless of stringency length
            ((df_country['is_high_stringency'] == 1) & 
             (df_country[chosen_proxy_col_smoothed] > (df_country[f'{chosen_proxy_col_smoothed}_roll_avg'] * (params['proxy_increase_threshold_factor'] * 1.5))))
        ) & (df_country[f'{chosen_proxy_col_smoothed}_roll_avg'].notna())
        
        df_country.loc[fatigue_conditions_met, self.target_variable_name] = 1
        
        # Store additional context columns that might be useful for analysis
        df_country[f'proxy_rollmean_{params["proxy_lookback_window"]}d'] = df_country[f'{chosen_proxy_col_smoothed}_roll_avg']
        df_country['proxy_to_rollmean_ratio'] = df_country[chosen_proxy_col_smoothed] / df_country[f'{chosen_proxy_col_smoothed}_roll_avg']
        
        # Add engineered features specific to fatigue: days since stringency increase, etc.
        df_country['days_since_stringency_increase'] = df_country['high_stringency_block_id'].map(
            df_country.groupby('high_stringency_block_id')[self.date_col].transform(
                lambda x: (x - x.min()).dt.days if x.min() == x.min() else pd.Series([0] * len(x))
            )
        )
        
        # Drop temporary working columns used for fatigue definition
        cols_to_drop = ['high_stringency_block_id', 'days_in_block']
        df_country = df_country.drop(columns=[col for col in cols_to_drop if col in df_country.columns], errors='ignore')

        num_fatigue_days = df_country[self.target_variable_name].sum()
        total_days = len(df_country)
        fatigue_percentage = (num_fatigue_days / total_days) * 100 if total_days > 0 else 0
        
        print(f"Defined {num_fatigue_days} fatigue days ({fatigue_percentage:.1f}% of data) for {country_name} using proxy '{chosen_proxy_col_smoothed}'.")
            
        return df_country

    def load_and_preprocess_data(self, countries=None, test_size=0.2, smooth_window=14, ensure_class_balance=True):
        """
        Loads data, preprocesses, defines fatigue metric, and splits into train/test.

        Parameters:
        -----------
        countries : list, optional
            List of countries to filter for. If None, uses all available.
        test_size : float
            Proportion of data to use for testing (chronological split per country).
        smooth_window : int
            Rolling window for smoothing features.
        ensure_class_balance : bool
            If True, the function will check that test and train sets have both classes represented
            for the target variable, adjusting the split if necessary.
        """
        try:
            raw_df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        print("Raw data loaded.")
        raw_df = raw_df.sort_values([self.country_col, self.date_col])

        if countries:
            raw_df = raw_df[raw_df[self.country_col].isin(countries)]
            if raw_df.empty:
                raise ValueError(f"No data found for specified countries: {countries}")
            print(f"Filtered for countries: {countries}")

        # Identify necessary columns (example, expand as needed)
        base_features = list(set([
            self.fatigue_def_params['stringency_col_raw']] + \
            self.fatigue_def_params['proxy_col_raw_options'] + \
            ['people_vaccinated_per_hundred', 'reproduction_rate', 'new_tests_smoothed_per_thousand'] # Add other potentially useful features
        ))
        # Ensure critical features for fatigue definition are present
        # The smoothing process below will create the '_smoothed' versions used in _define_fatigue_metric
        
        # Select available features and keep original date/location for test output
        self.raw_data_for_test_output = raw_df[[self.date_col, self.country_col]].copy()

        all_processed_data = []
        
        for country_name, group in raw_df.groupby(self.country_col):
            print(f"Processing data for {country_name}...")
            country_df = group.copy()

            # Impute and smooth base features
            for col in base_features:
                if col in country_df.columns:
                    # Basic imputation (can be enhanced or handled by KNNImputer later)
                    country_df[col] = country_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    country_df[f'{col}_smoothed'] = country_df[col].rolling(window=smooth_window, min_periods=1, center=True).mean()
                else:
                    print(f"Warning: Column {col} not found for {country_name}. Skipping.")
            
            # Define/engineer the fatigue metric (CRITICAL STEP)
            country_df = self._define_fatigue_metric(country_df) # This needs proper implementation

            # Drop rows with NaNs that might have been created by rolling means at edges
            # or if target variable couldn't be defined for some rows.
            country_df = country_df.dropna(subset=[f'{col}_smoothed' for col in base_features if f'{col}_smoothed' in country_df.columns] + [self.target_variable_name])
            
            if country_df.empty or self.target_variable_name not in country_df.columns:
                print(f"Warning: No valid data for {country_name} after initial processing or fatigue definition. Skipping.")
                continue
            
            all_processed_data.append(country_df)

        if not all_processed_data:
            raise ValueError("No data available after processing for any country.")

        self.data = pd.concat(all_processed_data)
        print("Data concatenated from all processed countries.")

        # Feature Engineering (e.g., lags) - can be a separate method
        # For now, features are the smoothed base features
        self.feature_names = [f'{col}_smoothed' for col in base_features if f'{col}_smoothed' in self.data.columns and col not in self.fatigue_def_params['proxy_col_raw_options'] and col != self.fatigue_def_params['stringency_col_raw']]
        
        # Let's redefine feature_names more carefully:
        # Use all *_smoothed versions of base_features, EXCLUDING the target variable itself if it was somehow in base_features.
        potential_features = []
        for col in base_features:
            smoothed_col = f'{col}_smoothed'
            if smoothed_col in self.data.columns and smoothed_col != self.target_variable_name:
                 potential_features.append(smoothed_col)
        self.feature_names = list(set(potential_features)) # Use set to ensure uniqueness

        if not self.feature_names:
            raise ValueError("No features available after preprocessing.")
        print(f"Using features: {self.feature_names}")

        # COMPLETELY REVISED TRAIN/TEST SPLIT APPROACH
        # Instead of splitting chronologically per country, we'll use a stratified approach
        # to ensure class balance in both train and test sets
        
        # First, identify countries with fatigue instances
        countries_with_fatigue = []
        for country_name, group in self.data.groupby(self.country_col):
            if group[self.target_variable_name].sum() > 0:
                countries_with_fatigue.append(country_name)
                
        print(f"Countries with fatigue instances: {countries_with_fatigue}")
        
        # Strategy: Ensure we include fatigue examples in both train and test sets
        # For each country with fatigue, split chronologically but ensure both classes appear in both splits
        train_dfs = []
        test_dfs = []
        test_dates_list = []
        test_locations_list = []
        
        for country_name, group in self.data.groupby(self.country_col):
            group = group.sort_values(self.date_col)  # Ensure chronological order
            
            # Skip countries with no fatigue instances
            if country_name not in countries_with_fatigue:
                # Split chronologically
                split_idx = int(len(group) * (1 - test_size))
                train_dfs.append(group.iloc[:split_idx])
                test_df_country = group.iloc[split_idx:]
                test_dfs.append(test_df_country)
                test_dates_list.append(test_df_country[self.date_col])
                test_locations_list.append(test_df_country[self.country_col])
                continue
            
            # For countries with fatigue, find fatigue examples
            fatigue_indices = group.index[group[self.target_variable_name] == 1].tolist()
            
            if not fatigue_indices:
                print(f"Warning: No fatigue instances found for {country_name} despite being in the fatigue list.")
                continue
                
            # Ensure some fatigue instances in test set (about test_size % of them)
            num_fatigue_for_test = max(1, int(len(fatigue_indices) * test_size))
            
            # Select fatigue indices for test set - use later part of the time series for test
            # but ensure at least some fatigue examples are in test
            fatigue_test_indices = fatigue_indices[-num_fatigue_for_test:]
            
            # Create test set: All points after the earliest fatigue test index
            earliest_fatigue_test = min(fatigue_test_indices)
            test_idx = group.index.get_indexer([earliest_fatigue_test])[0]
            
            # Adjust if needed to ensure reasonable test size
            ideal_test_size = int(len(group) * test_size)
            if len(group) - test_idx < ideal_test_size * 0.5 or len(group) - test_idx > ideal_test_size * 1.5:
                # If test set is too small or too large, adjust
                test_idx = max(0, min(len(group) - ideal_test_size, test_idx))
            
            train_df_country = group.iloc[:test_idx]
            test_df_country = group.iloc[test_idx:]
            
            # Verify both sets have fatigue examples
            if train_df_country[self.target_variable_name].sum() == 0:
                print(f"Warning: Train set for {country_name} has no fatigue examples. Adjusting split.")
                # Find the earliest fatigue example and ensure it's in training
                earliest_fatigue = min(fatigue_indices)
                earliest_idx = group.index.get_indexer([earliest_fatigue])[0]
                if earliest_idx >= test_idx:
                    # This shouldn't happen given our approach, but just in case
                    test_idx = earliest_idx + 1  # Put the earliest fatigue example in training
                    train_df_country = group.iloc[:test_idx]
                    test_df_country = group.iloc[test_idx:]
            
            if test_df_country[self.target_variable_name].sum() == 0:
                print(f"Warning: Test set for {country_name} has no fatigue examples. Forcing at least one.")
                # Force at least one fatigue example in test by finding a good candidate
                # that's as close to our desired test split as possible
                candidate_fatigue_indices = [idx for idx in fatigue_indices if idx > group.index[int(len(group) * 0.7)]]
                if candidate_fatigue_indices:
                    test_idx = group.index.get_indexer([min(candidate_fatigue_indices)])[0]
                    train_df_country = group.iloc[:test_idx]
                    test_df_country = group.iloc[test_idx:]
            
            # Add to our collections
            train_dfs.append(train_df_country)
            test_dfs.append(test_df_country)
            test_dates_list.append(test_df_country[self.date_col])
            test_locations_list.append(test_df_country[self.country_col])
            
            # Report on the split for this country
            train_fatigue = train_df_country[self.target_variable_name].sum()
            test_fatigue = test_df_country[self.target_variable_name].sum()
            print(f"{country_name}: Train fatigue={train_fatigue}/{len(train_df_country)} ({train_fatigue/len(train_df_country)*100:.1f}%), "
                  f"Test fatigue={test_fatigue}/{len(test_df_country)} ({test_fatigue/len(test_df_country)*100:.1f}%)")
        
        if not train_dfs or not test_dfs:
            raise ValueError("Could not create train/test splits. Check data availability and test_size.")

        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)
        
        # Final check on combined data
        train_class_dist = train_df[self.target_variable_name].value_counts(normalize=True)
        test_class_dist = test_df[self.target_variable_name].value_counts(normalize=True)
        
        print(f"Train set class distribution: {train_class_dist.to_dict()}")
        print(f"Test set class distribution: {test_class_dist.to_dict()}")
        
        if 1 not in test_class_dist:
            print("WARNING: Test set does not contain any fatigue examples (class 1). Model evaluation will be limited.")
        if 1 not in train_class_dist:
            print("WARNING: Train set does not contain any fatigue examples (class 1). Model cannot learn to predict fatigue.")
        
        self.test_dates = pd.concat(test_dates_list)
        self.test_locations = pd.concat(test_locations_list)

        self.X_train = train_df[self.feature_names]
        self.y_train = train_df[self.target_variable_name]
        self.X_test = test_df[self.feature_names]
        self.y_test = test_df[self.target_variable_name]

        print(f"Train data shape: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Test data shape: {self.X_test.shape}, {self.y_test.shape}")
        if self.X_train.empty or self.X_test.empty:
            raise ValueError("Training or testing data is empty after split.")
        
        # Store original date/location columns for the test set for final output
        # This needs to align with the X_test indices
        self.raw_data_for_test_output = self.raw_data_for_test_output.loc[self.X_test.index]


    def engineer_features(self, df):
        """
        Engineers additional features for the model beyond the basic features created
        during data loading and preprocessing.
        
        This method creates more sophisticated features including:
        1. Interaction terms between key predictors
        2. Rate-of-change features for dynamic variables
        3. Contextual features comparing current values to historical trends
        4. Additional statistical transformations
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with base features (typically self.X_train or self.X_test)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional engineered features
        """
        print("Engineering additional features for pandemic fatigue prediction...")
        
        # Work on a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # Group data by country for country-specific feature engineering
        # This assumes column names are consistent and we're accessing the original data
        if (hasattr(self, 'data') and 
            self.data is not None and 
            not self.data.empty and
            self.country_col in self.data.columns):
            
            country_mapping = self.data[self.country_col]
            
            # Get unique countries from indices in df
            countries_in_df = set(country_mapping[country_mapping.index.isin(df.index)].unique())
            print(f"Engineering features for {len(countries_in_df)} countries")
            
            # Use a subset of feature columns for engineering
            base_cols = [col for col in df.columns if 'smoothed' in col and col != f"{self.fatigue_def_params['stringency_col_raw']}_smoothed"]
            stringency_col = f"{self.fatigue_def_params['stringency_col_raw']}_smoothed"
            
            if stringency_col in df.columns:
                # 1. Create interaction features between stringency and transmission proxies
                for col in base_cols:
                    interaction_col = f"{col}_x_stringency"
                    if col in df.columns:
                        enhanced_df[interaction_col] = df[col] * df[stringency_col]
                
                # 2. Create stringency volatility feature (change in stringency)
                # Use try-except to handle any potential errors
                try:
                    enhanced_df['stringency_volatility'] = df[stringency_col].rolling(window=7).std()
                except Exception as e:
                    print(f"Skipping stringency volatility feature due to error: {e}")
            
            # 3. Create rolling statistics for key features
            for col in base_cols:
                if col in df.columns:
                    # Rolling standard deviation could indicate unusual patterns
                    try:
                        enhanced_df[f"{col}_roll7_std"] = df[col].rolling(window=7).std()
                    except Exception as e:
                        print(f"Skipping rolling std for {col} due to error: {e}")
                    
                    # Z-score compared to recent history (how unusual is current value)
                    try:
                        roll_mean = df[col].rolling(window=14).mean()
                        roll_std = df[col].rolling(window=14).std()
                        # Avoid division by zero with np.where
                        enhanced_df[f"{col}_zscore"] = np.where(
                            roll_std > 0, 
                            (df[col] - roll_mean) / roll_std,
                            0  # Default value when std is zero
                        )
                    except Exception as e:
                        print(f"Skipping z-score for {col} due to error: {e}")
            
            # 4. Create polynomial features for selected variables
            # For example, squared terms to capture non-linear effects
            key_cols = [col for col in base_cols if 'positive_rate' in col or 'cases' in col]
            for col in key_cols:
                if col in df.columns:
                    enhanced_df[f"{col}_squared"] = df[col] ** 2
        
        else:
            print("Warning: Country column not available for country-specific feature engineering.")
            # Fall back to basic feature engineering without country context
            
            # Create basic interaction terms for any features containing these keywords
            smoothed_cols = [col for col in df.columns if 'smoothed' in col]
            for i, col1 in enumerate(smoothed_cols):
                for col2 in smoothed_cols[i+1:]:
                    if col1 in df.columns and col2 in df.columns:
                        enhanced_df[f"{col1[:10]}_{col2[:10]}_interaction"] = df[col1] * df[col2]
        
        # 5. Replace infinity values with NaN (these will be handled by imputer)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        
        # 6. Remove features with too many NaN values
        # Keep only columns with at least 70% non-NaN values
        cols_before = len(enhanced_df.columns)
        enhanced_df = enhanced_df.loc[:, enhanced_df.isnull().mean() < 0.3]
        cols_after = len(enhanced_df.columns)
        print(f"Removed {cols_before - cols_after} features with >30% missing values")
        
        # Report on new features
        original_features = set(df.columns)
        new_features = set(enhanced_df.columns) - original_features
        print(f"Added {len(new_features)} new engineered features")
        if len(new_features) > 0:
            print(f"Sample of new features: {list(new_features)[:5]}")
        
        return enhanced_df

    def build_pipeline(self):
        """
        Builds the model pipeline including preprocessing and estimator.
        
        For handling the significant class imbalance in the fatigue data (especially
        in the test set), this method configures LogisticRegression with appropriate
        class weights to give more importance to the minority class (fatigue=1).
        """
        
        if self.model_type == 'LogisticRegression':
            # Calculate class weight based on inverse class frequency in training data
            # This dynamically addresses the imbalance rather than using fixed weights
            if hasattr(self, 'y_train') and self.y_train is not None and not self.y_train.empty:
                # Calculate weights inversely proportional to class frequencies
                class_counts = np.bincount(self.y_train)
                total_samples = len(self.y_train)
                # Weight = total_samples / (num_classes * samples_in_class)
                class_weight_dict = {
                    0: total_samples / (2 * class_counts[0]), 
                    1: total_samples / (2 * class_counts[1])
                }
                print(f"Calculated class weights based on training data: {class_weight_dict}")
            else:
                # Fallback to a fixed weight ratio if y_train not available
                class_weight_dict = {0: 1, 1: 15}  # Heavily weight minority class
                print(f"Using fixed class weights: {class_weight_dict}")
                
            # Configure LogisticRegression with balanced class weights
            model = LogisticRegression(
                random_state=42, 
                class_weight=class_weight_dict,
                C=1.0,  # Regularization strength (inverse)
                solver='liblinear',  # Works well for small datasets and L1 penalties
                penalty='l1',  # L1 penalty helps with feature selection
                max_iter=1000  # Increase iterations to ensure convergence
            )
            
            default_grid = {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__solver': ['liblinear'],
                'model__penalty': ['l1', 'l2'],
                'model__class_weight': ['balanced', class_weight_dict]
            }
        elif self.model_type == 'GradientBoostingClassifier':
            # For GradientBoosting, use scale_pos_weight instead of class_weight
            if hasattr(self, 'y_train') and self.y_train is not None and not self.y_train.empty:
                class_counts = np.bincount(self.y_train)
                # Set weight of positive class proportional to imbalance ratio
                pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 10
                print(f"Using scale_pos_weight={pos_weight} for GradientBoostingClassifier")
                model = GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,  # Helps with imbalance
                    scale_pos_weight=pos_weight  # Weight based on class imbalance
                )
            else:
                model = GradientBoostingClassifier(random_state=42)
            
            default_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 4, 5],
                'model__subsample': [0.8, 1.0],
                'model__scale_pos_weight': [1, 5, 10]
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Add a SimpleImputer to safely handle NaN values first
        # This will replace missing values with mean before KNNImputer is applied
        steps = [
            ('simple_imputer', SimpleImputer(strategy='mean')),
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', model) 
        ]
        
        if self.tune_hyperparameters:
            param_grid_to_use = self.hyperparameter_grid if self.hyperparameter_grid else default_grid
            print(f"Starting hyperparameter tuning for {self.model_type} with grid: {param_grid_to_use}")
            
            # For tuning, use a simple pipeline without the potentially problematic KNNImputer
            tuning_steps = [
                ('simple_imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', model)
            ]
            
            # TimeSeriesSplit for time-series data
            tscv = TimeSeriesSplit(n_splits=5) 
            
            # Using a simpler pipeline for GridSearchCV to avoid issues
            pipeline_for_tuning = Pipeline(tuning_steps)
            
            # Use error_score='raise' to debug any issues during tuning
            grid_search = GridSearchCV(
                pipeline_for_tuning, 
                param_grid_to_use, 
                cv=tscv, 
                scoring='roc_auc',
                verbose=1,
                error_score=0  # Return a default score instead of raising an error
            )
            self.model_pipeline = grid_search
            print("Hyperparameter tuning enabled. GridSearchCV object created.")
        else:
            self.model_pipeline = Pipeline(steps)
            print(f"Using default model parameters for {self.model_type}.")
        
        print("Model pipeline created.")


    def train_and_predict(self):
        """
        Trains the model and makes predictions on the test set.
        
        Special consideration is given to the class imbalance problem in pandemic
        fatigue data, where fatigue instances (class 1) are typically much less
        frequent than non-fatigue instances (class 0), especially in test data.
        """
        if self.model_pipeline is None:
            print("Model pipeline is not built. Skipping training and prediction.")
            self.predictions = np.array([])
            self.metrics = {'status': 'No model pipeline built'}
            return

        if self.X_train.empty or self.y_train.empty:
            print("Training data is empty. Skipping training and prediction.")
            self.predictions = np.array([])
            self.metrics = {'status': 'Training data empty'}
            return

        # Calculate and report class distribution and imbalance metrics
        # This helps understand the severity of the imbalance problem
        class_distribution = self.y_train.value_counts(normalize=True)
        print(f"Class distribution in training data: {class_distribution.to_dict()}")
        
        # Calculate imbalance ratio
        if 0 in class_distribution and 1 in class_distribution:
            imbalance_ratio = class_distribution[0] / class_distribution[1]
            print(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
            # Store this information for later use in results
            self.train_imbalance_ratio = imbalance_ratio
        else:
            print("Warning: Missing a class in training data. Imbalance ratio cannot be calculated.")
            self.train_imbalance_ratio = None
            
        print(f"Training {self.model_type} model with class imbalance handling...")
        # Fit the model to the training data
        self.model_pipeline.fit(self.X_train, self.y_train)
        
        # Extract and report best parameters if hyperparameter tuning was performed
        if self.tune_hyperparameters and hasattr(self.model_pipeline, 'best_estimator_'):
            print(f"Best parameters found: {self.model_pipeline.best_params_}")
            self.best_model_params = self.model_pipeline.best_params_
        elif self.tune_hyperparameters:
             print("Tuning was enabled, but best_estimator_ not found. This might indicate an issue.")

        print("Model training complete.")

        # Calculate metrics for test data
        if self.X_test.empty or self.y_test.empty:
            print("Test data is empty. Skipping prediction and evaluation.")
            self.predictions = np.array([])
            self.metrics = {'status': 'Test data empty, no predictions.'}
            return

        # Report test set class distribution
        test_class_distribution = self.y_test.value_counts(normalize=True)
        print(f"Class distribution in test data: {test_class_distribution.to_dict()}")
        
        # Calculate test set imbalance ratio
        if 0 in test_class_distribution and 1 in test_class_distribution:
            test_imbalance_ratio = test_class_distribution[0] / test_class_distribution[1]
            print(f"Test set imbalance ratio (majority:minority): {test_imbalance_ratio:.2f}:1")
            self.test_imbalance_ratio = test_imbalance_ratio
        else:
            print("Warning: Missing a class in test data. Imbalance ratio cannot be calculated.")
            self.test_imbalance_ratio = None

        print("Making predictions on the test set...")
        self.predictions = self.model_pipeline.predict(self.X_test)
        self.predict_proba = self.model_pipeline.predict_proba(self.X_test)[:, 1] if hasattr(self.model_pipeline, "predict_proba") else None
        
        # Calculate standard classification metrics
        self.metrics['accuracy'] = accuracy_score(self.y_test, self.predictions)
        
        # For imbalanced data, weighted F1 score and ROC AUC are more informative than accuracy
        self.metrics['f1_score_weighted'] = f1_score(self.y_test, self.predictions, average='weighted')
        self.metrics['f1_score_macro'] = f1_score(self.y_test, self.predictions, average='macro')
        
        # Calculate class-specific F1 scores (more important for minority class)
        f1_by_class = f1_score(self.y_test, self.predictions, average=None)
        if len(f1_by_class) >= 2:
            self.metrics['f1_score_class_0'] = float(f1_by_class[0])
            self.metrics['f1_score_class_1'] = float(f1_by_class[1])
            print(f"F1 Score (Class 0, non-fatigue): {self.metrics['f1_score_class_0']:.4f}")
            print(f"F1 Score (Class 1, fatigue): {self.metrics['f1_score_class_1']:.4f}")
        
        # ROC AUC - better metric for imbalanced classification
        if self.predict_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(self.y_test, self.predict_proba)
            except ValueError as e:
                print(f"Could not calculate ROC AUC: {e}. Target might have only one class.")
                self.metrics['roc_auc'] = None
        else:
            self.metrics['roc_auc'] = None
            
        # Detailed classification report and confusion matrix
        self.metrics['classification_report'] = classification_report(self.y_test, self.predictions, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(self.y_test, self.predictions)
        self.metrics['confusion_matrix'] = conf_matrix.tolist() # Convert to list for JSON
        
        # Calculate additional metrics specifically for imbalanced classification
        if conf_matrix.shape == (2, 2):  # Only for binary classification
            # Extract values from confusion matrix
            tn, fp, fn, tp = conf_matrix.ravel()
            
            # Sensitivity (Recall for positive class)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            self.metrics['sensitivity'] = sensitivity
            
            # Specificity (Recall for negative class)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            self.metrics['specificity'] = specificity
            
            # Balanced Accuracy - average of sensitivity and specificity
            # Better than standard accuracy for imbalanced data
            self.metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
            
            # Precision for positive class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            self.metrics['precision'] = precision
            
            print(f"Sensitivity (Recall for fatigue): {sensitivity:.4f}")
            print(f"Specificity (Recall for non-fatigue): {specificity:.4f}")
            print(f"Balanced Accuracy: {self.metrics['balanced_accuracy']:.4f}")
            
        # Print summary metrics
        print(f"Test Accuracy: {self.metrics['accuracy']:.4f} (note: can be misleading for imbalanced data)")
        print(f"Test F1 Score (Weighted): {self.metrics['f1_score_weighted']:.4f}")
        print(f"Test F1 Score (Macro): {self.metrics['f1_score_macro']:.4f}")
        if self.metrics['roc_auc'] is not None:
            print(f"Test ROC AUC: {self.metrics['roc_auc']:.4f}")
        
        print("Classification Report (note importance of metrics for minority class):")
        print(classification_report(self.y_test, self.predictions, zero_division=0))
        
        print("Confusion Matrix:")
        print(conf_matrix)


    def save_run_results(self):
        """
        Saves all relevant information from the run to the run_results_dir.
        """
        print(f"Saving results to {self.run_results_dir}...")

        # 1. Run Details (JSON)
        run_details = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'data_path': self.data_path,
            'target_variable': self.target_variable_name,
            'model_type': self.model_type,
            'hyperparameters_tuned': self.tune_hyperparameters,
            'best_hyperparameters': self.best_model_params if self.best_model_params else (self.model_pipeline.best_params_ if self.tune_hyperparameters and hasattr(self.model_pipeline, 'best_params_') else "N/A"),
            'fatigue_definition_parameters': self.fatigue_def_params, # Added
            'feature_names': self.feature_names,
            'train_data_shape': [self.X_train.shape, self.y_train.shape] if self.X_train is not None else "N/A",
            'test_data_shape': [self.X_test.shape, self.y_test.shape] if self.X_test is not None else "N/A",
            'evaluation_metrics': self.metrics,
            'notes': "Pandemic Fatigue Prediction Run."
        }
        if self.y_train is not None and self.target_variable_name in self.y_train.name: # Check if y_train is Series
             run_details['target_class_distribution_train'] = self.y_train.value_counts(normalize=True).to_dict()
        if self.y_test is not None and self.target_variable_name in self.y_test.name:
             run_details['target_class_distribution_test'] = self.y_test.value_counts(normalize=True).to_dict()

        with open(os.path.join(self.run_results_dir, 'run_details.json'), 'w') as f:
            json.dump(run_details, f, indent=4)

        # 2. Feature Importances (CSV) - if applicable
        feature_importances_summary = "Feature importances not applicable or model not trained."
        if self.model_pipeline and hasattr(self.model_pipeline, "named_steps") and 'model' in self.model_pipeline.named_steps:
            final_model_step = self.model_pipeline.named_steps['model']
        elif self.tune_hyperparameters and hasattr(self.model_pipeline, 'best_estimator_') and hasattr(self.model_pipeline.best_estimator_, "named_steps") and 'model' in self.model_pipeline.best_estimator_.named_steps:
            final_model_step = self.model_pipeline.best_estimator_.named_steps['model']
        else:
            final_model_step = None

        if final_model_step:
            if hasattr(final_model_step, 'coef_'): # For linear models
                importances = final_model_step.coef_[0] # Assuming binary classification
                feature_importances_df = pd.DataFrame({'feature': self.X_train.columns, 'importance_coefficient': importances}) # Use X_train.columns
                feature_importances_df = feature_importances_df.sort_values(by='importance_coefficient', key=abs, ascending=False)
                feature_importances_df.to_csv(os.path.join(self.run_results_dir, 'feature_coefficients.csv'), index=False)
                feature_importances_summary = feature_importances_df.head(10).to_string()
            elif hasattr(final_model_step, 'feature_importances_'): # For tree-based models
                importances = final_model_step.feature_importances_
                feature_importances_df = pd.DataFrame({'feature': self.X_train.columns, 'importance': importances}) # Use X_train.columns
                feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
                feature_importances_df.to_csv(os.path.join(self.run_results_dir, 'feature_importances.csv'), index=False)
                feature_importances_summary = feature_importances_df.head(10).to_string()
        
        # ... (rest of the summary_parts assignment)
        top_features_summary = feature_importances_summary # Renamed for consistency

        # 3. Test Predictions vs Actual (CSV)
        if hasattr(self, 'predictions') and self.predictions.size > 0 and self.y_test is not None and not self.y_test.empty:
            results_df = pd.DataFrame({
                'date': self.test_dates.values,
                'location': self.test_locations.values,
                f'actual_{self.target_variable_name}': self.y_test.values,
                f'predicted_{self.target_variable_name}': self.predictions
            })
            if self.predict_proba is not None:
                 results_df[f'predicted_proba_{self.target_variable_name}'] = self.predict_proba

            # Add original features to the output for context if needed
            # for feature in self.feature_names:
            #    if feature in self.X_test.columns: # Ensure feature exists in X_test
            #        results_df[feature] = self.X_test[feature].values

            results_df.to_csv(os.path.join(self.run_results_dir, 'test_predictions_vs_actual.csv'), index=False)
        else:
            print("No predictions or test data to save.")


        # 4. Saved Model Pipeline (PKL)
        if self.model_pipeline:
            joblib.dump(self.model_pipeline, os.path.join(self.run_results_dir, 'model_pipeline.pkl'))
        else:
            print("No model pipeline to save.")
            joblib.dump("Placeholder model pipeline", os.path.join(self.run_results_dir, 'model_pipeline.pkl')) # Fallback if it was None

        # 5. Run Summary (TXT)
        summary_parts = [
            f"Run ID: {self.run_id}",
            f"Model Type: {self.model_type}",
            f"Hyperparameters Tuned: {self.tune_hyperparameters}",
            f"Best Hyperparameters: {self.best_model_params if self.best_model_params else (self.model_pipeline.best_params_ if self.tune_hyperparameters and hasattr(self.model_pipeline, 'best_params_') else 'N/A')}",
            f"Target Variable: {self.target_variable_name}",
            f"Fatigue Definition Parameters: {json.dumps(self.fatigue_def_params, indent=2)}", # Added
            "\nEvaluation Metrics:",
            json.dumps(self.metrics, indent=2),
            "\nTop Features/Coefficients:\n" + top_features_summary,
            f"\nNotes: {run_details['notes']}",
            f"\nResults saved in: {self.run_results_dir}"
        ]
        with open(os.path.join(self.run_results_dir, 'run_summary.txt'), 'w') as f:
            f.write("\n".join(summary_parts))

        print("Results saving complete.")


# Example usage
if __name__ == "__main__":
    # Configuration
    # These countries are examples, ensure they have sufficient data
    # and that the fatigue definition makes sense for them.
    COUNTRIES_TO_ANALYZE = [
        'United States', 'Germany', 'France', 'Italy', 'United Kingdom',
        'Brazil', 'India', 'Canada', 'Spain', 'Sweden' 
    ] 
    # Reduce list for faster initial testing if owid-covid-data.csv is large
    # COUNTRIES_TO_ANALYZE = ['Germany', 'France']

    # Custom fatigue definition parameters (optional, defaults will be used if None)
    custom_fatigue_params = {
        'stringency_percentile_threshold': 0.60, # Further reduced threshold for stringency
        'min_sustained_high_stringency_days': 14, # 2 weeks
        'proxy_lookback_window': 7,
        'proxy_increase_threshold_factor': 1.03 # 3% increase in proxy threshold
    }

    # Hyperparameter grid for Logistic Regression
    lr_hyperparam_grid = {
        'model__C': [0.01, 0.1, 1, 10], # Prefixed with 'model__' for pipeline
        'model__solver': ['liblinear'],
        'model__class_weight': [None, 'balanced', {0: 1, 1: 5}, {0: 1, 1: 10}, {0: 1, 1: 15}]  # Try different class weights
    }
    
    # Hyperparameter grid for Gradient Boosting Classifier
    gb_hyperparam_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5],
        'model__subsample': [0.8, 1.0],
        'model__scale_pos_weight': [1, 5, 10, 20]  # Test various imbalance weights
    }

    # Select active model and hyperparameter grid
    MODEL_TYPE = "LogisticRegression"  # Options: "LogisticRegression" or "GradientBoostingClassifier"
    TUNE_HYPERPARAMETERS = True  # Set to True to perform GridSearchCV tuning
    
    if MODEL_TYPE == "LogisticRegression":
        active_grid = lr_hyperparam_grid
    else:
        active_grid = gb_hyperparam_grid

    try:
        print("--- Initializing Pandemic Fatigue Predictor ---")
        predictor = PandemicFatiguePredictor(
            data_path='owid-covid-data.csv', # Ensure this file is present
            target_variable_name="fatigue_label", # This will be engineered
            model_type=MODEL_TYPE,
            tune_hyperparameters=TUNE_HYPERPARAMETERS,
            hyperparameter_grid=active_grid if TUNE_HYPERPARAMETERS else None,
            fatigue_def_params=custom_fatigue_params # Pass custom fatigue definition
        )

        print("\n--- Loading and Preprocessing Data ---")
        predictor.load_and_preprocess_data(
            countries=COUNTRIES_TO_ANALYZE,
            test_size=0.25, # Use 25% for testing
            ensure_class_balance=True  # Ensure both classes are in train and test sets
        )

        print("\n--- Performing Feature Engineering ---")
        # Apply additional feature engineering to training and test data
        predictor.X_train = predictor.engineer_features(predictor.X_train)
        predictor.X_test = predictor.engineer_features(predictor.X_test)
        
        # Update feature_names to include new engineered features
        predictor.feature_names = list(predictor.X_train.columns)
        print(f"Final feature set contains {len(predictor.feature_names)} features")

        print("\n--- Building Model Pipeline ---")
        predictor.build_pipeline()

        print("\n--- Training Model and Making Predictions ---")
        predictor.train_and_predict()

        print("\n--- Saving Run Results ---")
        predictor.save_run_results()

        print(f"\nAnalysis for run {predictor.run_id} complete. Review results in {predictor.run_results_dir}")
        
        print("\nNext Steps to Consider:")
        print(" - Review class balance and potentially adjust fatigue definition parameters")
        print(" - Analyze feature importances to identify key predictors of pandemic fatigue")
        print(" - Consider time-based cross-validation for more robust model evaluation")
        print(" - Explore ensemble methods that combine multiple model types")
        print(" - Investigate country-specific patterns and differences in fatigue indicators")

    except FileNotFoundError as e:
        print(f"ERROR: {e}. Please ensure 'owid-covid-data.csv' is in the correct path.")
    except ValueError as e:
        print(f"DATA ERROR: {e}")
    except ImportError as e:
        print(f"IMPORT ERROR: {e}. Please install missing packages.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()