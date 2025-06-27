"""
Pandemic Fatigue Analysis and Prediction

This module aims to identify, analyze, and potentially forecast periods of "pandemic fatigue."
Pandemic fatigue is operationally defined based on indicators such as high stringency 
coinciding with high or unexpectedly increasing disease transmission proxies 
(e.g., positive rate, cases per test).

Enhanced with social media sentiment analysis and mobility data integration for improved 
detection of pandemic fatigue across different regions and cultural contexts.

The script preprocesses data, engineers features, trains various classification models,
and saves results in a structured manner.
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Plotting will be handled separately or in EDA notebooks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix

import os
import json
import joblib
from datetime import datetime

# Try importing optional packages for sentiment analysis
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Sentiment analysis features will be limited.")

# Try importing transformers for advanced sentiment analysis
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Advanced sentiment analysis will not be available.")

# Base directory for results
RESULTS_BASE_DIR = "results/pandemic_fatigue"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


class PandemicFatiguePredictor:
    """
    Identifies, analyzes, and potentially forecasts pandemic fatigue using
    both epidemiological data and social media sentiment analysis.
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
                 sentiment_data_path=None,
                 mobility_data_path=None,
                 use_sentiment_analysis=False,
                 use_mobility_data=False,
                 sentiment_model='vader',  # 'vader' or 'transformers'
                 results_base_dir=RESULTS_BASE_DIR,
                 run_id=None):
        """
        Initialize the PandemicFatiguePredictor with enhanced features.

        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing the OWID dataset.
        target_variable_name : str
            Name of the engineered target variable representing fatigue.
        model_type : str
            Type of model to use (e.g., 'LogisticRegression', 'GradientBoosting', 'RandomForest').
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning.
        hyperparameter_grid : dict, optional
            Grid of hyperparameters for tuning.
        fatigue_def_params : dict, optional
            Parameters for defining the fatigue metric.
        country_col : str
            Name of the column identifying countries/locations.
        date_col : str
            Name of the column for dates.
        sentiment_data_path : str, optional
            Path to CSV file containing sentiment data from social media.
        mobility_data_path : str, optional
            Path to CSV file containing mobility data.
        use_sentiment_analysis : bool
            Whether to incorporate sentiment analysis features.
        use_mobility_data : bool
            Whether to incorporate mobility data features.
        sentiment_model : str
            Which sentiment analysis model to use ('vader' or 'transformers').
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
        
        # New parameters for enhanced features
        self.sentiment_data_path = sentiment_data_path
        self.mobility_data_path = mobility_data_path
        self.use_sentiment_analysis = use_sentiment_analysis
        self.use_mobility_data = use_mobility_data
        self.sentiment_model = sentiment_model
        
        # Check if sentiment analysis is requested but NLTK is not available
        if use_sentiment_analysis and not NLTK_AVAILABLE:
            print("Warning: Sentiment analysis requested but NLTK not available. Disabling sentiment analysis.")
            self.use_sentiment_analysis = False
            
        # Check if advanced transformers sentiment analysis is requested but not available
        if use_sentiment_analysis and sentiment_model == 'transformers' and not TRANSFORMERS_AVAILABLE:
            print("Warning: Transformers sentiment analysis requested but library not available. Falling back to VADER.")
            self.sentiment_model = 'vader'

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

        # Additional properties for run tracking and results
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_base_dir, f"{self.run_id}_{self.model_type}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize sentiment analysis tools if requested
        self.sentiment_analyzer = None
        if self.use_sentiment_analysis:
            self._initialize_sentiment_analyzer()
            
        # Model and results properties
        self.data = None  # Will store the raw data
        self.processed_data = None  # Will store processed data
        self.model = None  # Will store the trained model
        self.feature_importances = None  # Will store feature importances
        self.evaluation_results = None  # Will store evaluation metrics
            
    def _initialize_sentiment_analyzer(self):
        """Initialize the appropriate sentiment analyzer based on settings."""
        if self.sentiment_model == 'vader':
            if not NLTK_AVAILABLE:
                print("NLTK not available. Cannot initialize VADER sentiment analyzer.")
                return
                
            try:
                # Download necessary NLTK data if not already present
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                print("VADER sentiment analyzer initialized successfully.")
            except Exception as e:
                print(f"Error initializing VADER sentiment analyzer: {e}")
                self.use_sentiment_analysis = False
                
        elif self.sentiment_model == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                print("Transformers library not available. Cannot initialize transformer sentiment analyzer.")
                return
                
            try:
                # Initialize the transformer sentiment pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True
                )
                print("Transformer sentiment analyzer initialized successfully.")
            except Exception as e:
                print(f"Error initializing transformer sentiment analyzer: {e}")
                self.use_sentiment_analysis = False
        else:
            print(f"Unknown sentiment model: {self.sentiment_model}")
            self.use_sentiment_analysis = False
    
    def load_and_preprocess_data(self):
        """
        Load COVID-19 data, sentiment data, and mobility data (if available),
        then preprocess and combine them for analysis.
        """
        # Load main COVID data
        print(f"Loading COVID data from {self.data_path}...")
        self.covid_data = pd.read_csv(self.data_path, parse_dates=[self.date_col])
        
        # Load sentiment data if available and requested
        self.sentiment_data = None
        if self.use_sentiment_analysis and self.sentiment_data_path:
            try:
                print(f"Loading sentiment data from {self.sentiment_data_path}...")
                self.sentiment_data = pd.read_csv(self.sentiment_data_path, parse_dates=['date'])
                print(f"Loaded sentiment data with {len(self.sentiment_data)} rows.")
            except Exception as e:
                print(f"Error loading sentiment data: {e}")
                self.use_sentiment_analysis = False
                
        # Load mobility data if available and requested
        self.mobility_data = None
        if self.use_mobility_data and self.mobility_data_path:
            try:
                print(f"Loading mobility data from {self.mobility_data_path}...")
                self.mobility_data = pd.read_csv(self.mobility_data_path, parse_dates=['date'])
                print(f"Loaded mobility data with {len(self.mobility_data)} rows.")
            except Exception as e:
                print(f"Error loading mobility data: {e}")
                self.use_mobility_data = False
        
        # Preprocess and merge datasets
        self._preprocess_and_merge_data()
        
    def _preprocess_and_merge_data(self):
        """Preprocess each dataset and merge them based on country and date."""
        # Process COVID data
        # ...existing code for COVID data preprocessing...
        
        # Process sentiment data if available
        if self.sentiment_data is not None:
            # Convert date to datetime if needed
            if not pd.api.types.is_datetime64_dtype(self.sentiment_data['date']):
                self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
            
            # Standardize country/location column name if different
            if 'country' in self.sentiment_data.columns and self.country_col != 'country':
                self.sentiment_data = self.sentiment_data.rename(columns={'country': self.country_col})
                
            # Aggregate sentiment data by day and country
            agg_cols = ['sentiment_score', 'positive_ratio', 'negative_ratio', 'neutral_ratio']
            available_cols = [col for col in agg_cols if col in self.sentiment_data.columns]
            
            if available_cols:
                sentiment_agg = self.sentiment_data.groupby([self.country_col, 'date'])[available_cols].mean().reset_index()
                
                # Merge with COVID data
                self.covid_data = pd.merge(
                    self.covid_data, 
                    sentiment_agg, 
                    on=[self.country_col, self.date_col], 
                    how='left'
                )
                
                # Create lagged sentiment features (7, 14, 21 days)
                for lag in [7, 14, 21]:
                    for col in available_cols:
                        self.covid_data[f'{col}_lag_{lag}'] = self.covid_data.groupby(self.country_col)[col].shift(lag)
            
        # Process mobility data if available
        if self.mobility_data is not None:
            # Convert date to datetime if needed
            if not pd.api.types.is_datetime64_dtype(self.mobility_data['date']):
                self.mobility_data['date'] = pd.to_datetime(self.mobility_data['date'])
            
            # Standardize country/location column name if different
            if 'country' in self.mobility_data.columns and self.country_col != 'country':
                self.mobility_data = self.mobility_data.rename(columns={'country': self.country_col})
                
            # Define mobility columns - adjust based on actual data format
            mobility_cols = [col for col in self.mobility_data.columns 
                            if any(term in col.lower() for term in ['retail', 'grocery', 'parks', 
                                                               'transit', 'workplace', 'residential'])]
            
            if mobility_cols:
                # Merge with COVID data
                self.covid_data = pd.merge(
                    self.covid_data, 
                    self.mobility_data[[self.country_col, 'date'] + mobility_cols], 
                    on=[self.country_col, self.date_col], 
                    how='left'
                )
                
                # Create lagged mobility features (7, 14 days)
                for lag in [7, 14]:
                    for col in mobility_cols:
                        self.covid_data[f'{col}_lag_{lag}'] = self.covid_data.groupby(self.country_col)[col].shift(lag)
        
        # Compute sentiment from text data if available and no pre-computed sentiment
        if self.use_sentiment_analysis and 'tweet_text' in self.covid_data.columns:
            print("Computing sentiment scores from text data...")
            self.covid_data['computed_sentiment'] = self.covid_data['tweet_text'].apply(self._compute_sentiment)
        
        # Store preprocessed data
        self.data = self.covid_data
        print(f"Preprocessing complete. Final dataset has {len(self.data)} rows and {len(self.data.columns)} columns.")
    
    def _compute_sentiment(self, text):
        """Compute sentiment score for a given text using the initialized analyzer."""
        if not self.sentiment_analyzer or not isinstance(text, str):
            return 0.0
            
        try:
            if self.sentiment_model == 'vader':
                # VADER returns a dictionary with different scores
                scores = self.sentiment_analyzer.polarity_scores(text)
                return scores['compound']  # The compound score is a normalized score between -1 and 1
                
            elif self.sentiment_model == 'transformers':
                # Transformers pipeline returns a list with label and score
                result = self.sentiment_analyzer(text)[0]
                # Convert POSITIVE/NEGATIVE to a score between -1 and 1
                score = result['score']
                if result['label'] == 'NEGATIVE':
                    score = -score
                return score
        except Exception as e:
            print(f"Error computing sentiment for text: {e}")
            return 0.0

    def define_fatigue_periods(self):
        """
        Define pandemic fatigue periods based on the criteria specified in fatigue_def_params.
        
        Pandemic fatigue is characterized by:
        1. Sustained high stringency measures
        2. Increasing or high transmission despite restrictions
        
        Now enhanced with sentiment data where available.
        """
        # ...existing fatigue period definition code...
        
        # Incorporate sentiment data if available
        if 'sentiment_score' in self.data.columns:
            # Define additional fatigue indicators based on sentiment
            # Example: Identify periods where sentiment becomes significantly more negative
            # during high stringency periods
            
            # Calculate rolling sentiment average
            self.data['sentiment_14d_avg'] = self.data.groupby(self.country_col)['sentiment_score'].transform(
                lambda x: x.rolling(14, min_periods=7).mean()
            )
            
            # Calculate sentiment change rate
            self.data['sentiment_change'] = self.data.groupby(self.country_col)['sentiment_14d_avg'].transform(
                lambda x: x.pct_change(periods=7)
            )
            
            # Define sentiment-based fatigue as periods of high stringency + deteriorating sentiment
            stringency_high = self._get_high_stringency_periods()
            sentiment_deteriorating = (self.data['sentiment_change'] < -0.1)  # 10% deterioration threshold
            
            # Combine with existing fatigue definition
            self.data['sentiment_fatigue'] = stringency_high & sentiment_deteriorating
            
            # Create a combined fatigue indicator
            if 'fatigue_indicator' in self.data.columns:
                self.data['combined_fatigue'] = self.data['fatigue_indicator'] | self.data['sentiment_fatigue']
            else:
                print("Warning: Main fatigue indicator not found, using sentiment-based definition only")
                self.data['fatigue_indicator'] = self.data['sentiment_fatigue']
                
        # Incorporate mobility data if available
        mobility_cols = [col for col in self.data.columns 
                        if any(term in col.lower() for term in ['retail', 'grocery', 'parks', 
                                                           'transit', 'workplace', 'residential'])]
        
        if mobility_cols:
            # Define mobility-based fatigue (e.g., increasing mobility despite high restrictions)
            # ...code to define mobility-based fatigue indicators...
            pass
        
        # --- Ensure a main fatigue indicator is always created ---
        if 'fatigue_indicator' not in self.data.columns:
            # Simple operationalization: high stringency + rising cases
            stringency_threshold = self.fatigue_def_params.get('stringency_threshold', 60) if self.fatigue_def_params else 60
            case_increase_window = self.fatigue_def_params.get('case_increase_window', 14) if self.fatigue_def_params else 14
            case_increase_threshold = self.fatigue_def_params.get('case_increase_threshold', 0.2) if self.fatigue_def_params else 0.2
            
            # High stringency
            self.data['high_stringency'] = self.data['stringency_index'] >= stringency_threshold
            # Rolling case increase
            self.data['case_rolling_avg'] = self.data.groupby(self.country_col)['new_cases_smoothed_per_million'].transform(lambda x: x.rolling(case_increase_window, min_periods=7).mean())
            self.data['case_pct_change'] = self.data.groupby(self.country_col)['case_rolling_avg'].transform(lambda x: x.pct_change(periods=case_increase_window))
            self.data['rising_cases'] = self.data['case_pct_change'] > case_increase_threshold
            # Fatigue indicator: high stringency & rising cases
            self.data['fatigue_indicator'] = self.data['high_stringency'] & self.data['rising_cases']

        # Ensure target variable is properly set
        self.target_variable_name = 'combined_fatigue' if 'combined_fatigue' in self.data.columns else self.target_variable_name
        
        print(f"Fatigue periods defined. Using target variable: {self.target_variable_name}")

    def train_model(self, data, features, target=None, test_size=0.2, random_state=42):
        """
        Train a model to predict pandemic fatigue based on available features.
        Now supports multiple model types and advanced feature selection.
        """
        if target is None:
            target = self.target_variable_name
        X = data[features]
        y = data[target].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Create the model based on specified type
        if self.model_type == 'LogisticRegression':
            base_model = LogisticRegression(random_state=42, class_weight='balanced')
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            } if self.tune_hyperparameters else {}
            
        elif self.model_type == 'GradientBoosting':
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            } if self.tune_hyperparameters else {}
            
        elif self.model_type == 'RandomForest':
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, 30, None]
            } if self.tune_hyperparameters else {}
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Create the full modeling pipeline with preprocessing
        pipeline_steps = [
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', base_model)
        ]
        
        model_pipeline = Pipeline(pipeline_steps)
        
        # Implement hyperparameter tuning if requested
        if self.tune_hyperparameters and param_grid:
            time_series_cv = TimeSeriesSplit(n_splits=5)
            model = GridSearchCV(
                model_pipeline,
                param_grid={f'model__{param}': values for param, values in param_grid.items()},
                cv=time_series_cv,
                scoring='balanced_accuracy',
                n_jobs=-1
            )
        else:
            model = model_pipeline
            
        # Fit the model
        model.fit(X_train, y_train)
        
        # Extract best model if using GridSearchCV
        if self.tune_hyperparameters and param_grid:
            print(f"Best parameters: {model.best_params_}")
            self.best_params = model.best_params_
            best_model = model.best_estimator_
        else:
            best_model = model
            
        # Make predictions and evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Calculate evaluation metrics
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            evaluation['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
        # Extract feature importances if available
        if hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['model'], 'feature_importances_'):
            feature_importances = best_model.named_steps['model'].feature_importances_
            feature_names = X_train.columns
            self.feature_importances = {name: importance for name, importance in zip(feature_names, feature_importances)}
        
        # Save model and results
        self.model = best_model
        self.evaluation_results = evaluation
        
        # Save to disk
        joblib.dump(best_model, os.path.join(self.run_dir, f'{self.model_type}_model.pkl'))
        with open(os.path.join(self.run_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation, f)
            
        if self.feature_importances:
            with open(os.path.join(self.run_dir, 'feature_importances.json'), 'w') as f:
                json.dump(self.feature_importances, f)
                
        print(f"Model training complete. Balanced accuracy: {evaluation.get('balanced_accuracy', 'N/A')}")
        
        return evaluation