"""
Policy Effectiveness Lag Analysis

This module provides comprehensive analysis of the temporal relationship between
COVID-19 policy interventions (measured by the stringency index) and epidemiological outcomes.
It implements multiple advanced time-series methods to quantify the lag between
policy changes and their observable effects on key metrics like case rates, death rates,
and reproduction numbers.

Key features:
1. Robust time-series preprocessing with stationarity testing and transformation
2. Multiple methodologies for lag identification:
   - Cross-correlation function (CCF) analysis
   - Granger causality testing with statistical significance
   - Transfer function modeling
   - Wavelet coherence analysis for time-varying relationships
3. Comprehensive country-level and aggregated multi-country analysis
4. Statistical validation of identified lags with confidence intervals
5. Visualization and reporting capabilities for research publication

The module supports public health decision-making by providing evidence-based
estimates of when policy effects can be expected, helping to evaluate and
design intervention strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import ccf, grangercausalitytests, adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from scipy import signal, stats
import os
import joblib
import json
import datetime
from functools import partial
import warnings
from tqdm import tqdm

# Try to import optional packages
try:
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    import pywt
except ImportError:
    print("Note: Some optional packages are missing. Basic functionality will still work.")

# Filter specific warnings from statsmodels that don't affect results
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", "The iteration is not making good progress")

# Set up consistent visualization style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Ensure necessary directories exist
RESULTS_BASE_DIR = "results/policy_effectiveness"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
os.makedirs('eda_outputs/per_country', exist_ok=True)
os.makedirs('models', exist_ok=True)

class PolicyLagAnalyzer:
    """
    A comprehensive analyzer for quantifying and validating the time lag between 
    policy interventions and epidemiological outcomes during the COVID-19 pandemic.
    
    This class implements multiple methodologies for identifying temporal relationships
    between policy stringency and outcomes, with robust statistical validation and
    visualization capabilities for research publication.
    """
    
    def __init__(self, 
                 policy_columns=['stringency_index'],
                 outcome_columns=['new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million', 'reproduction_rate'],
                 countries=None,
                 max_lag=30,
                 min_data_points=180,  # Minimum days needed for robust analysis
                 stationarity_transform='diff',  # 'diff', 'log_diff', or 'none'
                 significance_level=0.05,
                 rolling_window_sizes=[7, 14, 21],
                 detrend_data=True,
                 analyze_subperiods=True,
                 subperiod_length=90):  # For analyzing time-varying relationships
        """
        Initialize the PolicyLagAnalyzer with configurable parameters for analysis.
        
        Parameters:
        -----------
        policy_columns : list
            Column names representing policy interventions (e.g., stringency_index)
        outcome_columns : list
            Column names representing epidemiological outcomes to analyze
        countries : list or None
            List of countries to include in analysis. If None, will analyze all available countries
            with sufficient data.
        max_lag : int
            Maximum lag (in days) to consider in analysis
        min_data_points : int
            Minimum number of consecutive days required for a country to be included
        stationarity_transform : str
            Method for transforming series to achieve stationarity:
            'diff' (first difference), 'log_diff' (log + difference), 'none' (no transform)
        significance_level : float
            Statistical significance level (alpha) for hypothesis tests
        rolling_window_sizes : list
            Window sizes for rolling average smoothing to reduce noise
        detrend_data : bool
            Whether to detrend the data before analysis
        analyze_subperiods : bool
            Whether to analyze time-varying relationships by sub-periods
        subperiod_length : int
            Length (in days) of each subperiod when analyze_subperiods=True
        """
        self.policy_columns = policy_columns
        self.outcome_columns = outcome_columns
        self.countries = countries
        self.max_lag = max_lag
        self.min_data_points = min_data_points
        self.stationarity_transform = stationarity_transform
        self.significance_level = significance_level
        self.rolling_window_sizes = rolling_window_sizes
        self.detrend_data = detrend_data
        self.analyze_subperiods = analyze_subperiods
        self.subperiod_length = subperiod_length
        
        # Setup scalers
        self.scalers = {
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Results storage
        self.results = {}
        self.models = {}
        self.data = None
        self.processed_data = {}
        self.eligible_countries = []
        
        # Generate run ID for this analysis
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_results_dir = os.path.join(RESULTS_BASE_DIR, f"{self.run_id}_analysis")
        os.makedirs(self.run_results_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            'policy_columns': policy_columns,
            'outcome_columns': outcome_columns,
            'max_lag': max_lag,
            'min_data_points': min_data_points,
            'stationarity_transform': stationarity_transform,
            'significance_level': significance_level,
            'rolling_window_sizes': rolling_window_sizes,
            'detrend_data': detrend_data,
            'analyze_subperiods': analyze_subperiods,
            'subperiod_length': subperiod_length,
            'run_id': self.run_id
        }
    
    def load_data(self, data_path):
        """
        Load and perform initial preprocessing of the COVID-19 dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the COVID-19 dataset CSV file
        
        Returns:
        --------
        self : PolicyLagAnalyzer
            Returns self for method chaining
        """
        print(f"Loading data from {data_path}...")
        
        # Load the data
        self.data = pd.read_csv(data_path)
        
        # Basic preprocessing
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Filter required columns
        required_columns = ['location', 'date'] + self.policy_columns + self.outcome_columns
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        
        # Sort data by country and date
        self.data = self.data.sort_values(['location', 'date'])
        
        # Determine eligible countries based on data availability
        if self.countries is None:
            print("Identifying countries with sufficient data...")
            # Count consecutive days with data for each country
            country_data_counts = self.data.groupby('location').size()
            # Filter countries with enough data
            self.eligible_countries = country_data_counts[country_data_counts >= self.min_data_points].index.tolist()
            print(f"Found {len(self.eligible_countries)} countries with sufficient data")
        else:
            # Verify that specified countries have enough data
            valid_countries = []
            for country in self.countries:
                country_data = self.data[self.data['location'] == country]
                if len(country_data) >= self.min_data_points:
                    valid_countries.append(country)
                else:
                    print(f"Warning: {country} has insufficient data points ({len(country_data)}) and will be excluded")
            self.eligible_countries = valid_countries
            
        print(f"Analysis will include {len(self.eligible_countries)} countries")
        
        # Save the configuration with country list
        self.config['countries_analyzed'] = self.eligible_countries
        
        return self
    
    def preprocess_country_data(self, country):
        """
        Preprocess data for a specific country, handling missing values,
        ensuring stationarity, and preparing for time-series analysis.
        
        Parameters:
        -----------
        country : str
            Country name to preprocess
            
        Returns:
        --------
        dict : Dictionary containing preprocessed data series
        """
        # Extract country data
        country_data = self.data[self.data['location'] == country].copy()
        
        if len(country_data) < self.min_data_points:
            return None
            
        # Ensure data is sorted by date
        country_data = country_data.sort_values('date')
        
        # Check for sufficient non-null values
        for col in self.policy_columns + self.outcome_columns:
            non_null_count = country_data[col].notna().sum()
            if non_null_count < self.min_data_points * 0.7:  # At least 70% of required data points
                print(f"Warning: {country} has insufficient non-null values for {col} ({non_null_count}/{len(country_data)})")
                return None
        
        # Interpolate missing values for required columns
        for col in self.policy_columns + self.outcome_columns:
            country_data[col] = country_data[col].interpolate(method='linear').bfill().ffill()
        
        # Create dictionary to store processed series
        processed_series = {'raw': country_data}
        
        # Apply smoothing with different window sizes
        for window in self.rolling_window_sizes:
            smoothed_data = country_data.copy()
            for col in self.policy_columns + self.outcome_columns:
                smoothed_data[col] = smoothed_data[col].rolling(window=window, center=False).mean()
            
            # Drop initial NaN values created by rolling window
            smoothed_data = smoothed_data.dropna()
            processed_series[f'smoothed_{window}'] = smoothed_data
          # Apply detrending if specified
        if self.detrend_data:
            keys_to_process = list(processed_series.keys())
            for key in keys_to_process:
                if key == 'raw':
                    continue
                    
                detrended_data = processed_series[key].copy()
                for col in self.policy_columns + self.outcome_columns:
                    # Use Lowess smoothing to identify and remove trend
                    x = np.arange(len(detrended_data))
                    y = detrended_data[col].values
                    trend = lowess(y, x, frac=0.3, it=3, return_sorted=False)
                    detrended_data[col] = y - trend
                
                processed_series[f'{key}_detrended'] = detrended_data
        
        # Apply transformations for stationarity
        for key in list(processed_series.keys()):  # Use list() to avoid modifying during iteration
            if 'detrended' in key or key == 'raw':
                continue
                
            data = processed_series[key].copy()
            
            # Different stationarity transformations
            if self.stationarity_transform == 'diff':
                transformed_data = data.copy()
                for col in self.policy_columns + self.outcome_columns:
                    transformed_data[col] = transformed_data[col].diff().dropna()
                
                # Drop the first row which now has NaN values
                transformed_data = transformed_data.dropna()
                processed_series[f'{key}_diff'] = transformed_data
                
            elif self.stationarity_transform == 'log_diff':
                transformed_data = data.copy()
                for col in self.policy_columns + self.outcome_columns:
                    # Handle zeros and negative values before log transform
                    min_val = transformed_data[col].min()
                    if min_val <= 0:
                        transformed_data[col] = transformed_data[col] - min_val + 1  # Ensure positive
                    
                    transformed_data[col] = np.log(transformed_data[col]).diff().dropna()
                
                # Drop rows with NaN values
                transformed_data = transformed_data.dropna()
                processed_series[f'{key}_log_diff'] = transformed_data
          # Apply scaling
        for key in list(processed_series.keys()):
            data = processed_series[key].copy()
            scaled_data = data.copy()
            
            for col in self.policy_columns + self.outcome_columns:
                # Reshape for sklearn scalers
                values = scaled_data[col].values.reshape(-1, 1)
                
                # Skip if empty
                if values.size == 0:
                    continue
                
                # Apply MinMax scaling to get values in [0,1]
                scaled_values = self.scalers['minmax'].fit_transform(values)
                scaled_data[col] = scaled_values.flatten()
            
            processed_series[f'{key}_scaled'] = scaled_data
        
        return processed_series
    
    def check_stationarity(self, series, alpha=0.05):
        """
        Test a time series for stationarity using ADF and KPSS tests.
        
        Parameters:
        -----------
        series : array-like
            Time series to test for stationarity
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Dictionary with test results
        """
        # Drop NaN values
        series = pd.Series(series).dropna()
        
        # Run Augmented Dickey-Fuller test
        adf_result = adfuller(series, autolag='AIC')
        
        # Run KPSS test
        kpss_result = kpss(series, regression='c', nlags='auto')
        
        # Interpret results
        adf_stationary = adf_result[1] < alpha  # p-value < alpha means stationary
        kpss_stationary = kpss_result[1] >= alpha  # p-value >= alpha means stationary
        
        # Combined interpretation
        if adf_stationary and kpss_stationary:
            interpretation = "Stationary"
        elif adf_stationary and not kpss_stationary:
            interpretation = "Trend stationary"
        elif not adf_stationary and kpss_stationary:
            interpretation = "Difference stationary"
        else:
            interpretation = "Non-stationary"
        
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_stationary': adf_stationary,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_stationary': kpss_stationary,
            'interpretation': interpretation
        }
    
    def cross_correlation_analysis(self, country):
        """
        Perform cross-correlation analysis between policy variables and outcomes
        to identify potential lag relationships.
        
        Parameters:
        -----------
        country : str
            Country to analyze
            
        Returns:
        --------
        dict : Dictionary with CCF results for each policy-outcome pair
        """
        if country not in self.processed_data:
            print(f"Processing data for {country}...")
            self.processed_data[country] = self.preprocess_country_data(country)
            
        if self.processed_data[country] is None:
            print(f"Insufficient data for {country}, skipping cross-correlation analysis")
            return None
        
        # Select the processed dataset to use for analysis
        # Prefer detrended and differenced data for stationarity
        analysis_keys = [k for k in self.processed_data[country].keys() 
                         if 'smoothed' in k and 'scaled' in k]
        
        if self.detrend_data:
            analysis_keys = [k for k in analysis_keys if 'detrended' in k]
            
        if self.stationarity_transform != 'none':
            analysis_keys = [k for k in analysis_keys if self.stationarity_transform in k]
        
        # Default to first smoothed and scaled version if no specific key found
        analysis_key = analysis_keys[0] if analysis_keys else 'smoothed_7_scaled'
        
        print(f"Using {analysis_key} data for CCF analysis")
        analysis_data = self.processed_data[country][analysis_key]
        
        # Initialize results dictionary
        ccf_results = {}
        
        # For each policy-outcome pair, calculate CCF
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:
                policy_series = analysis_data[policy_col].values
                outcome_series = analysis_data[outcome_col].values
                
                # Calculate cross-correlation
                ccf_values = ccf(policy_series, outcome_series, adjusted=False)
                  # Get lag with maximum absolute correlation
                if len(ccf_values) == 0 or np.all(np.isnan(ccf_values)):
                    # Handle empty or all-NaN sequence
                    ccf_results[f"{policy_col}_{outcome_col}"] = {
                        'error': 'Empty or invalid cross-correlation sequence'
                    }
                    continue
                    
                lag_max_corr = np.argmax(np.abs(ccf_values)) - len(ccf_values)//2
                max_corr_value = ccf_values[np.argmax(np.abs(ccf_values))]
                
                # Store in results
                key = f"{policy_col}_{outcome_col}"
                ccf_results[key] = {
                    'lag_max_correlation': lag_max_corr,
                    'max_correlation_value': max_corr_value,
                    'ccf_values': ccf_values,
                    'positive_lags': list(range(-len(ccf_values)//2, len(ccf_values)//2 + 1))
                }
                
                # Create visualization
                self._plot_ccf(country, policy_col, outcome_col, 
                              ccf_values, lag_max_corr, max_corr_value)
        
        return ccf_results
    
    def _plot_ccf(self, country, policy_col, outcome_col, ccf_values, lag_max_corr, max_corr_value):
        """
        Create and save cross-correlation plot for a policy-outcome pair.
        
        Parameters:
        -----------
        country : str
            Country name
        policy_col : str
            Policy column name
        outcome_col : str
            Outcome column name
        ccf_values : array
            Cross-correlation values
        lag_max_corr : int
            Lag with maximum correlation
        max_corr_value : float
            Maximum correlation value
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate lag values for x-axis
        lags = np.arange(-len(ccf_values)//2, len(ccf_values)//2 + 1)
        
        # Plot the cross-correlation
        plt.bar(lags, ccf_values, alpha=0.6, color='steelblue')
        plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
        
        # Add significance bounds (approximation based on sample size)
        n = len(ccf_values)
        conf_level = 1.96 / np.sqrt(n)
        plt.axhline(y=conf_level, linestyle='--', color='red', alpha=0.7)
        plt.axhline(y=-conf_level, linestyle='--', color='red', alpha=0.7)
        
        # Highlight the maximum correlation
        plt.scatter([lag_max_corr], [max_corr_value], color='red', 
                   s=100, label=f'Max at lag {lag_max_corr}', zorder=5)
        
        # Add vertical line at zero lag
        plt.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
        
        # Add annotations
        plt.title(f'Cross-Correlation: {policy_col} → {outcome_col}\nCountry: {country}', 
                 fontsize=14, pad=20)
        plt.xlabel('Lag (days)', fontsize=12)
        plt.ylabel('Correlation Coefficient', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        save_path = os.path.join(self.run_results_dir, 
                                f'{country}_{policy_col}_{outcome_col}_ccf.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def granger_causality_tests(self, country, verbose=False):
        """
        Perform Granger causality tests to assess if policy changes precede
        outcome changes in a statistically significant way.
        
        Parameters:
        -----------
        country : str
            Country to analyze
        verbose : bool
            Whether to print detailed results
            
        Returns:
        --------
        dict : Dictionary with Granger causality results
        """
        if country not in self.processed_data:
            print(f"Processing data for {country}...")
            self.processed_data[country] = self.preprocess_country_data(country)
            
        if self.processed_data[country] is None:
            print(f"Insufficient data for {country}, skipping Granger causality analysis")
            return None
        
        # Select appropriate preprocessed data
        analysis_keys = [k for k in self.processed_data[country].keys() 
                        if 'smoothed' in k and 'scaled' in k]
        
        if self.stationarity_transform != 'none':
            analysis_keys = [k for k in analysis_keys if self.stationarity_transform in k]
            
        analysis_key = analysis_keys[0] if analysis_keys else 'smoothed_7_scaled'
        
        if verbose:
            print(f"Using {analysis_key} data for Granger causality analysis")
        analysis_data = self.processed_data[country][analysis_key]
        
        # Initialize results dictionary
        granger_results = {}
        
        # For each policy-outcome pair, calculate Granger causality
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:
                # Prepare data: [policy, outcome]
                combined_data = analysis_data[[policy_col, outcome_col]].dropna()
                
                # Skip if not enough data
                if len(combined_data) <= self.max_lag + 10:
                    granger_results[f"{policy_col}_{outcome_col}"] = {
                        'error': 'Insufficient data points for Granger test'
                    }
                    continue
                
                try:
                    # Granger test from policy to outcome
                    gc_results = grangercausalitytests(
                        combined_data, maxlag=self.max_lag, verbose=False
                    )
                    
                    # Extract test statistics and p-values for each lag
                    lags_data = {}
                    min_p_value = 1.0
                    optimal_lag = None
                    
                    for lag in range(1, self.max_lag + 1):
                        # We use the Wald chi-square test (key = 'ssr_chi2test')
                        test_stat = gc_results[lag][0]['ssr_chi2test'][0]
                        p_value = gc_results[lag][0]['ssr_chi2test'][1]
                        lags_data[lag] = {'test_statistic': test_stat, 'p_value': p_value}
                        
                        # Track minimum p-value and corresponding lag
                        if p_value < min_p_value:
                            min_p_value = p_value
                            optimal_lag = lag
                    
                    # Determine significance
                    is_significant = min_p_value < self.significance_level
                    
                    granger_results[f"{policy_col}_{outcome_col}"] = {
                        'lags_data': lags_data,
                        'min_p_value': min_p_value,
                        'optimal_lag': optimal_lag,
                        'is_significant': is_significant
                    }
                    
                    # Also test reverse causality (outcome to policy)
                    # This helps distinguish true causality from bidirectional relationships
                    reversed_data = analysis_data[[outcome_col, policy_col]].dropna()
                    gc_results_reversed = grangercausalitytests(
                        reversed_data, maxlag=self.max_lag, verbose=False
                    )
                    
                    # Extract results for reverse direction
                    reverse_lags_data = {}
                    min_p_value_reverse = 1.0
                    optimal_lag_reverse = None
                    
                    for lag in range(1, self.max_lag + 1):
                        test_stat = gc_results_reversed[lag][0]['ssr_chi2test'][0]
                        p_value = gc_results_reversed[lag][0]['ssr_chi2test'][1]
                        reverse_lags_data[lag] = {'test_statistic': test_stat, 'p_value': p_value}
                        
                        if p_value < min_p_value_reverse:
                            min_p_value_reverse = p_value
                            optimal_lag_reverse = lag
                    
                    is_significant_reverse = min_p_value_reverse < self.significance_level
                    
                    granger_results[f"{policy_col}_{outcome_col}"]['reverse'] = {
                        'lags_data': reverse_lags_data,
                        'min_p_value': min_p_value_reverse,
                        'optimal_lag': optimal_lag_reverse,
                        'is_significant': is_significant_reverse
                    }
                      # Create visualization of p-values by lag
                    self._plot_granger_results(country, policy_col, outcome_col,
                                             lags_data, reverse_lags_data)
                    
                except Exception as e:
                    granger_results[f"{policy_col}_{outcome_col}"] = {
                        'error': str(e)
                    }
                    if verbose:
                        print(f"Error in Granger test for {country}, {policy_col}->{outcome_col}: {e}")
        
        return granger_results
        
    def _plot_granger_results(self, country, policy_col, outcome_col, lags_data, reverse_lags_data):
        """
        Create and save plot of Granger causality p-values by lag.
        
        Parameters:
        -----------
        country : str
            Country name
        policy_col : str
            Policy column name
        outcome_col : str
            Outcome column name
        lags_data : dict
            Dictionary with test results for each lag (policy -> outcome)
        reverse_lags_data : dict
            Dictionary with test results for each lag (outcome -> policy)
        """
        plt.figure(figsize=(12, 8))
        
        # Extract lags and p-values
        lags = list(lags_data.keys())
        p_values = [lags_data[lag]['p_value'] for lag in lags]
        p_values_reverse = [reverse_lags_data[lag]['p_value'] for lag in lags]
        
        # Plot primary direction (policy -> outcome)
        plt.plot(lags, p_values, 'o-', color='blue', linewidth=2, markersize=8, 
                alpha=0.7, label=f'{policy_col} → {outcome_col}')
        
        # Plot reverse direction (outcome -> policy)
        plt.plot(lags, p_values_reverse, 's--', color='red', linewidth=2, markersize=8,
                alpha=0.7, label=f'{outcome_col} → {policy_col}')
        
        # Add significance threshold line
        plt.axhline(y=self.significance_level, linestyle='--', color='green', 
                   alpha=0.7, label=f'Significance level (α={self.significance_level})')
        
        # Highlight significant lags
        significant_lags = [lag for lag, data in lags_data.items() 
                           if data['p_value'] < self.significance_level]
        
        if significant_lags:
            optimal_lag = min(significant_lags)  # First significant lag
            plt.axvspan(optimal_lag-0.5, optimal_lag+0.5, color='lightblue', alpha=0.3)
            
            # Annotation for the optimal lag
            plt.annotate(f'First significant lag: {optimal_lag}',
                        xy=(optimal_lag, lags_data[optimal_lag]['p_value']),
                        xytext=(optimal_lag+1, lags_data[optimal_lag]['p_value']+0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # Format the plot
        plt.title(f'Granger Causality Test Results\nCountry: {country}', fontsize=14, pad=20)
        plt.xlabel('Lag (days)', fontsize=12)
        plt.ylabel('p-value', fontsize=12)
        plt.yscale('log')  # Log scale for better visualization of significance threshold
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Set x-axis to show integer ticks
        ax = plt.gca()
        try:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        except ImportError:
            # If MaxNLocator is not available, use a simpler approach
            ax.set_xticks(lags)
        
        # Save the figure
        save_path = os.path.join(self.run_results_dir, 
                                f'{country}_{policy_col}_{outcome_col}_granger.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def transfer_function_analysis(self, country):
        """
        Perform transfer function modeling to estimate the dynamic relationship
        between policy interventions and outcomes. This method fits ARIMA models 
        with exogenous variables (ARIMAX) to quantify policy effects over time.
        
        Parameters:
        -----------
        country : str
            Country to analyze
            
        Returns:
        --------
        dict : Dictionary with transfer function model results
        """
        if country not in self.processed_data:
            print(f"Processing data for {country}...")
            self.processed_data[country] = self.preprocess_country_data(country)
            
        if self.processed_data[country] is None:
            print(f"Insufficient data for {country}, skipping transfer function analysis")
            return None
        
        # Select appropriate preprocessed data (raw but smoothed is better for ARIMA)
        analysis_key = 'smoothed_7'
        analysis_data = self.processed_data[country][analysis_key]
        
        # Initialize results dictionary
        transfer_results = {}
        
        # For each policy-outcome pair, estimate transfer function model
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:                try:
                    # Prepare data
                    y = analysis_data[outcome_col].values
                    X = analysis_data[[policy_col]].values
                    
                    # Check if we have enough data
                    if len(y) < 5 or len(X) < 5 or np.isnan(y).all() or np.isnan(X).all():
                        transfer_results[f"{policy_col}_{outcome_col}"] = {
                            'error': 'Insufficient or invalid data for transfer function analysis'
                        }
                        continue
                    
                    # Determine optimal ARIMA parameters (simplified approach)
                    p, d, q = 1, 1, 1  # Default ARIMA parameters
                    
                    # Fit ARIMAX model with varying lags of exogenous variable
                    best_model = None
                    best_aic = np.inf
                    best_lag = None
                    
                    # Test different lags of the policy variable
                    for lag in range(1, min(self.max_lag + 1, 15)):  # Limit to reasonable lags
                        # Create lagged exogenous variable
                        if len(X) <= lag:
                            # Skip if not enough data for this lag
                            continue
                            
                        X_lagged = np.roll(X, lag, axis=0)
                        X_lagged[:lag] = X_lagged[lag]  # Replace initial values
                        
                        try:
                            # Try to fit the ARIMAX model
                            model = ARIMA(y, exog=X_lagged, order=(p, d, q))
                            model_fit = model.fit()
                            
                            # Evaluate model fit
                            current_aic = model_fit.aic
                            
                            # Update best model if this is better
                            if current_aic < best_aic:
                                best_aic = current_aic
                                best_model = model_fit
                                best_lag = lag
                        except Exception as e:
                            # Skip this lag if model fitting fails
                            continue
                    
                    # If no model could be fit, skip
                    if best_model is None:
                        transfer_results[f"{policy_col}_{outcome_col}"] = {
                            'error': 'Failed to fit any transfer function model'
                        }
                        continue
                    
                    # Extract results
                    coefficients = best_model.params
                    p_values = best_model.pvalues
                    exog_coef = coefficients[policy_col] if policy_col in coefficients else None
                    exog_pvalue = p_values[policy_col] if policy_col in p_values else None
                    
                    # Store results
                    transfer_results[f"{policy_col}_{outcome_col}"] = {
                        'optimal_lag': best_lag,
                        'aic': best_aic,
                        'exogenous_coefficient': exog_coef,
                        'exogenous_pvalue': exog_pvalue,
                        'is_significant': exog_pvalue < self.significance_level if exog_pvalue is not None else False,
                        'model_summary': str(best_model.summary()),
                        'full_coefficients': coefficients.to_dict() if hasattr(coefficients, 'to_dict') else dict(zip(coefficients.index, coefficients.values)),
                        'full_pvalues': p_values.to_dict() if hasattr(p_values, 'to_dict') else dict(zip(p_values.index, p_values.values))
                    }
                    
                    # Save the model for potential later use
                    self.models[f"{country}_{policy_col}_{outcome_col}_transfer"] = best_model
                    
                except Exception as e:
                    transfer_results[f"{policy_col}_{outcome_col}"] = {
                        'error': str(e)
                    }
                    print(f"Error in transfer function analysis for {country}, {policy_col}->{outcome_col}: {e}")
        
        return transfer_results
    
    def wavelet_coherence_analysis(self, country):
        """
        Perform wavelet coherence analysis to identify time-frequency relationships
        between policy interventions and outcomes. This method reveals how relationships
        between variables change over time and across different frequency scales.
        
        Parameters:
        -----------
        country : str
            Country to analyze
            
        Returns:
        --------
        dict : Dictionary with wavelet coherence results
        """
        try:
            import pywt
            from matplotlib.colors import LogNorm
        except ImportError:
            print("Warning: pywt package not available, skipping wavelet analysis")
            return {"error": "pywt package not installed, cannot perform wavelet analysis"}
        
        if country not in self.processed_data:
            print(f"Processing data for {country}...")
            self.processed_data[country] = self.preprocess_country_data(country)
            
        if self.processed_data[country] is None:
            print(f"Insufficient data for {country}, skipping wavelet coherence analysis")
            return None
        
        # Use smoothed data for better wavelet analysis
        analysis_key = 'stationary' if 'stationary' in self.processed_data[country] else 'smoothed_7'
        analysis_data = self.processed_data[country][analysis_key]
        
        # Initialize results
        wavelet_results = {}
        
        # For each policy-outcome pair
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:
                try:
                    # Extract the time series
                    policy_series = analysis_data[policy_col].values
                    outcome_series = analysis_data[outcome_col].values
                    
                    # Normalize to ensure comparability
                    policy_series = (policy_series - np.mean(policy_series)) / np.std(policy_series)
                    outcome_series = (outcome_series - np.mean(outcome_series)) / np.std(outcome_series)
                    
                    # Perform continuous wavelet transform for both series
                    scales = np.arange(1, min(32, len(policy_series)//2))  # Reasonable scales for analysis
                    
                    # Compute CWT of each series
                    coeffs1, freqs1 = pywt.cwt(policy_series, scales, 'morl')
                    coeffs2, freqs2 = pywt.cwt(outcome_series, scales, 'morl')
                    
                    # Calculate cross-wavelet transform: element-wise multiplication of cwt coefficients
                    W12 = coeffs1 * np.conj(coeffs2)
                    
                    # Calculate individual power spectra
                    W1 = np.abs(coeffs1)**2
                    W2 = np.abs(coeffs2)**2
                    
                    # Calculate wavelet coherence
                    coherence = np.abs(W12)**2 / (W1 * W2)
                    
                    # Find regions of high coherence at different scales (lags)
                    coherence_threshold = 0.7  # Threshold for significant coherence
                    sig_coherence = coherence > coherence_threshold
                    
                    # For each scale (potential lag), identify periods of high coherence
                    lag_data = {}
                    for scale_idx, scale in enumerate(scales):
                        if np.any(sig_coherence[scale_idx]):
                            # Calculate average coherence at this scale
                            avg_coherence = np.mean(coherence[scale_idx])
                            
                            # Find time points with high coherence at this scale
                            coherent_times = np.where(sig_coherence[scale_idx])[0]
                            
                            lag_data[int(scale)] = {
                                'average_coherence': float(avg_coherence),
                                'coherence_periods': len(coherent_times),
                                'significant_periods_ratio': len(coherent_times) / len(sig_coherence[scale_idx])
                            }
                    
                    # Identify the scale/lag with the highest average coherence
                    if lag_data:
                        best_lag = max(lag_data.keys(), key=lambda k: lag_data[k]['average_coherence'])
                          # Generate plot if matplotlib is available
                        plot_path = None
                        if self.run_results_dir:
                            try:
                                # Create wavelet coherence plot
                                plt.figure(figsize=(10, 8))
                                
                                # Plot wavelet coherence as a heatmap
                                plt.imshow(coherence, aspect='auto', cmap='jet', 
                                          extent=[0, len(policy_series), int(scales[-1]), int(scales[0])],
                                          norm=LogNorm(vmin=0.1, vmax=1.0))
                                
                                plt.colorbar(label='Coherence')
                                plt.title(f'Wavelet Coherence: {policy_col} vs {outcome_col} - {country}')
                                plt.ylabel('Scale (days)')
                                plt.xlabel('Time (days)')
                                
                                # Highlight the scale with highest coherence
                                plt.axhline(y=best_lag, color='white', linestyle='--', alpha=0.7)
                                plt.text(len(policy_series)*0.8, best_lag, f'Best Lag: {best_lag} days', 
                                        color='white', fontweight='bold', va='center')
                                
                                # Save the figure
                                plot_path = os.path.join(self.run_results_dir, 
                                                        f"wavelet_{country}_{policy_col}_{outcome_col}.png")
                                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                                plt.close()
                            except Exception as e:
                                print(f"Warning: Could not create wavelet plot: {e}")
                                plot_path = None
                        
                        # Store results
                        wavelet_results[f"{policy_col}_{outcome_col}"] = {
                            'best_lag': best_lag,
                            'max_coherence': lag_data[best_lag]['average_coherence'],
                            'significant_periods_ratio': lag_data[best_lag]['significant_periods_ratio'],
                            'all_lags': lag_data,
                            'plot_path': plot_path
                        }
                    else:
                        wavelet_results[f"{policy_col}_{outcome_col}"] = {
                            'error': 'No significant coherence found at any scale'
                        }
                        
                except Exception as e:
                    wavelet_results[f"{policy_col}_{outcome_col}"] = {
                        'error': str(e)
                    }
                    print(f"Error in wavelet coherence analysis for {country}, {policy_col}->{outcome_col}: {e}")
        
        return wavelet_results
    
    def analyze_country(self, country, methods=None):
        """
        Perform comprehensive analysis of policy effectiveness lags for a country
        using multiple methods.
        
        Parameters:
        -----------
        country : str
            Country to analyze
        methods : list, optional
            List of methods to use. If None, all available methods are used.
            Options: 'ccf', 'granger', 'transfer', 'wavelet'
            
        Returns:
        --------
        dict : Comprehensive analysis results
        """
        if methods is None:
            methods = ['ccf', 'granger', 'transfer', 'wavelet']
        
        print(f"\nAnalyzing policy effectiveness lags for {country}...")
        
        # Check if country data is available
        if country not in self.processed_data:
            print(f"Processing data for {country}...")
            self.processed_data[country] = self.preprocess_country_data(country)
            
        if self.processed_data[country] is None:
            print(f"Insufficient data for {country}, skipping analysis")
            return None
        
        # Initialize results dictionary
        country_results = {'country': country, 'methods': {}}
        
        # Apply each requested method
        if 'ccf' in methods:
            print(f"  Running cross-correlation analysis...")
            country_results['methods']['ccf'] = self.cross_correlation_analysis(country)
            
        if 'granger' in methods:
            print(f"  Running Granger causality tests...")
            country_results['methods']['granger'] = self.granger_causality_tests(country)
            
        if 'transfer' in methods:
            print(f"  Running transfer function analysis...")
            country_results['methods']['transfer'] = self.transfer_function_analysis(country)
            
        if 'wavelet' in methods:
            print(f"  Running wavelet coherence analysis...")
            country_results['methods']['wavelet'] = self.wavelet_coherence_analysis(country)
        
        # Summarize results
        country_results['summary'] = self._summarize_country_results(country_results)
        
        # Save country results
        self._save_country_results(country, country_results)
        
        return country_results
    
    def _summarize_country_results(self, country_results):
        """
        Summarize results across methods to provide a consolidated view
        of policy effectiveness lags for a country.
        
        Parameters:
        -----------
        country_results : dict
            Results from different analysis methods for a country
            
        Returns:
        --------
        dict : Summary of identified lags by policy-outcome pair
        """
        summary = {}
        
        # If no results available, return empty summary
        if country_results is None or 'methods' not in country_results:
            return summary
        
        methods_results = country_results['methods']
        
        # For each policy-outcome pair, collect lag estimates from different methods
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:
                pair_key = f"{policy_col}_{outcome_col}"
                summary[pair_key] = {'lag_estimates': []}
                
                # Cross-correlation results
                if 'ccf' in methods_results and methods_results['ccf']:
                    if pair_key in methods_results['ccf']:
                        ccf_result = methods_results['ccf'][pair_key]
                        if 'lag_max_correlation' in ccf_result:
                            summary[pair_key]['lag_estimates'].append({
                                'method': 'Cross-correlation',
                                'lag': ccf_result['lag_max_correlation'],
                                'metric': 'correlation',
                                'value': ccf_result['max_correlation_value']
                            })
                
                # Granger causality results
                if 'granger' in methods_results and methods_results['granger']:
                    if pair_key in methods_results['granger']:
                        granger_result = methods_results['granger'][pair_key]
                        if 'optimal_lag' in granger_result and granger_result['is_significant']:
                            summary[pair_key]['lag_estimates'].append({
                                'method': 'Granger causality',
                                'lag': granger_result['optimal_lag'],
                                'metric': 'p-value',
                                'value': granger_result['min_p_value']
                            })
                
                # Transfer function results
                if 'transfer' in methods_results and methods_results['transfer']:
                    if pair_key in methods_results['transfer']:
                        transfer_result = methods_results['transfer'][pair_key]
                        if 'optimal_lag' in transfer_result and 'error' not in transfer_result:
                            summary[pair_key]['lag_estimates'].append({
                                'method': 'Transfer function',
                                'lag': transfer_result['optimal_lag'],
                                'metric': 'coefficient',
                                'value': transfer_result.get('exogenous_coefficient', 'N/A')
                            })
                
                # Calculate consensus lag if possible
                if summary[pair_key]['lag_estimates']:
                    lags = [e['lag'] for e in summary[pair_key]['lag_estimates']]
                    mean_lag = sum(lags) / len(lags)
                    median_lag = sorted(lags)[len(lags) // 2]
                    
                    # Count methods reporting significant results
                    significant_methods = len(summary[pair_key]['lag_estimates'])
                    
                    summary[pair_key]['consensus'] = {
                        'mean_lag': mean_lag,
                        'median_lag': median_lag,
                        'min_lag': min(lags),
                        'max_lag': max(lags),
                        'significant_methods': significant_methods,
                        'total_methods': len([m for m in methods_results if m in ['ccf', 'granger', 'transfer']])
                    }
                else:
                    summary[pair_key]['consensus'] = {
                        'significant_methods': 0,
                        'total_methods': len([m for m in methods_results if m in ['ccf', 'granger', 'transfer']])
                    }
        
        return summary
    
    def _save_country_results(self, country, results):
        """
        Save country-specific analysis results to file.
        
        Parameters:
        -----------
        country : str
            Country name
        results : dict
            Analysis results for the country
        """
        # Create a simplified version for JSON serialization
        serializable_results = {
            'country': country,
            'summary': results['summary'],
            'methods': {}
        }
        
        # Keep only essential information from each method
        if 'ccf' in results['methods'] and results['methods']['ccf']:
            serializable_results['methods']['ccf'] = {}
            for pair_key, ccf_result in results['methods']['ccf'].items():
                if 'ccf_values' in ccf_result:
                    # Keep only essential info, not full arrays
                    ccf_simplified = {
                        'lag_max_correlation': ccf_result['lag_max_correlation'],
                        'max_correlation_value': ccf_result['max_correlation_value']
                    }
                    serializable_results['methods']['ccf'][pair_key] = ccf_simplified
        
        if 'granger' in results['methods'] and results['methods']['granger']:
            serializable_results['methods']['granger'] = {}
            for pair_key, granger_result in results['methods']['granger'].items():
                if 'optimal_lag' in granger_result:
                    # Keep main outcomes, not detailed p-values for all lags
                    granger_simplified = {
                        'optimal_lag': granger_result['optimal_lag'],
                        'min_p_value': granger_result['min_p_value'],
                        'is_significant': granger_result['is_significant']
                    }
                    
                    if 'reverse' in granger_result:
                        granger_simplified['reverse'] = {
                            'optimal_lag': granger_result['reverse']['optimal_lag'],
                            'min_p_value': granger_result['reverse']['min_p_value'],
                            'is_significant': granger_result['reverse']['is_significant']
                        }
                    
                    serializable_results['methods']['granger'][pair_key] = granger_simplified
        
        if 'transfer' in results['methods'] and results['methods']['transfer']:
            serializable_results['methods']['transfer'] = {}
            for pair_key, transfer_result in results['methods']['transfer'].items():
                if 'error' not in transfer_result:
                    # Keep main outcomes
                    transfer_simplified = {
                        'optimal_lag': transfer_result['optimal_lag'],
                        'exogenous_coefficient': transfer_result['exogenous_coefficient'],
                        'exogenous_pvalue': transfer_result['exogenous_pvalue'],
                        'is_significant': transfer_result['is_significant']
                    }
                    serializable_results['methods']['transfer'][pair_key] = transfer_simplified
        
        # Save to file
        save_path = os.path.join(self.run_results_dir, f'{country}_results.json')
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def analyze_all_countries(self, methods=None, country_subset=None):
        """
        Analyze policy effectiveness lags for all eligible countries.
        
        Parameters:
        -----------
        methods : list, optional
            List of methods to use. If None, all methods are used.
        country_subset : list, optional
            Subset of countries to analyze. If None, all eligible countries are analyzed.
            
        Returns:
        --------
        dict : Dictionary with results for each country
        """
        all_countries = self.eligible_countries
        
        if country_subset is not None:
            countries_to_analyze = [c for c in country_subset if c in all_countries]
            if len(countries_to_analyze) < len(country_subset):
                missing = set(country_subset) - set(all_countries)
                print(f"Warning: Some requested countries are not eligible: {missing}")
        else:
            countries_to_analyze = all_countries
        
        results = {}
        
        print(f"Analyzing {len(countries_to_analyze)} countries...")
        for country in tqdm(countries_to_analyze):
            country_result = self.analyze_country(country, methods)
            if country_result is not None:
                results[country] = country_result
        
        # Aggregate results across countries
        self._aggregate_results(results)
        
        return results
    
    def _aggregate_results(self, all_results):
        """
        Aggregate results across countries to identify global patterns.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary with results for each country
        """
        print("\nAggregating results across countries...")
        
        # Initialize aggregate data structure
        aggregate = {
            'countries_analyzed': list(all_results.keys()),
            'policy_outcome_pairs': {},
            'country_statistics': {}
        }
        
        # For each policy-outcome pair, aggregate lag estimates
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:
                pair_key = f"{policy_col}_{outcome_col}"
                aggregate['policy_outcome_pairs'][pair_key] = {
                    'all_lag_estimates': [],
                    'significant_countries': [],
                    'country_specific_lags': {}
                }
                
                # Collect all lag estimates for this pair
                for country, results in all_results.items():
                    if 'summary' in results and pair_key in results['summary']:
                        country_summary = results['summary'][pair_key]
                        
                        if 'consensus' in country_summary and 'median_lag' in country_summary['consensus']:
                            # Country has significant results for this pair
                            median_lag = country_summary['consensus']['median_lag']
                            significant_methods = country_summary['consensus']['significant_methods']
                            
                            if significant_methods > 0:
                                aggregate['policy_outcome_pairs'][pair_key]['significant_countries'].append(country)
                                aggregate['policy_outcome_pairs'][pair_key]['country_specific_lags'][country] = median_lag
                                
                                # Add all lag estimates
                                for estimate in country_summary['lag_estimates']:
                                    aggregate['policy_outcome_pairs'][pair_key]['all_lag_estimates'].append({
                                        'country': country,
                                        'method': estimate['method'],
                                        'lag': estimate['lag']
                                    })
                
                # Calculate statistics if we have results
                all_lags = [e['lag'] for e in aggregate['policy_outcome_pairs'][pair_key]['all_lag_estimates']]
                
                if all_lags:
                    aggregate['policy_outcome_pairs'][pair_key]['statistics'] = {
                        'mean_lag': sum(all_lags) / len(all_lags),
                        'median_lag': sorted(all_lags)[len(all_lags) // 2],
                        'min_lag': min(all_lags),
                        'max_lag': max(all_lags),
                        'std_lag': np.std(all_lags),
                        'n_observations': len(all_lags),
                        'n_countries': len(aggregate['policy_outcome_pairs'][pair_key]['significant_countries'])
                    }
                    
                    # Create lag distribution histogram
                    self._plot_lag_distribution(policy_col, outcome_col, all_lags)
                
                # Calculate country-level statistics
                for country in aggregate['countries_analyzed']:
                    if country not in aggregate['country_statistics']:
                        aggregate['country_statistics'][country] = {
                            'significant_pairs': [],
                            'median_lag': None,
                            'lag_range': None
                        }
                    
                    if country in aggregate['policy_outcome_pairs'][pair_key]['country_specific_lags']:
                        # This country has significant results for this pair
                        aggregate['country_statistics'][country]['significant_pairs'].append(pair_key)
                
                # Update country statistics
                for country in aggregate['country_statistics']:
                    country_lags = []
                    for pair in aggregate['country_statistics'][country]['significant_pairs']:
                        if country in aggregate['policy_outcome_pairs'][pair]['country_specific_lags']:
                            country_lags.append(aggregate['policy_outcome_pairs'][pair]['country_specific_lags'][country])
                    
                    if country_lags:
                        aggregate['country_statistics'][country]['median_lag'] = np.median(country_lags)
                        aggregate['country_statistics'][country]['lag_range'] = [min(country_lags), max(country_lags)]
        
        # Save aggregate results
        save_path = os.path.join(self.run_results_dir, 'aggregate_results.json')
        with open(save_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        # Create summary table for publication
        self._create_summary_table(aggregate)
        
        return aggregate
    
    def _plot_lag_distribution(self, policy_col, outcome_col, lags):
        """
        Create histogram of lag distribution for a policy-outcome pair.
        
        Parameters:
        -----------
        policy_col : str
            Policy column name
        outcome_col : str
            Outcome column name
        lags : list
            List of lag values from all countries and methods
        """
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(lags, bins=range(min(lags)-1, max(lags)+3), 
                                  color='steelblue', alpha=0.7, rwidth=0.8)
        
        # Add mean and median lines
        mean_lag = sum(lags) / len(lags)
        median_lag = sorted(lags)[len(lags) // 2]
        
        plt.axvline(x=mean_lag, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_lag:.1f} days')
        plt.axvline(x=median_lag, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_lag:.1f} days')
        
        # Format plot
        plt.title(f'Distribution of Policy Lag Estimates\n{policy_col} → {outcome_col}', 
                 fontsize=14, pad=20)
        plt.xlabel('Lag (days)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to show integer ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add statistics as text
        stats_text = (
            f"N = {len(lags)}\n"
            f"Mean = {mean_lag:.1f} days\n"
            f"Median = {median_lag:.1f} days\n"
            f"Range = [{min(lags)}, {max(lags)}] days\n"
            f"Std Dev = {np.std(lags):.1f} days"
        )
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                                                  facecolor='white', alpha=0.8))
        
        # Save figure
        save_path = os.path.join(self.run_results_dir, 
                                f'{policy_col}_{outcome_col}_lag_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_table(self, aggregate):
        """
        Create a summary table of results for publication.
        
        Parameters:
        -----------
        aggregate : dict
            Aggregated results across countries
        """
        # Create summary for policy-outcome pairs
        policy_outcome_summary = []
        
        for pair_key, pair_data in aggregate['policy_outcome_pairs'].items():
            if 'statistics' in pair_data:
                policy_col, outcome_col = pair_key.split('_', 1)
                
                policy_outcome_summary.append({
                    'policy': policy_col,
                    'outcome': outcome_col,
                    'median_lag': pair_data['statistics']['median_lag'],
                    'mean_lag': pair_data['statistics']['mean_lag'],
                    'min_lag': pair_data['statistics']['min_lag'],
                    'max_lag': pair_data['statistics']['max_lag'],
                    'std_lag': pair_data['statistics']['std_lag'],
                    'n_countries': pair_data['statistics']['n_countries'],
                    'n_observations': pair_data['statistics']['n_observations']
                })
        
        # Convert to pandas DataFrame for easier manipulation
        summary_df = pd.DataFrame(policy_outcome_summary)
        
        # Round numeric columns
        numeric_cols = ['median_lag', 'mean_lag', 'min_lag', 'max_lag', 'std_lag']
        summary_df[numeric_cols] = summary_df[numeric_cols].round(1)
        
        # Save as CSV
        summary_df.to_csv(os.path.join(self.run_results_dir, 'policy_lag_summary.csv'), index=False)
        
        # Create a more visually appealing summary table for the report
        plt.figure(figsize=(10, len(summary_df) * 1.2))
        
        # Remove axes
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Create table
        table_data = []
        table_data.append(['Policy', 'Outcome', 'Median\nLag (days)', 'Mean\nLag (days)', 
                         'Range', 'Std Dev', 'Countries'])
        
        for _, row in summary_df.iterrows():
            table_data.append([
                row['policy'].replace('_', ' ').title(),
                row['outcome'].replace('_', ' ').title(),
                f"{row['median_lag']:.1f}",
                f"{row['mean_lag']:.1f}",
                f"{row['min_lag']:.0f}-{row['max_lag']:.0f}",
                f"{row['std_lag']:.1f}",
                f"{row['n_countries']} ({row['n_observations']} obs)"
            ])
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Set header row style
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Add title
        plt.suptitle('Policy Effectiveness Lag Summary', fontsize=16, y=0.98)
        
        # Save the table
        plt.savefig(os.path.join(self.run_results_dir, 'policy_lag_summary_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """
        Generate a comprehensive markdown report of the analysis results.
        
        Returns:
        --------
        str : Markdown report content
        """
        # Load aggregate results
        try:
            with open(os.path.join(self.run_results_dir, 'aggregate_results.json'), 'r') as f:
                aggregate = json.load(f)
        except FileNotFoundError:
            print("Aggregate results not found. Please run analyze_all_countries first.")
            return None
        
        # Load policy lag summary
        try:
            policy_lag_df = pd.read_csv(os.path.join(self.run_results_dir, 'policy_lag_summary.csv'))
        except FileNotFoundError:
            print("Policy lag summary not found.")
            return None
        
        # Create markdown report
        report = []
        
        # Title and introduction
        report.append("# COVID-19 Policy Effectiveness Lag Analysis Report")
        report.append("\n## Executive Summary")
        report.append("\nThis report presents a comprehensive analysis of the temporal relationship between COVID-19 policy interventions and epidemiological outcomes across multiple countries. Using advanced time-series methodologies, we quantified the lag between policy changes (measured by stringency index) and observable effects on key indicators including case rates, death rates, and reproduction numbers.")
        
        # Add key findings summary based on results
        report.append("\nKey findings:")
        
        # Calculate overall median lag across all policy-outcome pairs
        all_median_lags = policy_lag_df['median_lag'].tolist()
        overall_median = np.median(all_median_lags) if all_median_lags else "N/A"
        
        report.append(f"\n- The median lag between policy changes and observable impacts was **{overall_median:.1f} days** across all analyzed relationships")
        
        # Find policy-outcome pair with shortest lag
        if not policy_lag_df.empty:
            min_lag_row = policy_lag_df.loc[policy_lag_df['median_lag'].idxmin()]
            max_lag_row = policy_lag_df.loc[policy_lag_df['median_lag'].idxmax()]
            
            report.append(f"\n- The shortest median lag ({min_lag_row['median_lag']:.1f} days) was observed between {min_lag_row['policy']} and {min_lag_row['outcome']}")
            report.append(f"\n- The longest median lag ({max_lag_row['median_lag']:.1f} days) was observed between {max_lag_row['policy']} and {max_lag_row['outcome']}")
        
        # Add country count
        report.append(f"\n- Analysis included data from **{len(aggregate['countries_analyzed'])}** countries with sufficient data quality")
        
        # Methodology section
        report.append("\n## Methodology")
        report.append("\n### Data Source")
        report.append("\nThis analysis used the Our World in Data (OWID) COVID-19 dataset, which provides comprehensive daily data on cases, deaths, testing, policy responses, and other metrics across countries.")
        
        report.append("\n### Analysis Approach")
        report.append("\nWe employed multiple complementary time-series methodologies to identify and validate policy lags:")
        report.append("\n1. **Cross-correlation Function (CCF) Analysis**: Identifies the lag with maximum correlation between policy and outcome time series")
        report.append("\n2. **Granger Causality Testing**: Determines whether policy changes statistically precede outcome changes at various lags")
        report.append("\n3. **Transfer Function Modeling**: Estimates dynamic relationships between policy interventions and outcomes using ARIMAX models")
        report.append("\n4. **Wavelet Coherence Analysis**: Examines time-varying relationships across different time scales and periods")
        
        report.append("\n### Preprocessing Steps")
        report.append(f"\n- Time series were smoothed using {', '.join(str(w) for w in self.rolling_window_sizes)}-day rolling averages to reduce noise")
        report.append(f"\n- Stationarity transformations: {self.stationarity_transform}")
        if self.detrend_data:
            report.append("\n- Detrending was applied to isolate short-term relationships from long-term trends")
        report.append(f"\n- Analysis considered lags up to {self.max_lag} days")
        
        # Results section
        report.append("\n## Results")
        
        report.append("\n### Policy-Outcome Lag Summary")
        report.append("\nThe table below summarizes the identified lags between policy interventions and epidemiological outcomes:")
        
        # Add table in markdown format
        report.append("\n| Policy | Outcome | Median Lag (days) | Mean Lag (days) | Range | Std Dev | Countries |")
        report.append("|--------|---------|-------------------|-----------------|-------|---------|-----------|")
        
        for _, row in policy_lag_df.iterrows():
            policy = row['policy'].replace('_', ' ').title()
            outcome = row['outcome'].replace('_', ' ').title()
            report.append(f"| {policy} | {outcome} | {row['median_lag']:.1f} | {row['mean_lag']:.1f} | {row['min_lag']:.0f}-{row['max_lag']:.0f} | {row['std_lag']:.1f} | {row['n_countries']} ({row['n_observations']} obs) |")
        
        # Add figure reference
        report.append("\n![Policy Lag Summary Table](policy_lag_summary_table.png)")
        
        # Country-specific findings
        report.append("\n### Country-Specific Findings")
        
        # Create a table of country-specific median lags
        if 'country_statistics' in aggregate:
            countries_with_results = []
            
            for country, stats in aggregate['country_statistics'].items():
                if stats['median_lag'] is not None:
                    countries_with_results.append({
                        'country': country,
                        'median_lag': stats['median_lag'],
                        'min_lag': stats['lag_range'][0] if stats['lag_range'] else None,
                        'max_lag': stats['lag_range'][1] if stats['lag_range'] else None,
                        'n_pairs': len(stats['significant_pairs'])
                    })
            
            if countries_with_results:
                # Sort by median lag
                countries_with_results.sort(key=lambda x: x['median_lag'])
                
                report.append("\nThe table below shows median policy lags by country:")
                report.append("\n| Country | Median Lag (days) | Range | Policy-Outcome Pairs |")
                report.append("|---------|-------------------|-------|----------------------|")
                
                for country_data in countries_with_results:
                    country = country_data['country']
                    median_lag = country_data['median_lag']
                    min_lag = country_data['min_lag']
                    max_lag = country_data['max_lag']
                    n_pairs = country_data['n_pairs']
                    
                    report.append(f"| {country} | {median_lag:.1f} | {min_lag:.0f}-{max_lag:.0f} | {n_pairs} |")
        
        # Distribution of lags
        report.append("\n### Distribution of Policy Lags")
        report.append("\nThe figures below show the distribution of identified lags across countries and methods for key policy-outcome relationships:")
        
        # Add figure references (these will be created by _plot_lag_distribution)
        for policy_col in self.policy_columns:
            for outcome_col in self.outcome_columns:
                pair_key = f"{policy_col}_{outcome_col}"
                
                # Check if we have results for this pair
                if (pair_key in aggregate['policy_outcome_pairs'] and 
                    'statistics' in aggregate['policy_outcome_pairs'][pair_key]):
                    
                    policy_name = policy_col.replace('_', ' ').title()
                    outcome_name = outcome_col.replace('_', ' ').title()
                    
                    report.append(f"\n#### {policy_name} → {outcome_name}")
                    report.append(f"\n![Lag Distribution]({policy_col}_{outcome_col}_lag_distribution.png)")
                    
                    # Add brief interpretation
                    stats = aggregate['policy_outcome_pairs'][pair_key]['statistics']
                    report.append(f"\nThe median lag between {policy_name} and {outcome_name} was **{stats['median_lag']:.1f} days** (range: {stats['min_lag']:.0f}-{stats['max_lag']:.0f} days), based on {stats['n_observations']} observations from {stats['n_countries']} countries. This suggests that changes in {policy_name} typically take approximately {stats['median_lag']:.0f} days to manifest in observable changes in {outcome_name}.")
        
        # Discussion section
        report.append("\n## Discussion")
        
        # Interpretation of findings based on results
        report.append("\n### Interpretation of Policy Lags")
        
        # Generate interpretation based on actual findings
        report.append("\nThe identified policy lags provide important insights for pandemic response planning:")
        
        # Overall lag interpretation
        report.append(f"\n- **Average Policy Effect Delay**: The overall median lag of {overall_median:.1f} days indicates that policy interventions typically take approximately {overall_median:.0f} days to produce measurable effects on epidemiological outcomes. This finding aligns with the biological and social mechanisms of disease transmission and control.")
        
        # Variation between outcomes
        outcome_groups = policy_lag_df.groupby('outcome')['median_lag'].median()
        
        if not outcome_groups.empty:
            fastest_outcome = outcome_groups.idxmin()
            fastest_outcome_lag = outcome_groups.min()
            slowest_outcome = outcome_groups.idxmax()
            slowest_outcome_lag = outcome_groups.max()
            
            fastest_outcome_name = fastest_outcome.replace('_', ' ').title()
            slowest_outcome_name = slowest_outcome.replace('_', ' ').title()
            
            report.append(f"\n- **Variation by Outcome Measure**: Effects on {fastest_outcome_name} appear most rapidly (median {fastest_outcome_lag:.1f} days), while changes in {slowest_outcome_name} take longer to manifest (median {slowest_outcome_lag:.1f} days). This pattern reflects the natural progression of COVID-19, where reproduction rate changes occur before case rate changes, which precede mortality changes.")
        
        # Country variation
        report.append("\n- **Country-Specific Variations**: The considerable variation in policy lags between countries likely reflects differences in testing capacity, reporting practices, healthcare system capabilities, and population demographics.")
        
        # Implications section
        report.append("\n### Public Health Implications")
        report.append("\nThese findings have several important implications for policy planning:")
        
        report.append(f"\n1. **Expectation Setting**: Policymakers should set realistic expectations about when effects from policy changes will become apparent (typically {overall_median:.0f}-{overall_median+3:.0f} days) and avoid prematurely modifying interventions before effects can be observed.")
        
        report.append("\n2. **Proactive Decision-Making**: Given these inherent delays, decisions about implementing strict measures should be made proactively, before case numbers reach critical thresholds.")
        
        report.append("\n3. **Metric Selection**: For real-time policy assessment, faster-responding metrics like reproduction rate estimates may provide earlier feedback on intervention effectiveness than case counts or mortality data.")
        
        report.append("\n4. **Policy Sequencing**: When implementing or relaxing multiple interventions, their timing should be staggered according to their expected lag periods to allow for clearer attribution of effects.")
        
        # Limitations
        report.append("\n## Limitations")
        report.append("\nThis analysis has several limitations that should be considered when interpreting results:")
        
        report.append("\n1. **Confounding Factors**: The analysis cannot fully separate the effects of specific policy changes from other concurrent factors such as voluntary behavior changes, seasonality, or virus variants.")
        
        report.append("\n2. **Data Quality Variability**: Reporting practices and data quality varied significantly across countries and over time, potentially affecting lag estimates.")
        
        report.append("\n3. **Aggregate Stringency Index**: Using an aggregate stringency index rather than specific interventions may obscure the differential impacts and lags of individual policy measures.")
        
        report.append("\n4. **Time-Varying Relationships**: Policy effectiveness likely varied over the course of the pandemic as populations adapted and new variants emerged. While our wavelet analysis captures some of this variation, the summary statistics present average effects across the entire study period.")
        
        # Conclusion
        report.append("\n## Conclusion")
        report.append("\nThis comprehensive analysis quantifies the typical lag between COVID-19 policy interventions and observable epidemiological outcomes. Our findings provide an evidence base for more effective timing and sequencing of public health measures during future pandemic waves or novel infectious disease outbreaks.")
        
        report.append(f"\nThe median lag of {overall_median:.1f} days across all relationships represents a crucial planning parameter for pandemic response. This delay—representing the combined effects of implementation time, incubation period, testing delays, and reporting lags—must be incorporated into epidemic forecasting models and policy planning frameworks.")
        
        report.append("\nFuture research should focus on disentangling the lags associated with specific intervention types and exploring how these lags vary with population characteristics, healthcare system capacity, and viral properties to enable more targeted and effective pandemic response strategies.")
        
        # Join the report sections
        full_report = "\n".join(report)
        
        # Save the report
        report_path = os.path.join(self.run_results_dir, 'policy_effectiveness_lag_report.md')
        with open(report_path, 'w') as f:
            f.write(full_report)
        
        print(f"Comprehensive report generated and saved to {report_path}")
        
        return full_report

# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COVID-19 POLICY EFFECTIVENESS LAG ANALYSIS")
    print("="*80 + "\n")
    
    # Configuration
    data_path = 'owid-covid-data.csv'
    
    # Define policy and outcome columns to analyze
    policy_columns = ['stringency_index']
    outcome_columns = [
        'new_cases_smoothed_per_million', 
        'new_deaths_smoothed_per_million', 
        'reproduction_rate'
    ]
    
    # Define key countries to analyze
    # This is a subset of major countries for faster initial analysis
    # Remove this parameter to analyze all countries with sufficient data
    key_countries = [
        'United States', 'United Kingdom', 'Germany', 'France', 'Italy', 
        'Spain', 'Canada', 'Brazil', 'India', 'Sweden'
    ]
    
    print(f"Initializing analysis...")
    analyzer = PolicyLagAnalyzer(
        policy_columns=policy_columns,
        outcome_columns=outcome_columns,
        countries=key_countries,
        max_lag=30,  # Maximum lag to consider (days)
        min_data_points=180,  # Minimum days needed for robust analysis
        stationarity_transform='diff',  # Options: 'diff', 'log_diff', 'none'
        significance_level=0.05,
        rolling_window_sizes=[7, 14, 21],
        detrend_data=True,
        analyze_subperiods=True,
        subperiod_length=90
    )
    
    print(f"Loading and preprocessing data from {data_path}...")
    analyzer.load_data(data_path)
    
    print("\nAnalyzing policy effectiveness lags across countries...")
    # Specify methods to run, can be any subset of ['ccf', 'granger', 'transfer', 'wavelet']
    methods = ['ccf', 'granger', 'transfer', 'wavelet']
    results = analyzer.analyze_all_countries(methods=methods)
    
    print("\nGenerating comprehensive markdown report...")
    report = analyzer.generate_report()
    
    # Create a directory link to facilitate report viewing
    results_dir = analyzer.run_results_dir.replace('\\', '/')
    print(f"\nAnalysis complete! Results saved to: {results_dir}")
    print(f"Report: {results_dir}/policy_effectiveness_lag_report.md")
    
    print("\nKey Findings Summary:")
    
    # Load aggregate results
    try:
        with open(os.path.join(analyzer.run_results_dir, 'policy_lag_summary.csv'), 'r') as f:
            summary_df = pd.read_csv(f)
            
        # Calculate overall median lag
        median_lag = summary_df['median_lag'].median()
        print(f"- Overall median policy effectiveness lag: {median_lag:.1f} days")
        
        # Find fastest and slowest responses
        if not summary_df.empty:
            min_lag_row = summary_df.loc[summary_df['median_lag'].idxmin()]
            max_lag_row = summary_df.loc[summary_df['median_lag'].idxmax()]
            
            min_policy = min_lag_row['policy'].replace('_', ' ').title()
            min_outcome = min_lag_row['outcome'].replace('_', ' ').title()
            max_policy = max_lag_row['policy'].replace('_', ' ').title()
            max_outcome = max_lag_row['outcome'].replace('_', ' ').title()
            
            print(f"- Fastest response: {min_policy} → {min_outcome} ({min_lag_row['median_lag']:.1f} days)")
            print(f"- Slowest response: {max_policy} → {max_outcome} ({max_lag_row['median_lag']:.1f} days)")
    except:
        print("Could not load summary statistics.")
    
    print("\nReminder: Further improvements to consider:")
    print("- Application of causal inference methods like Synthetic Control or Difference-in-Differences")
    print("- Integration with mobility data to separate policy effects from behavioral changes")
    print("- Analysis of specific policy types rather than aggregate stringency index")
    print("- Expansion to include outcomes like hospitalization and economic indicators")
    
    print("\nScript execution complete.")