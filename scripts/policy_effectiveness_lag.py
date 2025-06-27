"""
Policy Effectiveness Lag Analysis

This module provides comprehensive analysis of the temporal relationship between
COVID-19 policy interventions (measured by the stringency index) and epidemiological outcomes.
It implements multiple advanced time-series methods to quantify the lag between
policy changes and their observable effects on key metrics like case rates, death rates,
and reproduction numbers.

Enhanced with causal inference techniques and regional analysis for more robust
identification of policy effects, particularly for high-quality data subsets.

Key features:
1. Robust time-series preprocessing with stationarity testing and transformation
2. Multiple methodologies for lag identification:
   - Cross-correlation function (CCF) analysis
   - Granger causality testing with statistical significance
   - Transfer function modeling
   - Wavelet coherence analysis for time-varying relationships
3. Causal inference techniques:
   - Difference-in-differences analysis
   - Synthetic control methods
   - Regression discontinuity design around policy changes
4. Regional analysis focusing on high-quality data subsets (US states, European regions)
5. Policy decomposition to analyze specific interventions (masks, lockdowns, etc.)
6. Comprehensive country-level and aggregated multi-country analysis
7. Statistical validation of identified lags with confidence intervals
8. Visualization and reporting capabilities for research publication

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
    OPTIONAL_PACKAGES_AVAILABLE = True
except ImportError:
    print("Note: Some optional visualization packages are missing. Basic functionality will still work.")
    OPTIONAL_PACKAGES_AVAILABLE = False

# Try to import causal inference packages
try:
    import causalinference
    from causalinference import CausalModel
    CAUSALINFERENCE_AVAILABLE = True
except ImportError:
    print("Note: CausalInference package not available. Causal inference analyses will be limited.")
    CAUSALINFERENCE_AVAILABLE = False

# Try to import synthetic control packages
try:
    from synth import Synth
    SYNTH_AVAILABLE = True
except ImportError:
    print("Note: Synthetic control package not available. Synthetic control analyses will not be available.")
    SYNTH_AVAILABLE = False

# Try to import fixed effects regression packages
try:
    import linearmodels
    from linearmodels.panel import PanelOLS
    LINEARMODELS_AVAILABLE = True
except ImportError:
    print("Note: LinearModels package not available. Panel regression analyses will be limited.")
    LINEARMODELS_AVAILABLE = False

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
    between policy stringency and outcomes, with robust statistical validation,
    causal inference techniques, and visualization capabilities for research publication.
    
    Enhanced with regional analysis capabilities and policy decomposition for analyzing
    specific intervention types.
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
                 subperiod_length=90,  # For analyzing time-varying relationships
                 use_causal_inference=True,  # Added for causal inference
                 regional_analysis=True,  # Added for regional analysis
                 high_quality_regions=None,  # Added for focusing on high-quality data regions
                 decompose_policies=False,  # Added for policy decomposition
                 policy_components=['masks', 'stay_at_home', 'business_closures', 'travel_restrictions']):  # Added policy types
        """
        Initialize the PolicyLagAnalyzer with configurable parameters for analysis.
        
        Parameters:
        -----------
        policy_columns : list
            Column names for policy indicators (usually stringency index).
        outcome_columns : list
            Column names for outcome measures (e.g., cases, deaths, reproduction rate).
        countries : list or None
            Countries to analyze. If None, all available countries will be used.
        max_lag : int
            Maximum lag (in days) to consider between policy and outcome.
        min_data_points : int
            Minimum number of days with valid data required for including a country.
        stationarity_transform : str
            Transformation to apply for achieving stationarity ('diff', 'log_diff', 'none').
        significance_level : float
            P-value threshold for statistical significance.
        rolling_window_sizes : list
            Window sizes for rolling averages to reduce noise.
        detrend_data : bool
            Whether to remove trends from time series.
        analyze_subperiods : bool
            Whether to analyze time-varying relationships by period.
        subperiod_length : int
            Length of subperiods for time-varying analysis.
        use_causal_inference : bool
            Whether to apply causal inference methods.
        regional_analysis : bool
            Whether to analyze regions within countries (e.g., US states).
        high_quality_regions : list or None
            List of specific regions with high-quality data to focus on.
        decompose_policies : bool
            Whether to analyze specific policy components separately.
        policy_components : list
            Specific policy types to analyze when decompose_policies is True.
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
        
        # New parameters for enhanced analysis
        self.use_causal_inference = use_causal_inference
        self.regional_analysis = regional_analysis
        self.high_quality_regions = high_quality_regions
        self.decompose_policies = decompose_policies
        self.policy_components = policy_components
        
        # Check package availability
        if self.use_causal_inference and not CAUSALINFERENCE_AVAILABLE:
            print("Warning: Causal inference requested but required packages not available.")
            self.use_causal_inference = False
            
        # Runtime properties
        self.data = None
        self.results = {}
        self.visualizations = {}
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(RESULTS_BASE_DIR, self.run_timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

    def difference_in_differences(self, data, treatment_col, outcome_col, time_col, unit_col, treatment_time):
        """
        Perform difference-in-differences analysis to estimate causal effects.
        
        Parameters:
        -----------
        data : DataFrame
            Panel data with units, time, treatment, and outcome.
        treatment_col : str
            Column name for treatment indicator.
        outcome_col : str
            Column name for outcome measure.
        time_col : str
            Column name for time variable.
        unit_col : str
            Column name for unit identifiers.
        treatment_time : int/date
            Time point when treatment begins.
            
        Returns:
        --------
        dict
            Results of difference-in-differences analysis.
        """
        if not CAUSALINFERENCE_AVAILABLE:
            return {'error': 'CausalInference package not available'}
            
        try:
            # Create treatment indicator
            data = data.copy()
            data['post_treatment'] = (data[time_col] >= treatment_time).astype(int)
            data['did'] = data[treatment_col] * data['post_treatment']
            
            # Prepare model
            Y = data[outcome_col].values
            D = data[treatment_col].values
            X = sm.add_constant(
                np.column_stack((
                    data['post_treatment'].values,
                    data['did'].values
                ))
            )
            
            # Fit OLS with clustered standard errors
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': data[unit_col]})
            
            # Extract results
            did_effect = results.params[2]  # Coefficient on the interaction term
            p_value = results.pvalues[2]
            ci_lower, ci_upper = results.conf_int().loc[2]
            
            return {
                'effect_size': did_effect,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'model_summary': results.summary().as_text()
            }
        except Exception as e:
            return {'error': f"Failed to perform difference-in-differences: {str(e)}"}
            
    def synthetic_control(self, data, treatment_unit, outcome_col, unit_col, time_col, treatment_time, predictors):
        """
        Perform synthetic control analysis for causal inference.
        
        Parameters:
        -----------
        data : DataFrame
            Panel data with units, time, and outcome.
        treatment_unit : str/int
            Identifier for the treated unit.
        outcome_col : str
            Column name for outcome measure.
        unit_col : str
            Column name for unit identifiers.
        time_col : str
            Column name for time variable.
        treatment_time : int/date
            Time point when treatment begins.
        predictors : list
            Column names for predictor variables.
            
        Returns:
        --------
        dict
            Results of synthetic control analysis.
        """
        if not SYNTH_AVAILABLE:
            return {'error': 'Synthetic control package not available'}
            
        try:
            # Prepare data for synthetic control
            pre_treatment = data[data[time_col] < treatment_time]
            post_treatment = data[data[time_col] >= treatment_time]
            
            # Create synthetic control object
            synth = Synth(
                data=data,
                outcome=outcome_col,
                unit_col=unit_col,
                time_col=time_col,
                treatment_unit=treatment_unit,
                treatment_time=treatment_time,
                predictors=predictors
            )
            
            # Fit synthetic control
            synth.fit()
            
            # Calculate treatment effect
            att = synth.att()
            
            # Calculate p-value using placebo test
            p_value = synth.pvalue(method='placebo')
            
            return {
                'att': att,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'weights': synth.weights,
                'pre_treatment_fit': synth.mspe(),
                'synth_object': synth
            }
        except Exception as e:
            return {'error': f"Failed to perform synthetic control analysis: {str(e)}"}
            
    def fixed_effects_regression(self, data, outcome_col, policy_col, unit_col, time_col, controls=None):
        """
        Perform panel fixed effects regression to estimate policy effects.
        
        Parameters:
        -----------
        data : DataFrame
            Panel data with units, time, policy, and outcome.
        outcome_col : str
            Column name for outcome measure.
        policy_col : str
            Column name for policy measure.
        unit_col : str
            Column name for unit identifiers.
        time_col : str
            Column name for time variable.
        controls : list, optional
            Column names for control variables.
            
        Returns:
        --------
        dict
            Results of fixed effects regression.
        """
        if not LINEARMODELS_AVAILABLE:
            return {'error': 'LinearModels package not available'}
            
        try:
            # Prepare panel data
            data = data.copy()
            data = data.set_index([unit_col, time_col])
            
            # Define formula
            exog_vars = [policy_col]
            if controls:
                exog_vars.extend(controls)
                
            exog = sm.add_constant(data[exog_vars])
            
            # Fit fixed effects model
            model = PanelOLS(
                data[outcome_col], 
                exog,
                entity_effects=True,
                time_effects=True
            )
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            # Extract results
            policy_effect = results.params[policy_col]
            p_value = results.pvalues[policy_col]
            
            return {
                'effect_size': policy_effect,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'model_summary': results.summary.as_text()
            }
        except Exception as e:
            return {'error': f"Failed to perform fixed effects regression: {str(e)}"}
            
    def analyze_regional_policy_effects(self, data, regions, policy_col, outcome_col, date_col, region_col, 
                                      max_lag=30, methods=None):
        """
        Analyze policy effects for specific regions with high-quality data.
        
        Parameters:
        -----------
        data : DataFrame
            Data containing regional COVID and policy information.
        regions : list
            List of regions to analyze.
        policy_col : str
            Column name for policy measure.
        outcome_col : str
            Column name for outcome measure.
        date_col : str
            Column name for date.
        region_col : str
            Column name for region identifier.
        max_lag : int
            Maximum lag to consider.
        methods : list, optional
            Specific analysis methods to use.
            
        Returns:
        --------
        dict
            Results of regional policy analysis by region.
        """
        if methods is None:
            methods = ['ccf', 'granger', 'wavelet']
            
            if self.use_causal_inference:
                methods.extend(['did', 'synthetic_control', 'fixed_effects'])
                
        results = {}
        
        for region in tqdm(regions, desc="Analyzing regions"):
            region_data = data[data[region_col] == region].copy()
            
            if len(region_data) < self.min_data_points:
                results[region] = {'error': f"Insufficient data points ({len(region_data)})" }
                continue
                
            region_results = {}
            
            # Time series analyses
            if 'ccf' in methods:
                region_results['ccf'] = self._analyze_ccf(
                    region_data[policy_col].values,
                    region_data[outcome_col].values,
                    max_lag=max_lag
                )
                
            if 'granger' in methods:
                region_results['granger'] = self._analyze_granger_causality(
                    region_data[policy_col].values,
                    region_data[outcome_col].values,
                    max_lag=max_lag
                )
                
            if 'wavelet' in methods and OPTIONAL_PACKAGES_AVAILABLE:
                region_results['wavelet'] = self._analyze_wavelet_coherence(
                    region_data[policy_col].values,
                    region_data[outcome_col].values
                )
                
            # Causal inference methods
            if 'did' in methods and self.use_causal_inference:
                # Find significant policy changes as treatment points
                policy_changes = self._identify_policy_changes(region_data, policy_col)
                
                if policy_changes:
                    treatment_time = policy_changes[0]['date']
                    region_data['treatment'] = (region_data[date_col] >= treatment_time).astype(int)
                    
                    # Need to create a comparison group - often neighboring regions
                    # This is just a placeholder - actual implementation would require
                    # gathering data from comparable regions
                    # ...
                    
            if 'synthetic_control' in methods and self.use_causal_inference and SYNTH_AVAILABLE:
                # Synthetic control requires multiple control units
                # Implementation would require gathering data from all potential control regions
                # ...
                pass
                
            results[region] = region_results
            
        return results
        
    def decompose_policy_analysis(self, data, policy_components, outcome_col, date_col, region_col):
        """
        Analyze effects of specific policy components instead of aggregate stringency.
        
        Parameters:
        -----------
        data : DataFrame
            Data containing policy component indicators and outcomes.
        policy_components : list
            List of column names for specific policy measures.
        outcome_col : str
            Column name for outcome measure.
        date_col : str
            Column name for date.
        region_col : str
            Column name for region/country identifier.
            
        Returns:
        --------
        dict
            Results of decomposed policy analysis by policy component.
        """
        results = {}
        
        for policy in policy_components:
            if policy not in data.columns:
                results[policy] = {'error': f"Policy component {policy} not found in data"}
                continue
                
            policy_results = {}
            
            # Group by region/country
            for region, group in data.groupby(region_col):
                if len(group) < self.min_data_points:
                    continue
                    
                # Analyze lag using CCF
                ccf_result = self._analyze_ccf(
                    group[policy].values,
                    group[outcome_col].values,
                    max_lag=self.max_lag
                )
                
                # Analyze using Granger causality
                granger_result = self._analyze_granger_causality(
                    group[policy].values, 
                    group[outcome_col].values,
                    max_lag=self.max_lag
                )
                
                policy_results[region] = {
                    'ccf': ccf_result,
                    'granger': granger_result
                }
                
            # Aggregate results across regions
            significant_lags = [
                r['ccf']['best_lag'] for region, r in policy_results.items()
                if r['ccf']['significant']
            ]
            
            if significant_lags:
                median_lag = np.median(significant_lags)
                mean_lag = np.mean(significant_lags)
                
                results[policy] = {
                    'median_lag': median_lag,
                    'mean_lag': mean_lag,
                    'regions_with_significant_effect': sum(1 for r in policy_results.values() if r['ccf']['significant']),
                    'total_regions': len(policy_results),
                    'detailed_results': policy_results
                }
            else:
                results[policy] = {
                    'significant_effect': False,
                    'regions_with_significant_effect': 0,
                    'total_regions': len(policy_results),
                    'detailed_results': policy_results
                }
                
        return results
        
    def _identify_policy_changes(self, data, policy_col, threshold_percentile=90):
        """
        Identify significant changes in policy stringency as potential treatment points.
        
        Parameters:
        -----------
        data : DataFrame
            Time series data for a single region.
        policy_col : str
            Column name for policy measure.
        threshold_percentile : int
            Percentile to use as threshold for significant changes.
            
        Returns:
        --------
        list
            List of dictionaries containing information about significant policy changes.
        """
        # Calculate day-to-day changes
        data = data.copy().sort_values('date')
        data['policy_change'] = data[policy_col].diff()
        
        # Find significant increases
        threshold = np.percentile(data['policy_change'].abs(), threshold_percentile)
        significant_changes = data[data['policy_change'].abs() >= threshold].copy()
        
        # Format results
        changes = []
        for _, row in significant_changes.iterrows():
            changes.append({
                'date': row['date'],
                'magnitude': row['policy_change'],
                'pre_level': row[policy_col] - row['policy_change'],
                'post_level': row[policy_col]
            })
            
        return sorted(changes, key=lambda x: abs(x['magnitude']), reverse=True)

    # ...existing time series methods (_analyze_ccf, _analyze_granger_causality, etc.)...
                
    def analyze_policy_effects(self, data_path, output_dir=None):
        """
        Comprehensive analysis of policy effects using multiple methodologies,
        including both time-series analysis and causal inference approaches.
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file containing COVID data.
        output_dir : str, optional
            Directory to save results. If None, uses the default run directory.
            
        Returns:
        --------
        dict
            Comprehensive results of policy effectiveness analysis.
        """
        # ...existing code for loading data...
        
        # Add regional analysis if requested
        if self.regional_analysis:
            print("Performing regional analysis...")
            
            # Identify regions with high-quality data
            high_quality_regions = self.high_quality_regions or self._identify_high_quality_regions()
            
            # Analyze regional policy effects
            regional_results = self.analyze_regional_policy_effects(
                self.data,
                high_quality_regions,
                self.policy_columns[0],
                self.outcome_columns[0],
                'date',
                'region'
            )
            
            self.results['regional_analysis'] = regional_results
            
        # Add policy decomposition if requested
        if self.decompose_policies and all(p in self.data.columns for p in self.policy_components):
            print("Performing policy decomposition analysis...")
            
            policy_component_results = self.decompose_policy_analysis(
                self.data,
                self.policy_components,
                self.outcome_columns[0],
                'date',
                'location'
            )
            
            self.results['policy_component_analysis'] = policy_component_results
            
        # ...existing code for visualization and results saving...
        
        return self.results
        
    def _identify_high_quality_regions(self, min_data_completeness=0.8):
        """
        Identify regions with high-quality data based on completeness and consistency.
        
        Parameters:
        -----------
        min_data_completeness : float
            Minimum ratio of non-missing values required.
            
        Returns:
        --------
        list
            List of regions with high-quality data.
        """
        quality_metrics = {}
        
        for region, group in self.data.groupby('region'):
            # Calculate data completeness
            completeness = 1 - (group[self.policy_columns + self.outcome_columns].isna().sum().sum() / 
                              (len(group) * (len(self.policy_columns) + len(self.outcome_columns))))
                              
            # Check for reporting consistency (e.g., no sudden jumps due to backlog reporting)
            consistency_score = 0
            for col in self.outcome_columns:
                if col in group.columns:
                    # Calculate day-to-day percent changes
                    pct_changes = group[col].pct_change().abs()
                    # Count extreme changes (>200%)
                    extreme_changes = (pct_changes > 2).sum() / len(group)
                    # Higher score means more consistent (fewer extreme changes)
                    consistency_score += 1 - extreme_changes
                    
            consistency_score /= len(self.outcome_columns)
            
            # Combine metrics
            quality_score = completeness * 0.6 + consistency_score * 0.4
            
            quality_metrics[region] = {
                'completeness': completeness,
                'consistency': consistency_score,
                'quality_score': quality_score
            }
            
        # Filter regions by quality threshold
        high_quality_regions = [
            region for region, metrics in quality_metrics.items()
            if metrics['quality_score'] >= 0.7 and metrics['completeness'] >= min_data_completeness
        ]
        
        print(f"Identified {len(high_quality_regions)} high-quality regions out of {len(quality_metrics)} total regions")
        return high_quality_regions