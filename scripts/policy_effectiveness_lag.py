"""
Policy Effectiveness Lag Analysis

Analyzes the temporal relationship between COVID-19 policy interventions
(stringency index) and epidemiological outcomes (cases, deaths, reproduction rate).

Methods implemented:
1. Cross-correlation function (CCF) analysis
2. Granger causality testing
3. Wavelet coherence analysis for time-varying relationships

Produces per-country and aggregate results with statistical validation.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy import signal, stats
import os
import json
import datetime
import warnings
from tqdm import tqdm

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Note: PyWavelets not available. Wavelet coherence will be skipped.")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", "The iteration is not making good progress")

RESULTS_BASE_DIR = "results/policy_effectiveness"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


class PolicyLagAnalyzer:
    """
    Quantifies the time lag between policy interventions and epidemiological
    outcomes using cross-correlation, Granger causality, and wavelet coherence.
    """

    def __init__(self,
                 policy_col='stringency_index',
                 outcome_columns=None,
                 countries=None,
                 max_lag=30,
                 min_data_points=180,
                 significance_level=0.05):
        """
        Args:
            policy_col: Column name for the policy indicator.
            outcome_columns: Outcome measure column names.
            countries: Countries to analyze. None = auto-select from data.
            max_lag: Maximum lag in days to test.
            min_data_points: Minimum valid data points required per country.
            significance_level: P-value threshold for statistical significance.
        """
        self.policy_col = policy_col
        self.outcome_columns = outcome_columns or [
            'new_cases_smoothed_per_million',
            'new_deaths_smoothed_per_million',
            'reproduction_rate'
        ]
        self.countries = countries
        self.max_lag = max_lag
        self.min_data_points = min_data_points
        self.significance_level = significance_level

        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(RESULTS_BASE_DIR, f"{self.run_timestamp}_analysis")
        os.makedirs(self.run_dir, exist_ok=True)

        self.data = None
        self.results = {}

    # ------------------------------------------------------------------
    # Data loading & preparation
    # ------------------------------------------------------------------

    def load_data(self, csv_path):
        """Load OWID data and select countries with sufficient data."""
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df.sort_values(['location', 'date'], inplace=True)

        # Filter to real countries (exclude aggregates like 'World', 'Europe', etc.)
        aggregates = {'World', 'Europe', 'European Union', 'Asia', 'Africa',
                      'North America', 'South America', 'Oceania',
                      'High income', 'Low income', 'Lower middle income',
                      'Upper middle income', 'International'}
        df = df[~df['location'].isin(aggregates)].copy()

        # Auto-select countries if not specified
        if self.countries is None:
            self.countries = self._select_countries(df)

        self.data = df[df['location'].isin(self.countries)].copy()
        print(f"Loaded data for {len(self.countries)} countries: {', '.join(self.countries)}")

    def _select_countries(self, df):
        """Select countries that have enough data for all required columns."""
        required_cols = [self.policy_col] + self.outcome_columns
        valid_countries = []

        for country, group in df.groupby('location'):
            valid_rows = group[required_cols].dropna(how='any')
            if len(valid_rows) >= self.min_data_points:
                valid_countries.append(country)

        print(f"Found {len(valid_countries)} countries with >= {self.min_data_points} "
              f"complete data points across all required columns.")
        return valid_countries

    def _make_stationary(self, series):
        """First-difference a series to achieve stationarity. Returns differenced series."""
        diffed = series.diff().dropna()
        return diffed

    def _prepare_country_series(self, country):
        """
        Extract and prepare policy + outcome series for a single country.
        Returns dict of {outcome_col: (policy_series, outcome_series)} with
        aligned, stationary, NaN-free series.
        """
        cdf = self.data[self.data['location'] == country].copy()
        cdf = cdf.sort_values('date').set_index('date')

        pairs = {}
        for outcome_col in self.outcome_columns:
            subset = cdf[[self.policy_col, outcome_col]].dropna()
            if len(subset) < self.min_data_points:
                continue

            # Differencing for stationarity
            policy_diff = self._make_stationary(subset[self.policy_col])
            outcome_diff = self._make_stationary(subset[outcome_col])

            # Align after differencing
            aligned = pd.concat([policy_diff, outcome_diff], axis=1).dropna()
            if len(aligned) < 60:  # Need enough points after differencing
                continue

            pairs[outcome_col] = (aligned[self.policy_col], aligned[outcome_col])

        return pairs

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _analyze_ccf(self, policy_series, outcome_series):
        """
        Cross-correlation analysis between policy and outcome at various lags.

        Returns dict with best_lag, best_correlation, all correlations, and
        whether the result is statistically significant.
        """
        n = len(policy_series)
        policy_vals = policy_series.values
        outcome_vals = outcome_series.values

        # Normalize
        policy_norm = (policy_vals - policy_vals.mean()) / (policy_vals.std() + 1e-10)
        outcome_norm = (outcome_vals - outcome_vals.mean()) / (outcome_vals.std() + 1e-10)

        correlations = {}
        for lag in range(0, min(self.max_lag + 1, n // 3)):
            if lag == 0:
                corr = np.corrcoef(policy_norm, outcome_norm)[0, 1]
            else:
                corr = np.corrcoef(policy_norm[:-lag], outcome_norm[lag:])[0, 1]
            correlations[lag] = corr

        if not correlations:
            return {'significant': False, 'error': 'No valid lags computed'}

        # Find the lag with the strongest negative correlation
        # (policy increase should reduce outcome)
        best_lag = min(correlations, key=correlations.get)
        best_corr = correlations[best_lag]

        # Significance: Bartlett's approximation for confidence bounds
        conf_bound = 1.96 / np.sqrt(n)
        significant = abs(best_corr) > conf_bound

        return {
            'best_lag': int(best_lag),
            'best_correlation': round(float(best_corr), 4),
            'significant': bool(significant),
            'confidence_bound': round(float(conf_bound), 4),
            'correlations_by_lag': {int(k): round(float(v), 4) for k, v in correlations.items()}
        }

    def _analyze_granger(self, policy_series, outcome_series):
        """
        Granger causality test: does the policy series help predict the outcome?

        Tests multiple lag orders up to max_lag (capped for data size).
        Returns the best lag order and its p-value.
        """
        combined = pd.DataFrame({
            'outcome': outcome_series.values,
            'policy': policy_series.values
        })

        max_test_lag = min(self.max_lag, len(combined) // 5, 15)
        if max_test_lag < 1:
            return {'significant': False, 'error': 'Insufficient data for Granger test'}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(combined[['outcome', 'policy']], maxlag=max_test_lag, verbose=False)
        except Exception as e:
            return {'significant': False, 'error': str(e)}

        # Extract the best (lowest p-value) lag
        best_lag = None
        best_pvalue = 1.0
        lag_results = {}

        for lag_order, result in results.items():
            # Use the F-test p-value
            f_test = result[0]['ssr_ftest']
            p_value = f_test[1]
            lag_results[int(lag_order)] = round(float(p_value), 6)

            if p_value < best_pvalue:
                best_pvalue = p_value
                best_lag = lag_order

        return {
            'best_lag': int(best_lag) if best_lag else None,
            'best_pvalue': round(float(best_pvalue), 6),
            'significant': bool(best_pvalue < self.significance_level),
            'pvalues_by_lag': lag_results
        }

    def _analyze_wavelet_coherence(self, policy_series, outcome_series):
        """
        Wavelet coherence analysis to identify time-varying lag relationships.

        Uses continuous wavelet transform to find frequency-dependent coherence
        between policy and outcome series.
        """
        if not PYWT_AVAILABLE:
            return {'significant': False, 'error': 'PyWavelets not installed'}

        policy_vals = policy_series.values.astype(float)
        outcome_vals = outcome_series.values.astype(float)
        n = len(policy_vals)

        # Standardize
        policy_std = (policy_vals - policy_vals.mean()) / (policy_vals.std() + 1e-10)
        outcome_std = (outcome_vals - outcome_vals.mean()) / (outcome_vals.std() + 1e-10)

        try:
            # Compute CWT for both series
            scales = np.arange(2, min(n // 4, 64))
            if len(scales) < 2:
                return {'significant': False, 'error': 'Series too short for wavelet analysis'}

            coef_policy, freqs_policy = pywt.cwt(policy_std, scales, 'morl')
            coef_outcome, freqs_outcome = pywt.cwt(outcome_std, scales, 'morl')

            # Cross-wavelet power
            cross_power = np.abs(coef_policy * np.conj(coef_outcome))

            # Wavelet coherence approximation via smoothed cross-spectrum
            smooth_window = max(3, n // 50)
            from scipy.ndimage import uniform_filter1d

            smooth_cross = uniform_filter1d(cross_power, size=smooth_window, axis=1)
            smooth_policy = uniform_filter1d(np.abs(coef_policy) ** 2, size=smooth_window, axis=1)
            smooth_outcome = uniform_filter1d(np.abs(coef_outcome) ** 2, size=smooth_window, axis=1)

            coherence = smooth_cross ** 2 / (smooth_policy * smooth_outcome + 1e-10)

            # Average coherence across time for each scale
            mean_coherence_by_scale = coherence.mean(axis=1)

            # Convert scales to approximate periods in days
            periods = 1.0 / (freqs_policy + 1e-10)

            # Find the scale/period band with highest coherence
            best_scale_idx = np.argmax(mean_coherence_by_scale)
            best_period = float(periods[best_scale_idx])
            best_coherence = float(mean_coherence_by_scale[best_scale_idx])

            # Phase difference at the best scale → approximate lag
            phase = np.angle(np.mean(coef_policy[best_scale_idx] * np.conj(coef_outcome[best_scale_idx])))
            estimated_lag_days = float(phase * best_period / (2 * np.pi))

            # Significance: coherence > 0.5 is a common threshold
            significant = best_coherence > 0.5

            return {
                'best_period_days': round(best_period, 1),
                'best_coherence': round(best_coherence, 4),
                'estimated_lag_days': round(abs(estimated_lag_days), 1),
                'significant': bool(significant),
                'coherence_by_period': {
                    round(float(p), 1): round(float(c), 4)
                    for p, c in zip(periods[::max(1, len(periods)//10)],
                                    mean_coherence_by_scale[::max(1, len(periods)//10)])
                }
            }
        except Exception as e:
            return {'significant': False, 'error': str(e)}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def analyze_country(self, country):
        """Run all analysis methods for a single country."""
        pairs = self._prepare_country_series(country)
        if not pairs:
            return {'error': f'Insufficient data for {country}'}

        country_results = {}
        for outcome_col, (policy_s, outcome_s) in pairs.items():
            pair_key = f"{self.policy_col}_vs_{outcome_col}"

            ccf_result = self._analyze_ccf(policy_s, outcome_s)
            granger_result = self._analyze_granger(policy_s, outcome_s)
            wavelet_result = self._analyze_wavelet_coherence(policy_s, outcome_s)

            # Consensus: combine lag estimates from significant methods
            lag_estimates = []
            methods_significant = 0
            methods_total = 3

            if ccf_result.get('significant') and ccf_result.get('best_lag') is not None:
                lag_estimates.append(ccf_result['best_lag'])
                methods_significant += 1
            if granger_result.get('significant') and granger_result.get('best_lag') is not None:
                lag_estimates.append(granger_result['best_lag'])
                methods_significant += 1
            if wavelet_result.get('significant') and wavelet_result.get('estimated_lag_days') is not None:
                lag_estimates.append(wavelet_result['estimated_lag_days'])
                methods_significant += 1

            consensus_lag = round(float(np.median(lag_estimates)), 1) if lag_estimates else None

            country_results[pair_key] = {
                'ccf': ccf_result,
                'granger': granger_result,
                'wavelet': wavelet_result,
                'consensus': {
                    'methods_significant': methods_significant,
                    'methods_total': methods_total,
                    'lag_estimates': [round(float(x), 1) for x in lag_estimates],
                    'consensus_lag_days': consensus_lag
                }
            }

        return country_results

    def run_analysis(self, csv_path):
        """
        Run the full analysis pipeline: load data, analyze each country,
        compute aggregate results, and save everything.
        """
        self.load_data(csv_path)

        all_results = {}
        for country in tqdm(self.countries, desc="Analyzing countries"):
            result = self.analyze_country(country)
            all_results[country] = result

            # Save per-country result
            country_file = os.path.join(self.run_dir, f"{country}_results.json")
            with open(country_file, 'w') as f:
                json.dump({'country': country, 'results': result}, f, indent=2)

        self.results = all_results

        # Compute and save aggregate results
        aggregate = self._compute_aggregate()
        agg_file = os.path.join(self.run_dir, 'aggregate_results.json')
        with open(agg_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        # Save run summary
        self._save_summary(aggregate)

        print(f"\nResults saved to: {self.run_dir}")
        return aggregate

    def _compute_aggregate(self):
        """Aggregate per-country results into cross-country summary."""
        aggregate = {
            'countries_analyzed': self.countries,
            'analysis_timestamp': self.run_timestamp,
            'pair_summaries': {}
        }

        # Collect all pair keys
        all_pair_keys = set()
        for country_result in self.results.values():
            if isinstance(country_result, dict) and 'error' not in country_result:
                all_pair_keys.update(country_result.keys())

        for pair_key in sorted(all_pair_keys):
            consensus_lags = []
            significant_countries = []
            country_details = {}

            for country in self.countries:
                cr = self.results.get(country, {})
                if isinstance(cr, dict) and pair_key in cr:
                    pair_result = cr[pair_key]
                    consensus = pair_result.get('consensus', {})
                    lag = consensus.get('consensus_lag_days')
                    n_sig = consensus.get('methods_significant', 0)

                    country_details[country] = {
                        'consensus_lag': lag,
                        'methods_significant': n_sig
                    }

                    if lag is not None and n_sig >= 1:
                        consensus_lags.append(lag)
                        significant_countries.append(country)

            summary = {
                'countries_with_significant_results': significant_countries,
                'n_significant': len(significant_countries),
                'n_total': len(self.countries),
                'country_details': country_details
            }

            if consensus_lags:
                summary['median_lag_days'] = round(float(np.median(consensus_lags)), 1)
                summary['mean_lag_days'] = round(float(np.mean(consensus_lags)), 1)
                summary['std_lag_days'] = round(float(np.std(consensus_lags)), 1)
                summary['min_lag_days'] = round(float(min(consensus_lags)), 1)
                summary['max_lag_days'] = round(float(max(consensus_lags)), 1)
            else:
                summary['median_lag_days'] = None

            aggregate['pair_summaries'][pair_key] = summary

        return aggregate

    def _save_summary(self, aggregate):
        """Save a human-readable summary text file."""
        lines = [
            "Policy Effectiveness Lag Analysis — Run Summary",
            "=" * 50,
            f"Run ID: {self.run_timestamp}",
            f"Countries analyzed: {len(self.countries)}",
            f"Methods: Cross-correlation, Granger causality, Wavelet coherence",
            f"Max lag tested: {self.max_lag} days",
            f"Significance level: {self.significance_level}",
            ""
        ]

        for pair_key, summary in aggregate.get('pair_summaries', {}).items():
            lines.append(f"\n--- {pair_key} ---")
            n_sig = summary.get('n_significant', 0)
            n_total = summary.get('n_total', 0)
            lines.append(f"Countries with significant results: {n_sig}/{n_total}")

            if summary.get('median_lag_days') is not None:
                lines.append(f"Median lag: {summary['median_lag_days']} days")
                lines.append(f"Mean lag: {summary['mean_lag_days']} days (std: {summary['std_lag_days']})")
                lines.append(f"Range: {summary['min_lag_days']} - {summary['max_lag_days']} days")

                for country in summary.get('countries_with_significant_results', []):
                    detail = summary['country_details'].get(country, {})
                    lines.append(f"  {country}: lag={detail.get('consensus_lag')} days "
                                 f"({detail.get('methods_significant')} methods significant)")
            else:
                lines.append("No significant lag relationships found.")

        summary_path = os.path.join(self.run_dir, 'run_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    analyzer = PolicyLagAnalyzer(
        countries=['United States', 'United Kingdom', 'Germany', 'France',
                   'Italy', 'Spain', 'Canada', 'Brazil', 'India', 'Sweden'],
        max_lag=30,
        min_data_points=180,
        significance_level=0.05
    )

    csv_path = 'owid-covid-data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
    else:
        results = analyzer.run_analysis(csv_path)
        print("\nDone.")
