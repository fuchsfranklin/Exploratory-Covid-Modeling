"""
Pandemic Fatigue Analysis

This module aims to identify and potentially forecast periods of "pandemic fatigue,"
where public compliance with restrictions may decrease despite ongoing risks,
potentially using changes in `positive_rate` or `new_cases_per_test` as proxies,
relative to stringency index and vaccination rates.

Methods:
- Analyze proxies relative to stringency and vaccination.
- Plot indicators over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Ensure output directories exist
os.makedirs('eda_outputs/per_country', exist_ok=True)

class PandemicFatigueAnalyzer:
    """Analyzes indicators potentially related to pandemic fatigue."""

    def __init__(self, data_path='owid-covid-data.csv'):
        """
        Initialize with the path to the data file.

        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing the data.
        """
        try:
            self.data = pd.read_csv(data_path, parse_dates=['date'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # Identify available relevant columns
        self.required_cols = ['date', 'location', 'stringency_index', 'new_cases']
        self.proxy_cols = ['positive_rate', 'new_tests'] # Used to calculate proxy if needed
        self.vaccination_cols = ['people_vaccinated_per_hundred']

        self.available_features = [col for col in self.required_cols + self.proxy_cols + self.vaccination_cols if col in self.data.columns]

        missing_required = [col for col in self.required_cols if col not in self.available_features]
        if missing_required:
             raise ValueError(f"Data is missing essential columns: {missing_required}")

        # Determine the best available fatigue proxy
        self.fatigue_proxy_col = None
        if 'positive_rate' in self.available_features:
            self.fatigue_proxy_col = 'positive_rate'
        elif 'new_cases' in self.available_features and 'new_tests' in self.available_features:
            # Calculate cases per test if positive_rate is missing
            self.data['new_tests'] = self.data['new_tests'].replace(0, np.nan) # Avoid division by zero
            self.data['cases_per_test'] = (self.data['new_cases'] / self.data['new_tests']).clip(0, 1) # Clip unreasonable values
            self.fatigue_proxy_col = 'cases_per_test'
            if 'cases_per_test' not in self.available_features:
                self.available_features.append('cases_per_test')
        else:
             print("Warning: No suitable fatigue proxy column ('positive_rate' or 'new_cases'/'new_tests') found.")

        self.data = self.data.sort_values(['location', 'date'])
        self.scaler = MinMaxScaler()

    def preprocess_country_data(self, country, smooth_window=14):
        """
        Preprocess data for fatigue analysis for a specific country.

        Parameters:
        -----------
        country : str
        smooth_window : int, default=14

        Returns:
        --------
        pandas.DataFrame or None
        """
        country_data = self.data[self.data['location'] == country].copy()

        if len(country_data) < smooth_window * 2:
            print(f"Warning: Insufficient data for {country}. Skipping.")
            return None

        # Columns to impute and smooth
        cols_to_process = ['stringency_index', 'new_cases']
        if self.fatigue_proxy_col:
             cols_to_process.append(self.fatigue_proxy_col)
        if 'people_vaccinated_per_hundred' in self.available_features:
             cols_to_process.append('people_vaccinated_per_hundred')

        for col in cols_to_process:
            if col not in country_data.columns: continue # Skip if column somehow missing for this country
            # Impute: Forward fill, backward fill, then 0
            country_data[col] = country_data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            # Smooth
            country_data[f'smoothed_{col}'] = country_data[col].rolling(
                window=smooth_window, min_periods=1, center=True).mean()

        # Drop rows with NaNs potentially introduced by smoothing start/end
        country_data = country_data.dropna(subset=[f'smoothed_{col}' for col in cols_to_process if f'smoothed_{col}' in country_data.columns])

        if len(country_data) < smooth_window:
             print(f"Warning: Insufficient data for {country} after smoothing. Skipping.")
             return None

        # Normalize smoothed data for plotting comparison
        norm_cols_in_data = [f'smoothed_{col}' for col in cols_to_process if f'smoothed_{col}' in country_data.columns]
        try:
            country_data[[f'normalized_{col}' for col in norm_cols_in_data]] = self.scaler.fit_transform(country_data[norm_cols_in_data])
        except ValueError as e:
            print(f"Warning: Normalization failed for {country}. Skipping. Error: {e}")
            return None

        return country_data

    def analyze_fatigue_indicators(self, countries, plot=True):
        """
        Analyze and plot potential indicators of pandemic fatigue for multiple countries.

        Parameters:
        -----------
        countries : list
        plot : bool, default=True
        """
        all_fatigue_periods = {}
        for country in countries:
            print(f"Analyzing fatigue indicators for {country}...")
            country_data = self.preprocess_country_data(country)

            if country_data is None or self.fatigue_proxy_col is None:
                print(f"Skipping {country} due to insufficient data or missing proxy.")
                continue

            # --- Simple Fatigue Proxy Logic ---
            # Define 'high' stringency (e.g., above 75th percentile for the country)
            high_stringency_threshold = country_data['smoothed_stringency_index'].quantile(0.75)
            # Define 'high' fatigue proxy (e.g., above 75th percentile)
            proxy_col_smoothed = f'smoothed_{self.fatigue_proxy_col}'
            high_proxy_threshold = country_data[proxy_col_smoothed].quantile(0.75)

            # Identify periods where BOTH stringency AND the proxy are 'high'
            fatigue_condition = (country_data['smoothed_stringency_index'] >= high_stringency_threshold) & \
                                (country_data[proxy_col_smoothed] >= high_proxy_threshold)

            # Optional: Add condition for non-decreasing cases?
            # fatigue_condition &= (country_data['smoothed_new_cases'].diff().fillna(0) >= 0)

            potential_fatigue_periods = country_data[fatigue_condition]
            all_fatigue_periods[country] = potential_fatigue_periods
            print(f"{country}: Found {len(potential_fatigue_periods)} potential fatigue points.")


            if plot:
                self._plot_fatigue_indicators(country_data, potential_fatigue_periods, high_stringency_threshold, high_proxy_threshold)

        return all_fatigue_periods


    def _plot_fatigue_indicators(self, country_data, fatigue_periods, high_stringency_threshold, high_proxy_threshold):
        """Plot indicators for a single country."""
        country = country_data['location'].iloc[0]
        fig, ax1 = plt.subplots(figsize=(14, 7))
        fig.suptitle(f'Pandemic Fatigue Indicators for {country}', fontsize=16)

        # Plot Stringency (Smoothed)
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Smoothed Stringency Index', color=color)
        ax1.plot(country_data['date'], country_data['smoothed_stringency_index'], color=color, label='Stringency Index', alpha=0.8)
        ax1.axhline(high_stringency_threshold, color=color, linestyle=':', alpha=0.7, label=f'High Stringency Threshold ({high_stringency_threshold:.1f})')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)

        # Plot Fatigue Proxy (Smoothed) on secondary axis
        ax2 = ax1.twinx()
        proxy_col_smoothed = f'smoothed_{self.fatigue_proxy_col}'
        proxy_label = self.fatigue_proxy_col.replace('_', ' ').title()
        color = 'tab:red'
        ax2.set_ylabel(f'Smoothed {proxy_label}', color=color)
        ax2.plot(country_data['date'], country_data[proxy_col_smoothed], color=color, label=proxy_label, alpha=0.8)
        ax2.axhline(high_proxy_threshold, color=color, linestyle=':', alpha=0.7, label=f'High {proxy_label} Threshold ({high_proxy_threshold:.2f})')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        # Highlight potential fatigue periods
        if not fatigue_periods.empty:
             # Use scatter on ax1 to ensure it's layered correctly
             ax1.scatter(fatigue_periods['date'], fatigue_periods['smoothed_stringency_index'],
                         color='orange', marker='o', s=40, label='Potential Fatigue Period', zorder=5, alpha=0.7)
             # Update legend on ax1 to include the scatter plot marker
             handles, labels = ax1.get_legend_handles_labels()
             from matplotlib.lines import Line2D
             # Check if label already exists to avoid duplicates if function is called multiple times
             if 'Potential Fatigue Period' not in labels:
                 handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, alpha=0.7))
                 labels.append('Potential Fatigue Period')
                 ax1.legend(handles, labels, loc='upper left')


        # Optional: Add Smoothed New Cases on a third axis
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60)) # Offset the right spine
        color = 'tab:green'
        ax3.set_ylabel('Smoothed New Cases', color=color)
        ax3.plot(country_data['date'], country_data['smoothed_new_cases'], color=color, alpha=0.5, linestyle='--', label='New Cases')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.legend(loc='lower right')


        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        # Save the figure
        output_path = f'eda_outputs/per_country/fatigue_indicators_{country.replace(" ", "_")}.png'
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_path}")
        except Exception as e:
            print(f"Error saving plot {output_path}: {e}")
        plt.close(fig)

    def forecast_fatigue(self):
        """Placeholder for forecasting logic."""
        print("Forecasting functionality not yet implemented.")
        pass

# Example usage
if __name__ == "__main__":
    try:
        analyzer = PandemicFatigueAnalyzer(data_path='owid-covid-data.csv')

        # Select countries for analysis
        countries_to_analyze = [
            'United States', 'Germany', 'France', 'Italy', 'United Kingdom',
            'Brazil', 'India', 'Canada', 'Spain', 'Sweden'
        ]

        print("\n--- Analyzing Pandemic Fatigue Indicators ---")
        fatigue_results = analyzer.analyze_fatigue_indicators(countries_to_analyze, plot=True)

        # Example: Print dates for potential fatigue in one country
        if 'United States' in fatigue_results and not fatigue_results['United States'].empty:
            print("\nPotential Fatigue Dates for United States (example):")
            print(fatigue_results['United States']['date'].dt.date.head())

        # Call placeholder forecast function
        analyzer.forecast_fatigue()

        print("\nAnalysis complete. Plots saved in 'eda_outputs/per_country/'.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'owid-covid-data.csv' is in the root directory.")
    except ValueError as e:
        print(f"Data Error: {e}")
    except ImportError as e:
         print(f"Import Error: {e}. Please install missing packages (e.g., pip install matplotlib scikit-learn)")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()