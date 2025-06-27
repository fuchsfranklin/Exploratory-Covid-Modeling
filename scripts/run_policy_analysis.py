"""
Run the Policy Effectiveness Lag Analysis with enhanced causal inference techniques
and regional analysis for more robust identification of policy effects.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Make sure the script can find the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the policy lag analyzer
from policy_effectiveness_lag import PolicyLagAnalyzer

# Set up the results directory
results_dir = "../results/policy_effectiveness"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(results_dir, f"{run_id}_analysis")
os.makedirs(run_dir, exist_ok=True)

print(f"Starting policy effectiveness lag analysis run {run_id}")
print(f"Results will be saved to: {run_dir}")

# Define countries for analysis
countries = [
    'United States', 'Germany', 'United Kingdom', 'France',
    'Italy', 'Spain', 'Canada', 'Japan', 'South Korea', 'Brazil'
]

# Initialize the analyzer with enhanced settings
analyzer = PolicyLagAnalyzer(
    data_path='../owid-covid-data.csv',
    run_id=run_id,
    results_dir=results_dir,
    include_causal_inference=True,  # Enable causal inference techniques
    enable_wavelet_analysis=True,   # Enable wavelet coherence analysis
    enable_regional_analysis=True   # Enable regional analysis
)

# Load the data
print("Loading and preprocessing data...")
analyzer.load_data()

# Run the policy lag analysis for each country
print(f"Running policy analysis for {len(countries)} countries...")

for country in countries:
    print(f"\nAnalyzing {country}...")
    
    # Run cross-correlation function analysis
    print(f"  Running CCF analysis for {country}...")
    ccf_results = analyzer.run_ccf_analysis(country)
    
    # Run Granger causality analysis
    print(f"  Running Granger causality analysis for {country}...")
    granger_results = analyzer.run_granger_analysis(country)
    
    # Run transfer function modeling if available
    print(f"  Running transfer function modeling for {country}...")
    transfer_results = analyzer.run_transfer_function_modeling(country)
    
    # Run wavelet coherence analysis if enabled
    if analyzer.enable_wavelet_analysis:
        print(f"  Running wavelet coherence analysis for {country}...")
        wavelet_results = analyzer.run_wavelet_analysis(country)
    
    # Run causal inference analysis if enabled
    if analyzer.include_causal_inference:
        print(f"  Running causal inference analysis for {country}...")
        causal_results = analyzer.run_causal_analysis(country)

# Generate summary reports
print("\nGenerating summary reports...")
analyzer.generate_summary_report()

print("\nAnalysis complete. Detailed results saved to:", run_dir)
