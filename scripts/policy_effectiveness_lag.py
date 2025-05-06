"""
Policy Effectiveness Lag Analysis

This module analyzes the time lag between policy changes (measured by stringency index)
and observable effects on COVID-19 metrics (new cases, deaths).

Methods:
- Cross-correlation analysis to identify time lags
- Granger causality tests to verify causality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf, grangercausalitytests
from sklearn.preprocessing import MinMaxScaler
import os

os.makedirs('eda_outputs/per_country', exist_ok=True)

class PolicyLagAnalyzer:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()

    def preprocess_data(self):
        self.data['stringency_index'] = self.scaler.fit_transform(self.data[['stringency_index']])
        self.data['new_cases'] = self.scaler.fit_transform(self.data[['new_cases']])
        self.data['new_deaths'] = self.scaler.fit_transform(self.data[['new_deaths']])

    def cross_correlation_analysis(self, country):
        country_data = self.data[self.data['location'] == country]
        stringency_index = country_data['stringency_index']
        new_cases = country_data['new_cases']
        new_deaths = country_data['new_deaths']

        ccf_cases = ccf(stringency_index, new_cases)
        ccf_deaths = ccf(stringency_index, new_deaths)

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.plot(ccf_cases)
        plt.title(f'Cross-correlation (Stringency Index vs New Cases) - {country}')
        plt.subplot(122)
        plt.plot(ccf_deaths)
        plt.title(f'Cross-correlation (Stringency Index vs New Deaths) - {country}')
        plt.savefig(f'eda_outputs/per_country/{country}_cross_correlation.png')
        plt.close()

    def granger_causality_tests(self, country, max_lag=15):
        country_data = self.data[self.data['location'] == country]
        stringency_index = country_data['stringency_index']
        new_cases = country_data['new_cases']
        new_deaths = country_data['new_deaths']

        data_cases = pd.concat([stringency_index, new_cases], axis=1)
        data_deaths = pd.concat([stringency_index, new_deaths], axis=1)

        granger_cases = grangercausalitytests(data_cases, max_lag, verbose=False)
        granger_deaths = grangercausalitytests(data_deaths, max_lag, verbose=False)

        return granger_cases, granger_deaths

# Example usage
if __name__ == "__main__":
    data = pd.read_csv('owid-covid-data.csv')
    analyzer = PolicyLagAnalyzer(data)
    analyzer.preprocess_data()
    analyzer.cross_correlation_analysis('Country_Name')
    granger_cases, granger_deaths = analyzer.granger_causality_tests('Country_Name')
    print(granger_cases)
    print(granger_deaths)