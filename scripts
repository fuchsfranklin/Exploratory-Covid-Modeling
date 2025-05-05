import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np

COVID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

class CovidDataFetcher:
    def __init__(self, data_url: str = COVID_DATA_URL):
        self.data_url = data_url
        self._data = None

    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        if self._data is not None and not force_refresh:
            return self._data
        response = requests.get(self.data_url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, parse_dates=["date"])
        self._data = df
        return df

    def get_country_data(self, country: str) -> pd.DataFrame:
        df = self.fetch_data()
        country_df = df[df["location"] == country].copy()
        country_df.sort_values("date", inplace=True)
        country_df.reset_index(drop=True, inplace=True)
        return country_df

class CovidEDA:
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def describe_country(self, country: str):
        df = self.fetcher.get_country_data(country)
        print(f"\n--- {country} Data Overview ---")
        print(df.describe(include='all'))
        print("\nColumns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        return df

    def plot_time_series(self, df, columns, title=None, save_path=None):
        plt.figure(figsize=(14, 6))
        for col in columns:
            plt.plot(df['date'], df[col], label=col)
        plt.legend()
        plt.title(title or f"Time Series: {', '.join(columns)}")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_policy_vs_cases(self, df, save_path=None):
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stringency Index', color='tab:blue')
        ax1.plot(df['date'], df['stringency_index'], color='tab:blue', label='Stringency Index')
        ax2 = ax1.twinx()
        ax2.set_ylabel('New Cases', color='tab:red')
        ax2.plot(df['date'], df['new_cases'].rolling(7).mean(), color='tab:red', label='New Cases (7d MA)')
        plt.title('Stringency Index vs. New Cases')
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_vaccination_vs_cases(self, df, save_path=None):
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('People Vaccinated per Hundred', color='tab:green')
        ax1.plot(df['date'], df['people_vaccinated_per_hundred'], color='tab:green', label='Vaccinated/100')
        ax2 = ax1.twinx()
        ax2.set_ylabel('New Cases', color='tab:orange')
        ax2.plot(df['date'], df['new_cases'].rolling(7).mean(), color='tab:orange', label='New Cases (7d MA)')
        plt.title('Vaccination vs. New Cases')
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_icu_and_hospital(self, df, save_path=None):
        plt.figure(figsize=(14, 6))
        plt.plot(df['date'], df['icu_patients'], label='ICU Patients')
        plt.plot(df['date'], df['hosp_patients'], label='Hospital Patients')
        plt.legend()
        plt.title('ICU and Hospital Patients Over Time')
        plt.xlabel('Date')
        plt.ylabel('Patients')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def describe_global(self):
        df = self.fetcher.fetch_data()
        print(f"\n--- Global Data Overview ---")
        print(df.describe(include='all'))
        print("\nColumns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        return df

    def plot_global_time_series(self, columns, title=None, save_path=None):
        df = self.fetcher.fetch_data()
        df_global = df.groupby('date')[columns].sum().reset_index()
        plt.figure(figsize=(14, 6))
        for col in columns:
            plt.plot(df_global['date'], df_global[col], label=col)
        plt.legend()
        plt.title(title or f"Global Time Series: {', '.join(columns)}")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_global_policy_vs_cases(self, save_path=None):
        df = self.fetcher.fetch_data()
        df_global = df.groupby('date').agg({
            'stringency_index': 'mean',
            'new_cases': 'sum'
        }).reset_index()
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Mean Stringency Index', color='tab:blue')
        ax1.plot(df_global['date'], df_global['stringency_index'], color='tab:blue', label='Mean Stringency Index')
        ax2 = ax1.twinx()
        ax2.set_ylabel('New Cases', color='tab:red')
        ax2.plot(df_global['date'], df_global['new_cases'].rolling(7).mean(), color='tab:red', label='New Cases (7d MA)')
        plt.title('Global Stringency Index vs. New Cases')
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_global_vaccination_vs_cases(self, save_path=None):
        df = self.fetcher.fetch_data()
        df_global = df.groupby('date').agg({
            'people_vaccinated_per_hundred': 'mean',
            'new_cases': 'sum'
        }).reset_index()
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Mean People Vaccinated per Hundred', color='tab:green')
        ax1.plot(df_global['date'], df_global['people_vaccinated_per_hundred'], color='tab:green', label='Vaccinated/100')
        ax2 = ax1.twinx()
        ax2.set_ylabel('New Cases', color='tab:orange')
        ax2.plot(df_global['date'], df_global['new_cases'].rolling(7).mean(), color='tab:orange', label='New Cases (7d MA)')
        plt.title('Global Vaccination vs. New Cases')
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_global_icu_and_hospital(self, save_path=None):
        df = self.fetcher.fetch_data()
        df_global = df.groupby('date').agg({
            'icu_patients': 'sum',
            'hosp_patients': 'sum'
        }).reset_index()
        plt.figure(figsize=(14, 6))
        plt.plot(df_global['date'], df_global['icu_patients'], label='ICU Patients')
        plt.plot(df_global['date'], df_global['hosp_patients'], label='Hospital Patients')
        plt.legend()
        plt.title('Global ICU and Hospital Patients Over Time')
        plt.xlabel('Date')
        plt.ylabel('Patients')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Example EDA usage
if __name__ == "__main__":
    fetcher = CovidDataFetcher()
    eda = CovidEDA(fetcher)
    df_all = fetcher.fetch_data()
    countries = df_all['location'].unique()
    summary_rows = []
    for country in countries:
        df = fetcher.get_country_data(country)
        if df.empty:
            continue
        desc = df.describe(include='all')
        missing = df.isnull().sum()[df.isnull().sum() > 0]
        summary_rows.append({
            'country': country,
            'n_rows': len(df),
            'date_min': df['date'].min(),
            'date_max': df['date'].max(),
            'n_columns': len(df.columns),
            'n_missing_cols': len(missing),
            'missing_cols': ', '.join(missing.index[:5]) if len(missing) > 0 else '',
            'missing_vals': ', '.join(str(missing.values[:5])) if len(missing) > 0 else ''
        })
        # Optionally, save per-country summary as CSV or markdown
        desc.to_csv(f'summary_{country}.csv')
        missing.to_csv(f'missing_{country}.csv')
    # Save all-country summary
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv('country_eda_summary.csv', index=False)
    # Global EDA
    eda.describe_global()
    eda.plot_global_time_series(['new_cases', 'new_deaths'], title='Global: New Cases & Deaths', save_path='global_new_cases_deaths.png')
    eda.plot_global_policy_vs_cases(save_path='global_policy_vs_cases.png')
    eda.plot_global_vaccination_vs_cases(save_path='global_vaccination_vs_cases.png')
    eda.plot_global_icu_and_hospital(save_path='global_icu_and_hospital.png')