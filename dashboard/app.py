"""
COVID-19 Pandemic Analysis Dashboard

An interactive web application to explore and visualize key findings from the
Exploratory COVID-19 Demographic Modeling and Analysis project. Built with Streamlit.

Features:
- Interactive time series visualization of healthcare strain
- Pandemic fatigue detection visualization
- Policy effectiveness analysis visualization
- Cross-country comparison tools
- Custom filtering and exploration capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from datetime import datetime, timedelta
import warnings

# Add parent directory to path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
try:
    from scripts.healthcare_strain import HealthcareStrainPredictor
    from scripts.pandemic_fatigue import PandemicFatiguePredictor
    from scripts.policy_effectiveness_lag import PolicyLagAnalyzer
    modules_available = True
except ImportError:
    modules_available = False
    st.warning("Could not import analysis modules. Some interactive features may be limited.")

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the data path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "owid-covid-data.csv")

# Load data function
@st.cache_data
def load_data():
    """Load and minimally process the COVID data."""
    try:
        df = pd.read_csv(DATA_PATH)
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "Healthcare Strain", "Pandemic Fatigue", "Policy Effectiveness", "Cross-Country Comparison", "About"]
)

# Load data
df = load_data()

# Overview page
if page == "Overview":
    st.title("COVID-19 Pandemic Analysis Dashboard")
    
    st.markdown("""
    ## Exploratory COVID-19 Demographic Modeling and Analysis
    
    This dashboard provides interactive visualizations of key findings from our comprehensive
    analysis of COVID-19 pandemic dynamics. Explore the three main research areas:
    
    1. **Healthcare Strain Prediction**: Forecasting ICU utilization with machine learning models
    2. **Pandemic Fatigue Detection**: Data-driven identification of compliance fatigue periods
    3. **Policy Effectiveness Analysis**: Quantifying impact and lag of policy interventions
    
    Use the sidebar to navigate between different analyses.
    """)
    
    # Display global statistics
    if df is not None:
        # Filter to get the most recent data for each country
        latest_data = df.loc[df.groupby('location')['date'].idxmax()]
        
        # Calculate global stats
        total_cases = latest_data['total_cases'].sum()
        total_deaths = latest_data['total_deaths'].sum()
        countries_analyzed = len(latest_data)
        
        # Create metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cases Analyzed", f"{total_cases:,.0f}")
        col2.metric("Total Deaths Analyzed", f"{total_deaths:,.0f}")
        col3.metric("Countries in Dataset", countries_analyzed)
        
        # Create a global cases and deaths plot
        st.subheader("Global Pandemic Trends")
        global_data = df.groupby('date').agg({
            'new_cases_smoothed': 'sum',
            'new_deaths_smoothed': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=global_data['date'], y=global_data['new_cases_smoothed'], 
                      name='New Cases (7-day avg)', line=dict(color='#636EFA')),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=global_data['date'], y=global_data['new_deaths_smoothed'], 
                      name='New Deaths (7-day avg)', line=dict(color='#EF553B')),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Global COVID-19 Cases and Deaths',
            xaxis_title='Date',
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02)
        )
        fig.update_yaxes(title_text="New Cases", secondary_y=False)
        fig.update_yaxes(title_text="New Deaths", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display a sample map visualization
        st.subheader("Geographical Distribution of COVID-19 Impact")
        
        metric_descriptions = {
            "total_cases_per_million": "Cumulative confirmed cases per million population",
            "total_deaths_per_million": "Cumulative deaths per million population",
            "icu_patients_per_million": "Current ICU patients per million population"
        }
        
        map_metric = st.selectbox("Select Map Metric", 
                                list(metric_descriptions.keys()))
        
        st.caption(metric_descriptions[map_metric])
        
        # Filter out rows with missing data for the selected metric
        map_data = latest_data[latest_data[map_metric].notna()].copy()
        
        if len(map_data) > 0:
            fig = px.choropleth(
                map_data, 
                locations="iso_code",
                color=map_metric,
                hover_name="location",
                color_continuous_scale="Viridis",
                title=f"{map_metric.replace('_', ' ').title()} by Country"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for {map_metric}")
    else:
        st.error("Could not load data.")

# Healthcare Strain page
elif page == "Healthcare Strain":
    st.title("Healthcare Strain Analysis")
    
    st.markdown("""
    ## ICU Utilization Prediction
    
    This analysis uses machine learning to predict ICU utilization based on various features including:
    
    - Epidemiological indicators (cases, deaths, testing rates)
    - Demographic factors (age distribution, population density)
    - Health system capacity (hospital beds, healthcare spending)
    - Policy measures (stringency index and components)
    
    The models incorporate advanced techniques including LSTM networks and ensemble methods for improved accuracy.
    """)
    
    if df is not None:
        # Filter countries with sufficient healthcare data
        countries_with_data = []
        for country in df['location'].unique():
            country_df = df[df['location'] == country]
            has_icu = country_df['icu_patients_per_million'].notna().sum() > 10
            has_hospital = country_df['hosp_patients_per_million'].notna().sum() > 10
            if has_icu or has_hospital:
                countries_with_data.append(country)
        
        countries_with_data = sorted(countries_with_data)
        
        if len(countries_with_data) == 0:
            st.error("No countries with sufficient healthcare strain data.")
        else:
            st.info(f"Showing {len(countries_with_data)} countries with ICU or hospital utilization data.")
            selected_country = st.selectbox("Select a country to analyze:", countries_with_data)
        
        # Filter data for selected country
        country_data = df[df['location'] == selected_country].copy()
        
        # Check if ICU data exists
        has_icu_data = 'icu_patients_per_million' in country_data.columns and country_data['icu_patients_per_million'].notna().sum() > 10
        
        if has_icu_data:
            st.subheader("ICU Utilization vs. Key Predictors")
            st.markdown("""
            **What this shows:** ICU patients per million (purple) compared to deaths (red, scaled ×5) and policy stringency (green).
            Deaths are the strongest predictor of ICU demand.
            
            **How to interact:** Hover over lines to see exact values. Zoom by clicking and dragging. Double-click to reset.
            """)
            
            # Create a composite plot with ICU and key predictors
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add ICU patients line
            fig.add_trace(
                go.Scatter(x=country_data['date'], y=country_data['icu_patients_per_million'], 
                          name='ICU Patients per Million', line=dict(color='#AB63FA', width=2)),
                secondary_y=False
            )
            
            # Add deaths line (strongest predictor)
            if 'new_deaths_smoothed_per_million' in country_data.columns:
                fig.add_trace(
                    go.Scatter(x=country_data['date'], y=country_data['new_deaths_smoothed_per_million']*5,  # Scaled for visibility
                              name='New Deaths per Million (×5)', line=dict(color='#EF553B')),
                    secondary_y=False
                )
            
            # Add stringency index
            if 'stringency_index' in country_data.columns:
                fig.add_trace(
                    go.Scatter(x=country_data['date'], y=country_data['stringency_index'],
                              name='Stringency Index', line=dict(color='#00CC96')),
                    secondary_y=True
                )
            
            fig.update_layout(
                title=f'ICU Utilization and Key Predictors in {selected_country}',
                xaxis_title='Date',
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02)
            )
            fig.update_yaxes(title_text="Patients per Million", secondary_y=False)
            fig.update_yaxes(title_text="Stringency Index", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature Importance in ICU Prediction Model")
            st.markdown("""
            **What this shows:** Relative importance of each feature in predicting ICU utilization. Higher percentages indicate stronger predictive power.
            
            **How to interact:** Hover over bars to see exact importance values. Features are ranked from most to least important.
            """)
            
            # Sample feature importance data (in a real app, this would come from the trained model)
            feature_importance = {
                'new_deaths_smoothed_per_million': 43.8,
                'hospital_patients_per_million': 11.5,
                'hospital_patients_7day_avg': 5.8,
                'new_deaths_smoothed_7day_avg': 5.0,
                'reproduction_rate': 4.2,
                'new_cases_smoothed_per_million': 3.7,
                'stringency_index': 3.5,
                'positive_rate': 3.2,
                'aged_65_older': 2.8,
                'cardiovasc_death_rate': 2.6,
                'diabetes_prevalence': 2.5,
                'hospital_beds_per_thousand': 2.3,
                'gdp_per_capita': 2.1,
                'human_development_index': 1.8,
                'life_expectancy': 1.5,
                'population_density': 1.2,
                'median_age': 1.0,
                'extreme_poverty': 0.8
            }
            
            # Create bar chart for feature importance
            fig = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                labels={'x': 'Importance (%)', 'y': 'Feature'},
                title='Feature Importance in Healthcare Strain Prediction Model',
                color=list(feature_importance.values()),
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Model Performance Comparison")
            st.markdown("""
            **What this shows:** Performance comparison across 4 ML models. All metrics normalized so larger area = better performance.
            MAE/RMSE are inverted (lower is better). Ensemble combines multiple models for best results.
            
            **How to interact:** Hover to see exact values. Click model names in the legend to show/hide models for easier comparison.
            """)
            
            model_performance = {
                'Model': ['Gradient Boosting', 'Random Forest', 'LSTM', 'Ensemble'],
                'MAE': [3.96, 4.25, 3.52, 3.41],
                'RMSE': [5.22, 5.64, 4.89, 4.75],
                '7-Day Accuracy': [0.913, 0.895, 0.927, 0.934],
                '14-Day Accuracy': [0.847, 0.823, 0.865, 0.872]
            }
            
            model_df = pd.DataFrame(model_performance)
            
            # Create a radar chart for model comparison
            categories = ['MAE (lower is better)', 'RMSE (lower is better)', 
                         '7-Day Accuracy', '14-Day Accuracy']
            
            fig = go.Figure()
            
            for i, model in enumerate(model_df['Model']):
                # Invert MAE and RMSE so that higher is better for all metrics
                mae_inv = 5 - model_df.loc[i, 'MAE']  # 5 is a chosen ceiling
                rmse_inv = 6 - model_df.loc[i, 'RMSE']  # 6 is a chosen ceiling
                
                fig.add_trace(go.Scatterpolar(
                    r=[mae_inv, rmse_inv, model_df.loc[i, '7-Day Accuracy'], model_df.loc[i, '14-Day Accuracy']],
                    theta=categories,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Performance Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning(f"Not enough ICU data available for {selected_country}. Showing hospital utilization instead.")
            
            # Show hospital patients instead if available
            if 'hosp_patients_per_million' in country_data.columns and country_data['hosp_patients_per_million'].notna().sum() > 10:
                st.subheader("Hospital Utilization vs. Key Predictors")
                st.markdown("""
                **What this shows:** Hospital patients per million (purple) compared to deaths (red, scaled ×5) and policy stringency (green).
                Similar patterns to ICU data but includes all hospitalized COVID patients.
                
                **How to interact:** Hover over lines to see exact values. Zoom by clicking and dragging. Double-click to reset.
                """)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=country_data['date'], y=country_data['hosp_patients_per_million'], 
                              name='Hospital Patients per Million', line=dict(color='#AB63FA', width=2)),
                    secondary_y=False
                )
                
                if 'new_deaths_smoothed_per_million' in country_data.columns:
                    fig.add_trace(
                        go.Scatter(x=country_data['date'], y=country_data['new_deaths_smoothed_per_million']*5,
                                  name='New Deaths per Million (×5)', line=dict(color='#EF553B')),
                        secondary_y=False
                    )
                
                if 'stringency_index' in country_data.columns:
                    fig.add_trace(
                        go.Scatter(x=country_data['date'], y=country_data['stringency_index'],
                                  name='Stringency Index', line=dict(color='#00CC96')),
                        secondary_y=True
                    )
                
                fig.update_layout(
                    title=f'Hospital Utilization and Key Predictors in {selected_country}',
                    xaxis_title='Date',
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.02)
                )
                fig.update_yaxes(title_text="Patients per Million", secondary_y=False)
                fig.update_yaxes(title_text="Stringency Index", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not load data.")

# Pandemic Fatigue page
elif page == "Pandemic Fatigue":
    st.title("Pandemic Fatigue Analysis")
    
    st.markdown("""
    ## Data-Driven Pandemic Fatigue Detection
    
    **Pandemic Fatigue** occurs when cases rise despite high restrictions, suggesting reduced public compliance.
    
    **Key Definitions:**
    - **Stringency Index** (0-100): Government response measure combining closures, travel bans, and restrictions
    - **Fatigue Period**: Days with stringency ≥60 AND cases increasing >20% over 14 days
    - **Implication**: High fatigue indicates policies losing effectiveness, requiring strategy adjustment
    """)
    
    if df is not None:
        # Filter countries with sufficient data
        countries_with_data = []
        for country in df['location'].unique():
            country_df = df[df['location'] == country]
            has_stringency = country_df['stringency_index'].notna().sum() > 100
            has_cases = country_df['new_cases_smoothed_per_million'].notna().sum() > 100
            if has_stringency and has_cases:
                countries_with_data.append(country)
        
        countries_with_data = sorted(countries_with_data)
        
        if len(countries_with_data) == 0:
            st.error("No countries with sufficient data for fatigue analysis.")
        else:
            st.info(f"Showing {len(countries_with_data)} countries with complete stringency and case data.")
            selected_country = st.selectbox("Select a country:", countries_with_data, key='fatigue_country')
        
        country_data = df[df['location'] == selected_country].copy()
        country_data = country_data.sort_values('date')
        
        # Define fatigue indicator
        if 'stringency_index' in country_data.columns and 'new_cases_smoothed_per_million' in country_data.columns:
            country_data['case_14d_avg'] = country_data['new_cases_smoothed_per_million'].rolling(14, min_periods=7).mean()
            country_data['case_change'] = country_data['case_14d_avg'].pct_change(periods=14)
            country_data['high_stringency'] = country_data['stringency_index'] >= 60
            country_data['rising_cases'] = country_data['case_change'] > 0.2
            country_data['fatigue_indicator'] = country_data['high_stringency'] & country_data['rising_cases']
            
            st.subheader("Stringency vs. Case Trends")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=country_data['date'], y=country_data['stringency_index'],
                          name='Stringency Index', line=dict(color='#636EFA')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=country_data['date'], y=country_data['new_cases_smoothed_per_million'],
                          name='Cases per Million', line=dict(color='#EF553B')),
                secondary_y=True
            )
            
            fatigue_periods = country_data[country_data['fatigue_indicator'] == True]
            if len(fatigue_periods) > 0:
                fig.add_trace(
                    go.Scatter(x=fatigue_periods['date'], y=fatigue_periods['stringency_index'],
                              mode='markers', name='Fatigue Periods',
                              marker=dict(color='red', size=8, symbol='x')),
                    secondary_y=False
                )
            
            fig.update_layout(
                title=f'Pandemic Fatigue Detection in {selected_country}',
                xaxis_title='Date',
                height=500,
                hovermode="x unified"
            )
            fig.update_yaxes(title_text="Stringency Index", secondary_y=False)
            fig.update_yaxes(title_text="Cases per Million", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            fatigue_pct = (country_data['fatigue_indicator'].sum() / len(country_data)) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Fatigue Days", f"{country_data['fatigue_indicator'].sum()}")
            col2.metric("% of Pandemic", f"{fatigue_pct:.1f}%")
            col3.metric("Avg Stringency", f"{country_data['stringency_index'].mean():.1f}")
            
            st.subheader("Fatigue Indicator Over Time")
            st.caption("Binary indicator showing when fatigue conditions are met (1) or not (0). Shaded areas represent periods of pandemic fatigue.")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=country_data['date'],
                y=country_data['fatigue_indicator'].astype(int),
                fill='tozeroy',
                name='Fatigue Periods',
                line=dict(color='#FF6B6B')
            ))
            
            fig.update_layout(
                title='Pandemic Fatigue Timeline',
                xaxis_title='Date',
                yaxis_title='Fatigue Indicator (0=No, 1=Yes)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Insufficient data for fatigue analysis in {selected_country}")
    else:
        st.error("Could not load data.")

# Policy Effectiveness page
elif page == "Policy Effectiveness":
    st.title("Policy Effectiveness Analysis")
    
    st.markdown("""
    ## Policy Impact and Implementation Lag
    
    **Policy Lag**: Time between implementing restrictions and observing effects on transmission.
    
    **Key Concepts:**
    - **Stringency Index**: Composite measure of government response (0-100)
    - **Reproduction Rate (R)**: Average number of people infected by one case (R<1 means declining spread)
    - **Expected Lag**: Typically 7-21 days due to incubation period and reporting delays
    """)
    
    if df is not None:
        # Filter countries with sufficient data
        countries_with_data = []
        for country in df['location'].unique():
            country_df = df[df['location'] == country]
            has_stringency = country_df['stringency_index'].notna().sum() > 100
            has_reproduction = country_df['reproduction_rate'].notna().sum() > 100
            if has_stringency and has_reproduction:
                countries_with_data.append(country)
        
        countries_with_data = sorted(countries_with_data)
        
        if len(countries_with_data) == 0:
            st.error("No countries with sufficient data for policy analysis.")
        else:
            st.info(f"Showing {len(countries_with_data)} countries with complete policy and outcome data.")
            selected_country = st.selectbox("Select a country:", countries_with_data, key='policy_country')
            
            country_data = df[df['location'] == selected_country].copy()
            country_data = country_data.sort_values('date')
            
            st.subheader("Policy Stringency vs. Reproduction Rate")
            st.caption("Observe the relationship between policy changes and transmission. Effective policies show R decreasing after stringency increases.")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=country_data['date'], y=country_data['stringency_index'],
                          name='Stringency Index', line=dict(color='#636EFA', width=2)),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=country_data['date'], y=country_data['reproduction_rate'],
                          name='Reproduction Rate (R)', line=dict(color='#EF553B', width=2)),
                secondary_y=True
            )
            
            fig.add_hline(y=1, line_dash="dash", line_color="gray", 
                         annotation_text="R=1 (threshold)", secondary_y=True)
            
            fig.update_layout(
                title=f'Policy Stringency and Transmission in {selected_country}',
                xaxis_title='Date',
                height=500,
                hovermode="x unified"
            )
            fig.update_yaxes(title_text="Stringency Index", secondary_y=False)
            fig.update_yaxes(title_text="Reproduction Rate (R)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation with lags
            st.subheader("Policy Lag Analysis")
            st.caption("Correlation between stringency and reproduction rate at different time lags. Peak negative correlation indicates optimal policy lag.")
            
            valid_data = country_data[['stringency_index', 'reproduction_rate']].dropna()
            
            if len(valid_data) > 30:
                lags = range(0, 31)
                correlations = []
                
                for lag in lags:
                    if lag == 0:
                        corr = valid_data['stringency_index'].corr(valid_data['reproduction_rate'])
                    else:
                        corr = valid_data['stringency_index'].iloc[:-lag].corr(valid_data['reproduction_rate'].iloc[lag:])
                    correlations.append(corr)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(lags),
                    y=correlations,
                    marker_color=['#EF553B' if c < 0 else '#636EFA' for c in correlations],
                    name='Correlation'
                ))
                
                best_lag = lags[correlations.index(min(correlations))]
                
                fig.update_layout(
                    title=f'Cross-Correlation: Stringency vs Reproduction Rate',
                    xaxis_title='Lag (days)',
                    yaxis_title='Correlation',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Optimal Lag", f"{best_lag} days")
                col2.metric("Peak Correlation", f"{min(correlations):.3f}")
                col3.metric("Avg Stringency", f"{country_data['stringency_index'].mean():.1f}")
                
                st.info(f"💡 In {selected_country}, policy changes show strongest effect on transmission after ~{best_lag} days.")
            else:
                st.warning("Insufficient overlapping data for lag analysis.")

# Cross-Country Comparison page
elif page == "Cross-Country Comparison":
    st.title("Cross-Country Comparison Tool")
    
    st.markdown("""
    ## Comparative Pandemic Analysis
    
    Compare COVID-19 metrics across multiple countries to identify patterns and policy effectiveness.
    """)
    
    if df is not None:
        all_countries = sorted(df['location'].unique())
        
        selected_countries = st.multiselect(
            "Select countries to compare (2-5 recommended):",
            all_countries,
            default=['United States', 'United Kingdom', 'Germany'] if all(c in all_countries for c in ['United States', 'United Kingdom', 'Germany']) else all_countries[:3]
        )
        
        if len(selected_countries) == 0:
            st.warning("Please select at least one country.")
        else:
            metric = st.selectbox(
                "Select metric to compare:",
                ['new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million', 
                 'stringency_index', 'reproduction_rate', 'icu_patients_per_million']
            )
            
            metric_labels = {
                'new_cases_smoothed_per_million': 'Cases per Million (7-day avg)',
                'new_deaths_smoothed_per_million': 'Deaths per Million (7-day avg)',
                'stringency_index': 'Stringency Index',
                'reproduction_rate': 'Reproduction Rate (R)',
                'icu_patients_per_million': 'ICU Patients per Million'
            }
            
            st.subheader(f"Comparison: {metric_labels[metric]}")
            
            fig = go.Figure()
            
            for country in selected_countries:
                country_data = df[df['location'] == country].copy()
                country_data = country_data.sort_values('date')
                
                fig.add_trace(go.Scatter(
                    x=country_data['date'],
                    y=country_data[metric],
                    name=country,
                    mode='lines'
                ))
            
            fig.update_layout(
                title=f'{metric_labels[metric]} - Multi-Country Comparison',
                xaxis_title='Date',
                yaxis_title=metric_labels[metric],
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            summary_data = []
            for country in selected_countries:
                country_df = df[df['location'] == country]
                summary_data.append({
                    'Country': country,
                    'Mean': country_df[metric].mean(),
                    'Max': country_df[metric].max(),
                    'Latest': country_df[metric].iloc[-1] if len(country_df) > 0 else None
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({'Mean': '{:.2f}', 'Max': '{:.2f}', 'Latest': '{:.2f}'}), use_container_width=True)
    else:
        st.error("Could not load data.")

# About page
elif page == "About":
    st.title("About This Project")
    
    st.markdown("""
    ## Exploratory COVID-19 Demographic Modeling and Analysis
    
    This interactive dashboard is part of a comprehensive research project analyzing COVID-19 pandemic
    dynamics across three critical dimensions: healthcare system strain, pandemic fatigue, and policy
    effectiveness.
    
    ### Key Innovations
    
    - **Novel Research Angles**: Integration of variant-specific analysis and social media sentiment
    - **Advanced Methodology**: Deep learning models, ensemble methods, and causal inference techniques
    - **Interactive Exploration**: Dynamic visualization tools for exploring complex pandemic patterns
    
    ### Data Sources
    
    - Our World in Data COVID-19 Dataset
    - COVID-19 Twitter Sentiment Dataset
    - Google COVID-19 Community Mobility Reports
    - Oxford COVID-19 Government Response Tracker
    
    ### Contributors
    
    This project was developed by Franklin Fuchs.
    
    For more information or to view the source code, visit the [GitHub repository](https://github.com/yourusername/Exploratory-Covid-Modeling).
    """)
    
    # Rest of the about page code
    # ...

# Footer
st.markdown("---")
st.markdown("© 2023 | COVID-19 Pandemic Analysis Dashboard | [GitHub Repository](https://github.com/yourusername/Exploratory-Covid-Modeling)")
