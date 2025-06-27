"""
Dashboard component utilities for COVID-19 analysis dashboard.

This module provides reusable components and utilities for the Streamlit dashboard,
including visualization helpers, data processing functions, and custom UI elements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def create_time_series_chart(df, x_col, y_cols, labels, colors, title, height=500, secondary_y=False):
    """
    Create a time series chart with multiple series.
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing the data
    x_col : str
        Column name for the x-axis (typically a date)
    y_cols : list
        List of column names for y-axis values
    labels : list
        List of labels for the y-axis series
    colors : list
        List of colors for the y-axis series
    title : str
        Chart title
    height : int
        Chart height in pixels
    secondary_y : bool or list
        Whether to use secondary y-axis. If list, specifies which series use secondary axis.
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The created figure
    """
    if secondary_y and not isinstance(secondary_y, list):
        # Default: last series on secondary axis
        secondary_y = [False] * (len(y_cols) - 1) + [True]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for i, (y_col, label, color) in enumerate(zip(y_cols, labels, colors)):
        use_secondary_y = secondary_y[i] if isinstance(secondary_y, list) else False
        
        fig.add_trace(
            go.Scatter(
                x=df[x_col], 
                y=df[y_col], 
                name=label, 
                line=dict(color=color)
            ),
            secondary_y=use_secondary_y
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02)
    )
    
    # Set y-axis titles if provided
    if isinstance(secondary_y, list) and True in secondary_y:
        primary_series = [label for i, label in enumerate(labels) if not secondary_y[i]]
        secondary_series = [label for i, label in enumerate(labels) if secondary_y[i]]
        
        if primary_series:
            fig.update_yaxes(title_text=primary_series[0], secondary_y=False)
        if secondary_series:
            fig.update_yaxes(title_text=secondary_series[0], secondary_y=True)
    
    return fig

def create_choropleth_map(df, location_col, color_col, hover_name_col, title, color_scale="Viridis"):
    """
    Create a choropleth map visualization.
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame containing the data
    location_col : str
        Column name for location codes (ISO codes)
    color_col : str
        Column name for the values to color by
    hover_name_col : str
        Column name for hover text
    title : str
        Map title
    color_scale : str
        Color scale for the map
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The created figure
    """
    import plotly.express as px
    
    fig = px.choropleth(
        df, 
        locations=location_col,
        color=color_col,
        hover_name=hover_name_col,
        color_continuous_scale=color_scale,
        title=title
    )
    
    fig.update_layout(
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

def create_feature_importance_chart(feature_dict, title, height=600, color_scale="Viridis"):
    """
    Create a horizontal bar chart for feature importance.
    
    Parameters:
    -----------
    feature_dict : dict
        Dictionary with feature names as keys and importance values as values
    title : str
        Chart title
    height : int
        Chart height in pixels
    color_scale : str
        Color scale for the bars
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The created figure
    """
    import plotly.express as px
    
    # Sort features by importance
    sorted_features = {k: v for k, v in sorted(feature_dict.items(), 
                                              key=lambda item: item[1], 
                                              reverse=True)}
    
    fig = px.bar(
        x=list(sorted_features.values()),
        y=list(sorted_features.keys()),
        orientation='h',
        labels={'x': 'Importance (%)', 'y': 'Feature'},
        title=title,
        color=list(sorted_features.values()),
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(height=height)
    return fig

def display_metrics_row(metrics_dict, prefix="", suffix=""):
    """
    Display a row of metric values.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names as keys and values as values
    prefix : str
        Prefix for metric values (e.g., "$")
    suffix : str
        Suffix for metric values (e.g., "%")
    """
    # Create columns dynamically based on number of metrics
    cols = st.columns(len(metrics_dict))
    
    # Display each metric in its own column
    for i, (label, value) in enumerate(metrics_dict.items()):
        # Format value based on type
        if isinstance(value, (int, np.integer)):
            formatted_value = f"{prefix}{value:,d}{suffix}"
        elif isinstance(value, float):
            formatted_value = f"{prefix}{value:.2f}{suffix}"
        else:
            formatted_value = f"{prefix}{value}{suffix}"
            
        cols[i].metric(label, formatted_value)

def filter_dataframe(df, filters):
    """
    Apply multiple filters to a dataframe.
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame to filter
    filters : dict
        Dictionary of {column: filter_value} pairs
        
    Returns:
    --------
    DataFrame
        Filtered dataframe
    """
    result = df.copy()
    
    for col, value in filters.items():
        if col in result.columns:
            if isinstance(value, (list, tuple)):
                # For list values, check if the column value is in the list
                result = result[result[col].isin(value)]
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                # For range values, filter between min and max
                result = result[
                    (result[col] >= value['min']) & 
                    (result[col] <= value['max'])
                ]
            elif value is not None:
                # For single values, exact match
                result = result[result[col] == value]
    
    return result

def period_selector(df, date_col='date'):
    """
    Create a date range selector for a dataframe with date column.
    
    Parameters:
    -----------
    df : DataFrame
        Pandas DataFrame with a date column
    date_col : str
        Name of the date column
        
    Returns:
    --------
    tuple
        (start_date, end_date) selected by the user
    """
    if date_col not in df.columns:
        st.error(f"Date column '{date_col}' not found in dataframe")
        return None, None
    
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Convert to datetime if needed
    if not isinstance(min_date, datetime):
        min_date = pd.to_datetime(min_date)
    if not isinstance(max_date, datetime):
        max_date = pd.to_datetime(max_date)
        
    # Create the date range selector
    start_date, end_date = st.date_input(
        "Select date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    return start_date, end_date

class CountrySelector:
    """
    A class that provides country selection functionality for the dashboard.
    """
    def __init__(self, df, country_col='location'):
        """
        Initialize the country selector.
        
        Parameters:
        -----------
        df : DataFrame
            Pandas DataFrame with country data
        country_col : str
            Name of the country column
        """
        self.df = df
        self.country_col = country_col
        self.all_countries = sorted(df[country_col].unique())
        
    def select_countries(self, default=None, max_countries=5, key=None):
        """
        Create a multi-select widget for country selection.
        
        Parameters:
        -----------
        default : list or None
            Default selected countries
        max_countries : int
            Maximum number of countries that can be selected
        key : str or None
            Unique key for the widget
            
        Returns:
        --------
        list
            List of selected countries
        """
        # Create default list if not provided
        if default is None:
            # Select some major countries by default
            major_countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Italy']
            default = [c for c in major_countries if c in self.all_countries][:max_countries]
        
        selected = st.multiselect(
            "Select countries to compare",
            options=self.all_countries,
            default=default,
            key=key
        )
        
        # Warning if too many countries selected
        if len(selected) > max_countries:
            st.warning(f"You've selected {len(selected)} countries. Consider limiting to {max_countries} for better visualization.")
        
        return selected
    
    def filter_by_selected(self, selected_countries):
        """
        Filter the dataframe to include only selected countries.
        
        Parameters:
        -----------
        selected_countries : list
            List of countries to include
            
        Returns:
        --------
        DataFrame
            Filtered dataframe
        """
        return self.df[self.df[self.country_col].isin(selected_countries)].copy()
