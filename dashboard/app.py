"""
COVID-19 Pandemic Analysis Dashboard

Interactive Streamlit app for exploring healthcare strain predictions,
pandemic fatigue detection, and policy effectiveness lag analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# Page config
st.set_page_config(
    page_title="COVID-19 Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "owid-covid-data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------

@st.cache_data
def load_data():
    """Load the OWID COVID dataset."""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def _find_latest_run(analysis_dir):
    """Find the most recent run directory for a given analysis."""
    full_path = os.path.join(RESULTS_DIR, analysis_dir)
    if not os.path.exists(full_path):
        return None
    runs = sorted([d for d in os.listdir(full_path)
                   if os.path.isdir(os.path.join(full_path, d))])
    return os.path.join(full_path, runs[-1]) if runs else None


@st.cache_data
def load_healthcare_results():
    """Load the latest healthcare strain model results."""
    run_dir = _find_latest_run("healthcare_strain")
    if not run_dir:
        return None, None, None
    details_path = os.path.join(run_dir, "run_details.json")
    importance_path = os.path.join(run_dir, "feature_importances.csv")
    preds_path = os.path.join(run_dir, "test_predictions_vs_actual.csv")

    details = json.load(open(details_path)) if os.path.exists(details_path) else None
    importances = pd.read_csv(importance_path, index_col=0) if os.path.exists(importance_path) else None
    predictions = pd.read_csv(preds_path, parse_dates=['date']) if os.path.exists(preds_path) else None
    return details, importances, predictions


@st.cache_data
def load_policy_results():
    """Load the latest policy effectiveness aggregate results."""
    run_dir = _find_latest_run("policy_effectiveness")
    if not run_dir:
        return None
    agg_path = os.path.join(run_dir, "aggregate_results.json")
    return json.load(open(agg_path)) if os.path.exists(agg_path) else None


@st.cache_data
def load_fatigue_results():
    """Load the latest pandemic fatigue model results."""
    run_dir = _find_latest_run("pandemic_fatigue")
    if not run_dir:
        return None, None
    details_path = os.path.join(run_dir, "run_details.json")
    coeff_path = os.path.join(run_dir, "feature_coefficients.csv")
    details = json.load(open(details_path)) if os.path.exists(details_path) else None
    coeffs = pd.read_csv(coeff_path) if os.path.exists(coeff_path) else None
    return details, coeffs


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "Healthcare Strain", "Pandemic Fatigue",
     "Policy Effectiveness", "Cross-Country Comparison", "About"]
)

df = load_data()

# ==================================================================
# OVERVIEW PAGE
# ==================================================================
if page == "Overview":
    st.title("COVID-19 Pandemic Analysis Dashboard")

    st.markdown("""
    This dashboard presents findings from three research analyses:

    1. **Healthcare Strain Prediction** — Forecasting ICU utilization using lagged indicators
    2. **Pandemic Fatigue Detection** — Identifying periods where restrictions lose effectiveness
    3. **Policy Effectiveness Analysis** — Quantifying the lag between policy changes and outcomes

    Use the sidebar to navigate between analyses.
    """)

    if df is not None:
        latest = df.loc[df.groupby('location')['date'].idxmax()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cases", f"{latest['total_cases'].sum():,.0f}")
        col2.metric("Total Deaths", f"{latest['total_deaths'].sum():,.0f}")
        col3.metric("Countries in Dataset", len(latest))

        st.subheader("Global Pandemic Trends")
        global_data = df.groupby('date').agg({
            'new_cases_smoothed': 'sum',
            'new_deaths_smoothed': 'sum'
        }).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=global_data['date'], y=global_data['new_cases_smoothed'],
                       name='New Cases (7-day avg)', line=dict(color='#636EFA')),
            secondary_y=False)
        fig.add_trace(
            go.Scatter(x=global_data['date'], y=global_data['new_deaths_smoothed'],
                       name='New Deaths (7-day avg)', line=dict(color='#EF553B')),
            secondary_y=True)
        fig.update_layout(title='Global COVID-19 Cases and Deaths',
                          xaxis_title='Date', height=500, hovermode="x unified",
                          legend=dict(orientation="h", y=1.02))
        fig.update_yaxes(title_text="New Cases", secondary_y=False)
        fig.update_yaxes(title_text="New Deaths", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Geographical Distribution")
        metric_options = {
            "total_cases_per_million": "Cumulative cases per million",
            "total_deaths_per_million": "Cumulative deaths per million",
            "icu_patients_per_million": "ICU patients per million"
        }
        map_metric = st.selectbox("Select metric", list(metric_options.keys()))
        st.caption(metric_options[map_metric])

        map_data = latest[latest[map_metric].notna()].copy()
        if len(map_data) > 0:
            fig = px.choropleth(map_data, locations="iso_code", color=map_metric,
                                hover_name="location", color_continuous_scale="Viridis",
                                title=f"{map_metric.replace('_', ' ').title()}")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not load data.")


# ==================================================================
# HEALTHCARE STRAIN PAGE
# ==================================================================
elif page == "Healthcare Strain":
    st.title("Healthcare Strain Prediction")

    st.markdown("""
    Predicts ICU utilization per million using a Gradient Boosting model trained on
    **lagged** epidemiological indicators, demographic factors, and policy measures.

    Only lagged features are used as predictors — no contemporaneous values — so the
    model provides genuine forward-looking predictions (7-14 day horizon).
    """)

    details, importances, predictions = load_healthcare_results()

    if details:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{details['mae']:.2f} ICU/million")
        col2.metric("RMSE", f"{details['rmse']:.2f}")
        col3.metric("Features", details['n_features'])

    if df is not None:
        # Country selector for ICU visualization
        countries_with_icu = sorted([
            c for c in df['location'].unique()
            if df[df['location'] == c]['icu_patients_per_million'].notna().sum() > 10
        ])

        if countries_with_icu:
            selected = st.selectbox("Select country:", countries_with_icu)
            cdata = df[df['location'] == selected].copy()

            st.subheader("ICU Utilization vs Key Predictors")
            st.caption("ICU patients (purple) vs 7-day-lagged deaths (red, scaled) and stringency (green).")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['icu_patients_per_million'],
                                     name='ICU Patients/M', line=dict(color='#AB63FA', width=2)),
                          secondary_y=False)
            if 'new_deaths_smoothed_per_million' in cdata.columns:
                fig.add_trace(go.Scatter(x=cdata['date'],
                                         y=cdata['new_deaths_smoothed_per_million'].shift(-7) * 5,
                                         name='Deaths/M (7d lag, ×5)', line=dict(color='#EF553B')),
                              secondary_y=False)
            if 'stringency_index' in cdata.columns:
                fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['stringency_index'],
                                         name='Stringency Index', line=dict(color='#00CC96')),
                              secondary_y=True)
            fig.update_layout(title=f'ICU Utilization — {selected}', xaxis_title='Date',
                              height=500, hovermode="x unified",
                              legend=dict(orientation="h", y=1.02))
            fig.update_yaxes(title_text="Patients per Million", secondary_y=False)
            fig.update_yaxes(title_text="Stringency Index", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance from actual model results
    if importances is not None:
        st.subheader("Feature Importance (from trained model)")
        st.caption("Relative importance of each feature in predicting ICU utilization.")

        imp_df = importances.head(15).reset_index()
        imp_df.columns = ['Feature', 'Importance']
        imp_df['Importance %'] = imp_df['Importance'] * 100

        fig = px.bar(imp_df, x='Importance %', y='Feature', orientation='h',
                     color='Importance %', color_continuous_scale='Viridis',
                     title='Top 15 Feature Importances')
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Predictions scatter
    if predictions is not None:
        st.subheader("Model Predictions vs Actual (Test Set)")
        st.caption("Each point is one country-day. Closer to the diagonal = better prediction.")

        fig = px.scatter(predictions, x='actual', y='predicted',
                         opacity=0.3, title='Predicted vs Actual ICU Patients/Million')
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                 mode='lines', name='Perfect prediction',
                                 line=dict(dash='dash', color='gray')))
        fig.update_layout(height=500, xaxis_title='Actual', yaxis_title='Predicted')
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# PANDEMIC FATIGUE PAGE
# ==================================================================
elif page == "Pandemic Fatigue":
    st.title("Pandemic Fatigue Detection")

    st.markdown("""
    **Pandemic fatigue** occurs when cases rise despite high restrictions, suggesting
    reduced public compliance.

    **Definition used:** Stringency index ≥ 60 AND cases increasing > 20% over 14 days.
    """)

    fatigue_details, fatigue_coeffs = load_fatigue_results()

    if fatigue_details:
        eval_data = fatigue_details.get('evaluation', {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Balanced Accuracy", f"{eval_data.get('balanced_accuracy', 0):.3f}")
        col2.metric("ROC AUC", f"{eval_data.get('roc_auc', 0):.3f}" if eval_data.get('roc_auc') else "N/A")
        col3.metric("F1 (Fatigue Class)", f"{eval_data.get('f1_fatigue_class', 0):.3f}")

    if df is not None:
        countries_with_data = sorted([
            c for c in df['location'].unique()
            if (df[df['location'] == c]['stringency_index'].notna().sum() > 100 and
                df[df['location'] == c]['new_cases_smoothed_per_million'].notna().sum() > 100)
        ])

        if countries_with_data:
            selected = st.selectbox("Select country:", countries_with_data, key='fatigue_country')
            cdata = df[df['location'] == selected].copy().sort_values('date')

            # Compute fatigue indicator live
            cdata['case_14d_avg'] = cdata['new_cases_smoothed_per_million'].rolling(14, min_periods=7).mean()
            cdata['case_change'] = cdata['case_14d_avg'].pct_change(periods=14)
            cdata['high_stringency'] = cdata['stringency_index'] >= 60
            cdata['rising_cases'] = cdata['case_change'] > 0.2
            cdata['fatigue'] = cdata['high_stringency'] & cdata['rising_cases']

            st.subheader("Stringency vs Case Trends")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['stringency_index'],
                                     name='Stringency Index', line=dict(color='#636EFA')),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['new_cases_smoothed_per_million'],
                                     name='Cases/Million', line=dict(color='#EF553B')),
                          secondary_y=True)

            fatigue_pts = cdata[cdata['fatigue']]
            if len(fatigue_pts) > 0:
                fig.add_trace(go.Scatter(x=fatigue_pts['date'], y=fatigue_pts['stringency_index'],
                                         mode='markers', name='Fatigue Periods',
                                         marker=dict(color='red', size=6, symbol='x')),
                              secondary_y=False)

            fig.update_layout(title=f'Pandemic Fatigue — {selected}', xaxis_title='Date',
                              height=500, hovermode="x unified")
            fig.update_yaxes(title_text="Stringency Index", secondary_y=False)
            fig.update_yaxes(title_text="Cases per Million", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            fatigue_pct = (cdata['fatigue'].sum() / len(cdata)) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Fatigue Days", f"{cdata['fatigue'].sum()}")
            col2.metric("% of Pandemic", f"{fatigue_pct:.1f}%")
            col3.metric("Avg Stringency", f"{cdata['stringency_index'].mean():.1f}")

            st.subheader("Fatigue Timeline")
            st.caption("Shaded areas show periods meeting the fatigue criteria.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['fatigue'].astype(int),
                                     fill='tozeroy', name='Fatigue', line=dict(color='#FF6B6B')))
            fig.update_layout(title='Fatigue Indicator Over Time', xaxis_title='Date',
                              yaxis_title='Fatigue (0/1)', height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Feature coefficients from actual model
    if fatigue_coeffs is not None:
        st.subheader("Model Feature Coefficients")
        st.caption("Coefficients from the trained logistic regression model. "
                   "Positive = increases fatigue probability, Negative = decreases it.")
        top_n = fatigue_coeffs.head(15)
        fig = px.bar(top_n, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='RdBu_r',
                     title='Top 15 Feature Coefficients (by absolute value)')
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# POLICY EFFECTIVENESS PAGE
# ==================================================================
elif page == "Policy Effectiveness":
    st.title("Policy Effectiveness Lag Analysis")

    st.markdown("""
    Quantifies the time lag between policy interventions (stringency index) and
    their observable effect on transmission outcomes.

    **Methods:** Cross-correlation, Granger causality, Wavelet coherence.

    **Expected lag:** Typically 7-21 days due to incubation period and reporting delays.
    """)

    policy_results = load_policy_results()

    if policy_results:
        st.subheader("Aggregate Results Across Countries")

        for pair_key, summary in policy_results.get('pair_summaries', {}).items():
            nice_name = pair_key.replace('stringency_index_vs_', '').replace('_', ' ').title()
            st.markdown(f"**Stringency → {nice_name}**")

            n_sig = summary.get('n_significant', 0)
            n_total = summary.get('n_total', 0)
            median_lag = summary.get('median_lag_days')

            col1, col2, col3 = st.columns(3)
            col1.metric("Countries Significant", f"{n_sig}/{n_total}")
            col2.metric("Median Lag", f"{median_lag} days" if median_lag else "N/A")
            col3.metric("Lag Range",
                        f"{summary.get('min_lag_days', '?')}–{summary.get('max_lag_days', '?')} days"
                        if median_lag else "N/A")

            # Per-country lag chart
            details = summary.get('country_details', {})
            countries_with_lag = {c: d['consensus_lag'] for c, d in details.items()
                                 if d.get('consensus_lag') is not None}
            if countries_with_lag:
                lag_df = pd.DataFrame(list(countries_with_lag.items()),
                                      columns=['Country', 'Lag (days)'])
                lag_df = lag_df.sort_values('Lag (days)')
                fig = px.bar(lag_df, x='Lag (days)', y='Country', orientation='h',
                             color='Lag (days)', color_continuous_scale='Viridis',
                             title=f'Policy Lag by Country — {nice_name}')
                fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

    # Live cross-correlation for selected country
    if df is not None:
        st.subheader("Interactive Lag Analysis")
        st.caption("Explore the correlation between stringency and reproduction rate at different lags.")

        countries_for_policy = sorted([
            c for c in df['location'].unique()
            if (df[df['location'] == c]['stringency_index'].notna().sum() > 100 and
                df[df['location'] == c]['reproduction_rate'].notna().sum() > 100)
        ])

        if countries_for_policy:
            selected = st.selectbox("Select country:", countries_for_policy, key='policy_country')
            cdata = df[df['location'] == selected].copy().sort_values('date')

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['stringency_index'],
                                     name='Stringency', line=dict(color='#636EFA', width=2)),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=cdata['date'], y=cdata['reproduction_rate'],
                                     name='R (Reproduction Rate)', line=dict(color='#EF553B', width=2)),
                          secondary_y=True)
            fig.add_hline(y=1, line_dash="dash", line_color="gray",
                          annotation_text="R=1", secondary_y=True)
            fig.update_layout(title=f'Stringency vs Reproduction Rate — {selected}',
                              xaxis_title='Date', height=500, hovermode="x unified")
            fig.update_yaxes(title_text="Stringency Index", secondary_y=False)
            fig.update_yaxes(title_text="Reproduction Rate", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Compute cross-correlation
            valid = cdata[['stringency_index', 'reproduction_rate']].dropna()
            if len(valid) > 30:
                lags = range(0, 31)
                corrs = []
                for lag in lags:
                    if lag == 0:
                        c = valid['stringency_index'].corr(valid['reproduction_rate'])
                    else:
                        c = valid['stringency_index'].iloc[:-lag].corr(
                            valid['reproduction_rate'].iloc[lag:].reset_index(drop=True))
                    corrs.append(c)

                best_lag = list(lags)[corrs.index(min(corrs))]

                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(lags), y=corrs,
                                     marker_color=['#EF553B' if c < 0 else '#636EFA' for c in corrs]))
                fig.update_layout(title='Cross-Correlation: Stringency → Reproduction Rate',
                                  xaxis_title='Lag (days)', yaxis_title='Correlation', height=400)
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                col1.metric("Optimal Lag", f"{best_lag} days")
                col2.metric("Peak Correlation", f"{min(corrs):.3f}")


# ==================================================================
# CROSS-COUNTRY COMPARISON PAGE
# ==================================================================
elif page == "Cross-Country Comparison":
    st.title("Cross-Country Comparison")

    st.markdown("Compare COVID-19 metrics across multiple countries.")

    if df is not None:
        all_countries = sorted(df['location'].unique())
        defaults = ['United States', 'United Kingdom', 'Germany']
        defaults = [c for c in defaults if c in all_countries] or all_countries[:3]

        selected_countries = st.multiselect("Select countries (2-5 recommended):",
                                            all_countries, default=defaults)

        if not selected_countries:
            st.warning("Select at least one country.")
        else:
            metric_labels = {
                'new_cases_smoothed_per_million': 'Cases per Million (7-day avg)',
                'new_deaths_smoothed_per_million': 'Deaths per Million (7-day avg)',
                'stringency_index': 'Stringency Index',
                'reproduction_rate': 'Reproduction Rate (R)',
                'icu_patients_per_million': 'ICU Patients per Million'
            }
            metric = st.selectbox("Select metric:", list(metric_labels.keys()))

            st.subheader(f"Comparison: {metric_labels[metric]}")
            fig = go.Figure()
            for country in selected_countries:
                cdata = df[df['location'] == country].sort_values('date')
                fig.add_trace(go.Scatter(x=cdata['date'], y=cdata[metric],
                                         name=country, mode='lines'))
            fig.update_layout(title=f'{metric_labels[metric]} — Multi-Country',
                              xaxis_title='Date', yaxis_title=metric_labels[metric],
                              height=600, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Summary Statistics")
            summary_rows = []
            for country in selected_countries:
                cdf = df[df['location'] == country]
                summary_rows.append({
                    'Country': country,
                    'Mean': cdf[metric].mean(),
                    'Max': cdf[metric].max(),
                    'Latest': cdf[metric].iloc[-1] if len(cdf) > 0 else None
                })
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df.style.format({'Mean': '{:.2f}', 'Max': '{:.2f}', 'Latest': '{:.2f}'}),
                         use_container_width=True)
    else:
        st.error("Could not load data.")

# ==================================================================
# ABOUT PAGE
# ==================================================================
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ### Exploratory COVID-19 Modeling and Analysis

    This project analyzes COVID-19 pandemic dynamics across three dimensions:

    1. **Healthcare Strain** — Gradient Boosting regression predicting ICU utilization
       from lagged epidemiological and demographic features.
    2. **Pandemic Fatigue** — Logistic Regression classifying periods where cases rise
       despite high-stringency restrictions.
    3. **Policy Effectiveness** — Cross-correlation, Granger causality, and wavelet
       coherence quantifying the lag between policy changes and outcomes.

    ### Data Source

    [Our World in Data COVID-19 Dataset](https://ourworldindata.org/covid-cases)
    (Oxford COVID-19 Government Response Tracker for stringency index).

    ### Author

    Franklin Fuchs

    ### License

    MIT License — see [LICENSE](https://github.com/yourusername/Exploratory-Covid-Modeling/blob/main/LICENSE).
    """)

# Footer
st.markdown("---")
st.caption("COVID-19 Pandemic Analysis Dashboard · Franklin Fuchs · "
           "[GitHub](https://github.com/yourusername/Exploratory-Covid-Modeling)")
