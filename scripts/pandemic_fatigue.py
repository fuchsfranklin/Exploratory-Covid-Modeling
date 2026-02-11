"""
Pandemic Fatigue Analysis and Prediction

Identifies and classifies periods of "pandemic fatigue" — defined as periods
where cases rise despite sustained high-stringency policy measures, suggesting
reduced public compliance with restrictions.

Operationalization:
- High stringency: stringency_index above a configurable threshold (default >= 60)
- Rising transmission: new_cases_smoothed_per_million increasing over a lookback window

The script preprocesses OWID data, engineers features, trains a classification model,
and saves structured results.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             f1_score, balanced_accuracy_score)
import os
import json
import joblib
from datetime import datetime

RESULTS_BASE_DIR = "results/pandemic_fatigue"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


class PandemicFatiguePredictor:
    """
    Identifies and predicts pandemic fatigue periods using epidemiological
    and policy data from the OWID COVID-19 dataset.
    """

    def __init__(self,
                 data_path='owid-covid-data.csv',
                 model_type='LogisticRegression',
                 tune_hyperparameters=False,
                 fatigue_params=None,
                 country_col='location',
                 date_col='date',
                 run_id=None):
        """
        Args:
            data_path: Path to the OWID CSV.
            model_type: 'LogisticRegression', 'GradientBoosting', or 'RandomForest'.
            tune_hyperparameters: Whether to run GridSearchCV.
            fatigue_params: Dict overriding default fatigue definition thresholds.
            country_col: Column name for country/location.
            date_col: Column name for date.
            run_id: Unique run identifier. Auto-generated if None.
        """
        self.data_path = data_path
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.country_col = country_col
        self.date_col = date_col

        # Fatigue definition parameters
        default_params = {
            'stringency_threshold': 60,
            'case_lookback_window': 14,
            'case_increase_threshold': 0.20,  # 20% increase
            'min_sustained_days': 14,
        }
        self.fatigue_params = default_params
        if fatigue_params:
            self.fatigue_params.update(fatigue_params)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 'tuned' if tune_hyperparameters else 'default'
        self.run_dir = os.path.join(RESULTS_BASE_DIR,
                                    f"{self.run_id}_{model_type}_{suffix}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.data = None
        self.model = None
        self.feature_names = None
        self.evaluation_results = None

    # ------------------------------------------------------------------
    # Data loading & preprocessing
    # ------------------------------------------------------------------

    def load_and_preprocess_data(self):
        """Load OWID data, filter to countries with sufficient data, sort by location+date."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
        df.sort_values([self.country_col, self.date_col], inplace=True)

        # Filter out aggregates
        aggregates = {'World', 'Europe', 'European Union', 'Asia', 'Africa',
                      'North America', 'South America', 'Oceania',
                      'High income', 'Low income', 'Lower middle income',
                      'Upper middle income', 'International'}
        df = df[~df[self.country_col].isin(aggregates)].copy()

        # Keep countries with enough stringency + case data
        required_cols = ['stringency_index', 'new_cases_smoothed_per_million']
        valid_countries = []
        for country, group in df.groupby(self.country_col):
            valid_rows = group[required_cols].dropna(how='any')
            if len(valid_rows) >= 100:
                valid_countries.append(country)

        df = df[df[self.country_col].isin(valid_countries)].copy()
        print(f"Retained {len(valid_countries)} countries with sufficient data.")

        self.data = df
        return df

    # ------------------------------------------------------------------
    # Fatigue definition
    # ------------------------------------------------------------------

    def define_fatigue_periods(self):
        """
        Label each row as fatigue (1) or non-fatigue (0).

        Fatigue = high stringency AND cases rising over the lookback window.
        This captures periods where restrictions are strong but transmission
        is increasing anyway, suggesting reduced compliance.
        """
        if self.data is None:
            raise ValueError("Call load_and_preprocess_data first.")

        df = self.data
        params = self.fatigue_params

        # Rolling average of cases
        df['case_rolling_avg'] = df.groupby(self.country_col)[
            'new_cases_smoothed_per_million'
        ].transform(lambda x: x.rolling(params['case_lookback_window'], min_periods=7).mean())

        # Percent change over the lookback window
        df['case_pct_change'] = df.groupby(self.country_col)[
            'case_rolling_avg'
        ].transform(lambda x: x.pct_change(periods=params['case_lookback_window']))

        # Binary indicators
        df['high_stringency'] = (df['stringency_index'] >= params['stringency_threshold']).astype(int)
        df['rising_cases'] = (df['case_pct_change'] > params['case_increase_threshold']).astype(int)

        # Fatigue = both conditions met
        df['fatigue_indicator'] = (df['high_stringency'] & df['rising_cases']).astype(int)

        n_fatigue = df['fatigue_indicator'].sum()
        n_total = df['fatigue_indicator'].notna().sum()
        print(f"Fatigue periods defined: {n_fatigue} fatigue days out of {n_total} "
              f"({100*n_fatigue/n_total:.1f}%)")

        self.data = df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self):
        """
        Create features for the classification model.

        Features include:
        - Smoothed epidemiological indicators and their z-scores
        - Interaction terms with stringency
        - Rolling volatility measures
        - Vaccination progress
        """
        if self.data is None or 'fatigue_indicator' not in self.data.columns:
            raise ValueError("Call define_fatigue_periods first.")

        df = self.data

        # Base columns to use as features
        epi_cols = [
            'reproduction_rate',
            'new_cases_smoothed_per_million',
            'new_tests_smoothed_per_thousand',
            'stringency_index',
            'people_vaccinated_per_hundred',
            'positive_rate',
        ]

        # Smoothed versions (7-day rolling mean per country)
        for col in epi_cols:
            if col in df.columns:
                smoothed = f'{col}_smoothed'
                df[smoothed] = df.groupby(self.country_col)[col].transform(
                    lambda x: x.rolling(7, min_periods=3).mean()
                )

        smoothed_cols = [f'{c}_smoothed' for c in epi_cols if f'{c}_smoothed' in df.columns]

        # Interaction terms: each smoothed feature × stringency
        for col in smoothed_cols:
            if col != 'stringency_index_smoothed':
                interaction = f'{col}_x_stringency'
                df[interaction] = df[col] * df.get('stringency_index_smoothed', 0)

        # Rolling standard deviation (volatility) over 7 days
        for col in smoothed_cols:
            std_col = f'{col}_roll7_std'
            df[std_col] = df.groupby(self.country_col)[col].transform(
                lambda x: x.rolling(7, min_periods=3).std()
            )

        # Z-scores within each country
        for col in smoothed_cols:
            zscore_col = f'{col}_zscore'
            df[zscore_col] = df.groupby(self.country_col)[col].transform(
                lambda x: (x - x.expanding(min_periods=14).mean()) /
                          (x.expanding(min_periods=14).std() + 1e-10)
            )

        # Stringency volatility (how much policy is changing)
        df['stringency_volatility'] = df.groupby(self.country_col)[
            'stringency_index'
        ].transform(lambda x: x.rolling(14, min_periods=7).std())

        # Collect all engineered feature columns
        feature_cols = []
        for col in df.columns:
            if any(col.endswith(suffix) for suffix in
                   ['_smoothed', '_x_stringency', '_roll7_std', '_zscore', '_volatility']):
                feature_cols.append(col)

        self.feature_names = sorted(feature_cols)
        self.data = df
        print(f"Engineered {len(self.feature_names)} features.")
        return self.feature_names

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_model(self, test_size=0.2):
        """
        Train a classification model to predict fatigue periods.

        Uses a chronological train/test split (no shuffle) to respect
        the time-series nature of the data.
        """
        if self.feature_names is None:
            raise ValueError("Call engineer_features first.")

        df = self.data.dropna(subset=self.feature_names + ['fatigue_indicator']).copy()
        if len(df) == 0:
            raise ValueError("No rows remaining after dropping NaNs.")

        X = df[self.feature_names]
        y = df['fatigue_indicator'].astype(int)

        # Chronological split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
        print(f"Train fatigue rate: {y_train.mean():.3f}, Test fatigue rate: {y_test.mean():.3f}")

        # Build pipeline
        if self.model_type == 'LogisticRegression':
            base_model = LogisticRegression(random_state=42, max_iter=1000,
                                            class_weight='balanced')
            param_grid = {
                'model__C': [0.01, 0.1, 1, 10],
                'model__solver': ['liblinear', 'lbfgs']
            }
        elif self.model_type == 'GradientBoosting':
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5]
            }
        elif self.model_type == 'RandomForest':
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20, None]
            }
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        if self.tune_hyperparameters:
            print(f"Tuning {self.model_type} hyperparameters...")
            tscv = TimeSeriesSplit(n_splits=5)
            grid = GridSearchCV(pipeline, param_grid, cv=tscv,
                                scoring='balanced_accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            print(f"Best params: {best_params}")
        else:
            print(f"Training {self.model_type} with defaults...")
            best_model = pipeline
            best_model.fit(X_train, y_train)
            best_params = 'default'

        # Evaluate
        y_pred = best_model.predict(X_test)
        y_proba = (best_model.predict_proba(X_test)[:, 1]
                   if hasattr(best_model, 'predict_proba') else None)

        evaluation = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
            'f1_fatigue_class': float(f1_score(y_test, y_pred, pos_label=1)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }
        if y_proba is not None:
            try:
                evaluation['roc_auc'] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                evaluation['roc_auc'] = None

        self.model = best_model
        self.evaluation_results = evaluation

        # Extract feature importances / coefficients
        feature_importance = self._extract_feature_importance(best_model)

        # Save everything
        self._save_results(best_params, evaluation, feature_importance,
                           X_test, y_test, y_pred)

        print(f"\nBalanced accuracy: {evaluation['balanced_accuracy']:.4f}")
        if evaluation.get('roc_auc'):
            print(f"ROC AUC: {evaluation['roc_auc']:.4f}")
        print(f"F1 (fatigue class): {evaluation['f1_fatigue_class']:.4f}")

        return evaluation

    def _extract_feature_importance(self, model_pipeline):
        """Extract feature importances or coefficients from the trained pipeline."""
        inner_model = model_pipeline.named_steps['model']

        if hasattr(inner_model, 'feature_importances_'):
            importances = inner_model.feature_importances_
        elif hasattr(inner_model, 'coef_'):
            importances = inner_model.coef_[0]
        else:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', key=abs, ascending=False)

        return importance_df

    # ------------------------------------------------------------------
    # Saving results
    # ------------------------------------------------------------------

    def _save_results(self, best_params, evaluation, feature_importance,
                      X_test, y_test, y_pred):
        """Save model, evaluation, and predictions to the run directory."""

        # Model pipeline
        model_path = os.path.join(self.run_dir, 'model_pipeline.pkl')
        joblib.dump(self.model, model_path)

        # Run details
        details = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'hyperparameters_tuned': self.tune_hyperparameters,
            'best_params': str(best_params),
            'fatigue_params': self.fatigue_params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'evaluation': evaluation,
        }
        with open(os.path.join(self.run_dir, 'run_details.json'), 'w') as f:
            json.dump(details, f, indent=2, default=str)

        # Feature importances
        if feature_importance is not None:
            feature_importance.to_csv(
                os.path.join(self.run_dir, 'feature_coefficients.csv'), index=False
            )

        # Predictions
        pred_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred
        }, index=X_test.index)
        pred_df.to_csv(os.path.join(self.run_dir, 'test_predictions_vs_actual.csv'))

        # Human-readable summary
        lines = [
            "Pandemic Fatigue Prediction — Run Summary",
            "=" * 50,
            f"Run ID: {self.run_id}",
            f"Model: {self.model_type}",
            f"Tuned: {self.tune_hyperparameters}",
            f"Best params: {best_params}",
            "",
            "Fatigue Definition:",
            f"  Stringency threshold: >= {self.fatigue_params['stringency_threshold']}",
            f"  Case increase threshold: > {self.fatigue_params['case_increase_threshold']*100:.0f}% "
            f"over {self.fatigue_params['case_lookback_window']} days",
            "",
            "Evaluation:",
            f"  Balanced accuracy: {evaluation['balanced_accuracy']:.4f}",
            f"  ROC AUC: {evaluation.get('roc_auc', 'N/A')}",
            f"  F1 (fatigue class): {evaluation['f1_fatigue_class']:.4f}",
            f"  Accuracy: {evaluation['accuracy']:.4f}",
        ]

        if feature_importance is not None:
            lines.append("\nTop 10 Features:")
            for _, row in feature_importance.head(10).iterrows():
                lines.append(f"  {row['feature']}: {row['importance']:.4f}")

        with open(os.path.join(self.run_dir, 'run_summary.txt'), 'w') as f:
            f.write('\n'.join(lines))

        print(f"Results saved to: {self.run_dir}")


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    predictor = PandemicFatiguePredictor(
        data_path='owid-covid-data.csv',
        model_type='LogisticRegression',
        tune_hyperparameters=True,
    )

    try:
        predictor.load_and_preprocess_data()
        predictor.define_fatigue_periods()
        predictor.engineer_features()
        predictor.train_model()
    except FileNotFoundError:
        print("Error: owid-covid-data.csv not found.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
