"""
Healthcare System Strain Prediction

Predicts ICU patients per million one week ahead. Inputs are values from 7 and
14 days earlier (including ICU itself), lagged rolling means, and country
demographics. Nothing from the prediction day is used.

Design notes:
- By default the model fits the week-over-week change in ICU and adds it back to
  last week's value (predict_change=True). Fitting the level directly lost to
  plain persistence; see scripts/icu_model_comparison.py.
- Split is by date: the last 20% of dated rows (Nov 2022 onward) are the test set.
- Imputation/scaling (for the sklearn tree models) sit inside the Pipeline, so they
  are fitted on training rows only.
- Every run reports two baselines next to the model MAE: last week's ICU value, and
  the training-period median.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
import os
import joblib
import datetime
import json

RESULTS_BASE_DIR = "results/healthcare_strain"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


class HealthcareStrainPredictor:
    """
    Predicts healthcare strain (ICU patients per million) using configurable
    regression models. All dynamic inputs are lagged by at least 7 days.
    """

    def __init__(self,
                 base_dynamic_feature_cols,
                 static_feature_cols,
                 target_col,
                 lag_periods=None,
                 rolling_avg_windows=None,
                 model_type='GradientBoosting',
                 use_hyperparameter_tuning=False,
                 n_neighbors_imputation=5,
                 cv_splits=3,
                 predict_change=False):
        """
        Args:
            base_dynamic_feature_cols: Time-varying columns to create lagged features from.
            static_feature_cols: Demographic/structural columns (not lagged).
            target_col: Target variable column name.
            lag_periods: List of lag periods in days (default [7, 14]).
            rolling_avg_windows: Rolling average window sizes (default [7, 14]).
            model_type: 'GradientBoosting' or 'RandomForest'.
            use_hyperparameter_tuning: Whether to run GridSearchCV.
            n_neighbors_imputation: KNN imputer neighbors.
            cv_splits: TimeSeriesSplit folds for cross-validation.
            predict_change: If True, the model predicts the change from the target's
                value at the shortest lag, and that value is added back. Requires
                target_col to be in base_dynamic_feature_cols.
            model_type also accepts 'HistGB' (histogram gradient boosting, absolute loss).
        """
        self.predict_change = predict_change
        self.base_dynamic_feature_cols = base_dynamic_feature_cols
        self.static_feature_cols = static_feature_cols
        self.target_col = target_col
        self.lag_periods = lag_periods or [7, 14]
        self.rolling_avg_windows = rolling_avg_windows or [7, 14]
        self.model_type = model_type
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.n_neighbors_imputation = n_neighbors_imputation
        self.cv_splits = cv_splits

        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = 'tuned' if use_hyperparameter_tuning else 'default'
        self.run_results_dir = os.path.join(
            RESULTS_BASE_DIR, f"{self.run_id}_{self.model_type}_{suffix}")
        os.makedirs(self.run_results_dir, exist_ok=True)

        self.feature_cols = []
        self.model_pipeline = None
        self.best_params_ = None
        self.feature_importances_ = None

        self.data = None
        self.X_train_df = None
        self.y_train_series = None
        self.X_test_df = None
        self.y_test_series = None
        self.train_indices = None
        self.test_indices = None

    # ------------------------------------------------------------------
    # Data loading & feature engineering
    # ------------------------------------------------------------------

    def load_and_preprocess_data(self, csv_path):
        """
        Load data, engineer lagged and rolling features, split chronologically.

        IMPORTANT: Only lagged versions of dynamic features are used as predictors.
        Contemporaneous dynamic values are excluded to prevent data leakage.
        """
        df_orig = pd.read_csv(csv_path)
        df = df_orig.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(['location', 'date'], inplace=True)

        # Start with static features only
        current_features = list(self.static_feature_cols)

        # Create LAGGED features (these are the predictive dynamic features)
        for col in self.base_dynamic_feature_cols:
            for lag in self.lag_periods:
                lagged_name = f'{col}_lag{lag}'
                df[lagged_name] = df.groupby('location')[col].shift(lag)
                current_features.append(lagged_name)

        # Create LAGGED rolling averages
        for col in self.base_dynamic_feature_cols:
            for window in self.rolling_avg_windows:
                roll_name = f'{col}_roll_avg{window}'
                df[roll_name] = df.groupby('location')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                # Lag the rolling average too (by the min lag) so it can't see the target day
                lagged_roll_name = f'{roll_name}_lag{min(self.lag_periods)}'
                df[lagged_roll_name] = df.groupby('location')[roll_name].shift(min(self.lag_periods))
                current_features.append(lagged_roll_name)

        self.feature_cols = sorted(list(set(current_features)))

        # Convert to numeric
        for col in self.feature_cols + [self.target_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with missing target or all-NaN features
        df.dropna(subset=[self.target_col], inplace=True)
        df.dropna(subset=self.feature_cols, how='all', inplace=True)
        self._lag_col = f'{self.target_col}_lag{min(self.lag_periods)}'
        if self.predict_change:
            # need last week's value to add the predicted change back to
            df.dropna(subset=[self._lag_col], inplace=True)

        # Keep original data aligned for saving predictions with metadata
        self.data = df_orig.loc[df.index].copy()
        for f_col in self.feature_cols:
            if f_col in df.columns:
                self.data[f_col] = df[f_col]

        if df.empty:
            raise ValueError("No data remaining after preprocessing.")

        # Chronological split (last 20% as test)
        df_sorted = df.sort_values('date')
        train_df, test_df = train_test_split(df_sorted, test_size=0.2, shuffle=False)

        self.X_train_df = train_df[self.feature_cols]
        self.y_train_series = train_df[self.target_col]
        self.X_test_df = test_df[self.feature_cols]
        self.y_test_series = test_df[self.target_col]
        self.train_indices = self.X_train_df.index
        self.test_indices = self.X_test_df.index

        if self.X_train_df.empty or self.X_test_df.empty:
            raise ValueError("Train or test set is empty after split.")

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_model(self):
        """Train the regression model (with optional hyperparameter tuning)."""
        if self.X_train_df is None:
            raise ValueError("Call load_and_preprocess_data first.")

        # Pipeline: impute → scale → regress
        if self.model_type == 'GradientBoosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__max_depth': [3, 5]
            }
        elif self.model_type == 'RandomForest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [10, 20, None]
            }
        elif self.model_type == 'HistGB':
            # Handles missing values itself, so no imputer; absolute-error loss
            # matches the MAE we report. Best of the variants in icu_model_comparison.py.
            model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=300,
                                                  learning_rate=0.05, min_samples_leaf=50,
                                                  random_state=42)
            param_grid = {
                'regressor__learning_rate': [0.03, 0.05, 0.1],
                'regressor__min_samples_leaf': [20, 50, 100]
            }
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        if self.model_type == 'HistGB':
            pipeline = Pipeline([('regressor', model)])
        else:
            pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=self.n_neighbors_imputation)),
                ('scaler', MinMaxScaler()),
                ('regressor', model)
            ])

        y_fit = self.y_train_series
        if self.predict_change:
            y_fit = self.y_train_series - self.X_train_df[self._lag_col]

        if self.use_hyperparameter_tuning:
            print(f"Tuning {self.model_type} hyperparameters...")
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            grid = GridSearchCV(pipeline, param_grid, cv=tscv,
                                scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
            grid.fit(self.X_train_df, y_fit)
            self.model_pipeline = grid.best_estimator_
            self.best_params_ = grid.best_params_
            print(f"Best parameters: {self.best_params_}")
        else:
            print(f"Training {self.model_type} with defaults...")
            self.model_pipeline = pipeline
            self.model_pipeline.fit(self.X_train_df, y_fit)
            self.best_params_ = "default"

        # Feature importances (impurity-based if the model has them, otherwise
        # permutation importance on a training sample; normalised to sum to 1)
        regressor = self.model_pipeline.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            imp = regressor.feature_importances_
        else:
            sample = self.X_train_df.sample(min(5000, len(self.X_train_df)), random_state=42)
            pi = permutation_importance(self.model_pipeline, sample, y_fit.loc[sample.index],
                                        scoring='neg_mean_absolute_error', n_repeats=3,
                                        random_state=42, n_jobs=-1)
            imp = np.clip(pi.importances_mean, 0, None)
            imp = imp / imp.sum() if imp.sum() > 0 else imp
        self.feature_importances_ = pd.Series(
            imp, index=self.X_train_df.columns).sort_values(ascending=False)

        print("Training complete.")

    # ------------------------------------------------------------------
    # Prediction & evaluation
    # ------------------------------------------------------------------

    def predict(self, X_df):
        """Make predictions using the trained pipeline."""
        if self.model_pipeline is None:
            raise ValueError("Model not trained.")
        preds = self.model_pipeline.predict(X_df[self.feature_cols])
        if self.predict_change:
            preds = np.clip(preds + X_df[self._lag_col].values, 0, None)
        return preds

    def evaluate_model(self):
        """Evaluate on the held-out test set. Returns (MAE, RMSE)."""
        if self.model_pipeline is None:
            raise ValueError("Model not trained.")
        preds = self.predict(self.X_test_df)
        mae = mean_absolute_error(self.y_test_series, preds)
        rmse = np.sqrt(mean_squared_error(self.y_test_series, preds))
        self.baselines_ = self._baseline_maes(preds)
        return mae, rmse

    def _baseline_maes(self, preds):
        """MAE of naive forecasts on the same test rows, so the model MAE has a reference.

        - persistence: ICU value 7 days earlier (same 7-day horizon as the model)
        - train_median: one constant, the median ICU level in the training period
        """
        y = self.y_test_series
        out = {'train_median': float(mean_absolute_error(
            y, np.full(len(y), self.y_train_series.median())))}
        lag_col = f'{self.target_col}_lag{min(self.lag_periods)}'
        if lag_col in self.X_test_df.columns:
            mask = self.X_test_df[lag_col].notna().values
            out['persistence_lag7'] = float(mean_absolute_error(
                y[mask], self.X_test_df.loc[mask, lag_col]))
            out['model_on_same_rows'] = float(mean_absolute_error(y[mask], preds[mask]))
            out['n_rows_with_lag'] = int(mask.sum())
        return out

    # ------------------------------------------------------------------
    # Saving results
    # ------------------------------------------------------------------

    def save_run_results(self, mae, rmse, predictions):
        """Save all artifacts for this run."""
        # Run details JSON
        details = {
            'run_id': self.run_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_type': self.model_type,
            'target_column': self.target_col,
            'n_features': len(self.feature_cols),
            'feature_cols': self.feature_cols,
            'lag_periods': self.lag_periods,
            'rolling_avg_windows': self.rolling_avg_windows,
            'hyperparameter_tuning': self.use_hyperparameter_tuning,
            'predict_change': self.predict_change,
            'best_params': str(self.best_params_),
            'mae': mae,
            'rmse': rmse,
            'train_shape': list(self.X_train_df.shape),
            'test_shape': list(self.X_test_df.shape),
            'test_date_range': [str(self.data.loc[self.test_indices, 'date'].min())[:10],
                                str(self.data.loc[self.test_indices, 'date'].max())[:10]],
            'baseline_mae': getattr(self, 'baselines_', None),
        }
        with open(os.path.join(self.run_results_dir, 'run_details.json'), 'w') as f:
            json.dump(details, f, indent=2)

        # Feature importances
        if self.feature_importances_ is not None:
            self.feature_importances_.to_csv(
                os.path.join(self.run_results_dir, 'feature_importances.csv'),
                header=['importance']
            )

        # Predictions vs actual
        pred_df = self.data.loc[self.test_indices, ['date', 'location', self.target_col]].copy()
        pred_df.rename(columns={self.target_col: 'actual'}, inplace=True)
        pred_df['predicted'] = np.array(predictions).flatten()
        pred_df.to_csv(os.path.join(self.run_results_dir, 'test_predictions_vs_actual.csv'),
                       index=False)

        # Model pipeline
        joblib.dump(self.model_pipeline,
                    os.path.join(self.run_results_dir, 'model_pipeline.pkl'))

        # Human-readable summary
        summary = self._generate_summary(mae, rmse)
        with open(os.path.join(self.run_results_dir, 'run_summary.txt'), 'w') as f:
            f.write(summary)

        print(f"\nResults saved to: {self.run_results_dir}")
        print(summary)

    def _generate_summary(self, mae, rmse):
        lines = [
            "Healthcare Strain Prediction - Run Summary",
            "=" * 50,
            f"Run ID: {self.run_id}",
            f"Model: {self.model_type}",
            f"Target: {self.target_col}"
            + (f" (model fits change vs {self._lag_col})" if self.predict_change else ""),
            f"Tuned: {self.use_hyperparameter_tuning}",
            f"Best params: {self.best_params_}",
            f"Features: {len(self.feature_cols)}",
            f"Train: {self.X_train_df.shape[0]} rows, Test: {self.X_test_df.shape[0]} rows",
            "",
            "Evaluation:",
            f"  MAE:  {mae:.4f} (avg error in ICU patients per million)",
            f"  RMSE: {rmse:.4f}",
        ]
        b = getattr(self, 'baselines_', None) or {}
        if b:
            lines.append("  Baselines (MAE on the same test rows):")
            if 'persistence_lag7' in b:
                lines.append(f"    ICU 7 days earlier: {b['persistence_lag7']:.4f} "
                             f"(model on those {b['n_rows_with_lag']} rows: "
                             f"{b['model_on_same_rows']:.4f})")
            lines.append(f"    Training-period median: {b['train_median']:.4f}")
        lines += [
            "",
            "Note: predictors are values from 7 or more days before the prediction date.",
        ]

        if self.feature_importances_ is not None:
            lines.append("\nTop 10 Features:")
            for feat, imp in self.feature_importances_.head(10).items():
                lines.append(f"  {feat}: {imp:.4f} ({imp*100:.1f}%)")

        return '\n'.join(lines)

    def save_model(self, path):
        if self.model_pipeline is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model_pipeline, path)

    def load_model(self, path):
        self.model_pipeline = joblib.load(path)


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    MODEL_CHOICE = 'HistGB'
    PERFORM_TUNING = False
    LAG_PERIODS = [7, 14]
    ROLLING_WINDOWS = [7, 14]

    target = 'icu_patients_per_million'

    # Dynamic features. These get lagged before use as predictors.
    # The target's own history (ICU 7 and 14 days ago) is included, and the model fits
    # the week-over-week change (predict_change=True). Without these, the model lost
    # badly to simply carrying last week's ICU value forward (Nov 2026 check).
    base_dynamic_features = [
        'icu_patients_per_million',
        'new_cases_smoothed_per_million',
        'new_deaths_smoothed_per_million',
        'reproduction_rate',
        'hosp_patients_per_million',
        'new_tests_smoothed_per_thousand',
        'positive_rate',
        'people_fully_vaccinated_per_hundred',
        'total_boosters_per_hundred',
        'stringency_index'
    ]

    # Static features (demographics, structural)
    static_features = [
        'population_density',
        'median_age',
        'aged_65_older',
        'gdp_per_capita',
        'extreme_poverty',
        'cardiovasc_death_rate',
        'diabetes_prevalence',
        'hospital_beds_per_thousand',
        'life_expectancy',
        'human_development_index'
    ]

    predictor = HealthcareStrainPredictor(
        base_dynamic_feature_cols=base_dynamic_features,
        static_feature_cols=static_features,
        target_col=target,
        lag_periods=LAG_PERIODS,
        rolling_avg_windows=ROLLING_WINDOWS,
        model_type=MODEL_CHOICE,
        use_hyperparameter_tuning=PERFORM_TUNING,
        predict_change=True,
    )

    csv_path = 'owid-covid-data.csv'

    try:
        print(f"Run ID: {predictor.run_id}")
        print(f"Loading data from {csv_path}...")
        predictor.load_and_preprocess_data(csv_path)
        print(f"Train: {predictor.X_train_df.shape}, Test: {predictor.X_test_df.shape}")
        print(f"Features: {len(predictor.feature_cols)}")

        predictor.train_model()

        mae, rmse = predictor.evaluate_model()
        predictions = predictor.predict(predictor.X_test_df)
        predictor.save_run_results(mae, rmse, predictions)

    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
