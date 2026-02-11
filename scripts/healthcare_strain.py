"""
Healthcare System Strain Prediction

Predicts ICU patient counts per million using a Gradient Boosting or Random Forest
regression model. Features include lagged epidemiological indicators, rolling averages,
demographic factors, and policy measures.

Key design decisions:
- Only LAGGED features of dynamic indicators are used as predictors (no contemporaneous
  values) to avoid data leakage and enable genuine forward-looking prediction.
- KNN imputation and MinMax scaling are applied within a sklearn Pipeline to prevent
  train/test leakage.
- Chronological train/test split (last 20%) respects time-series ordering.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import os
import joblib
import datetime
import json

RESULTS_BASE_DIR = "results/healthcare_strain"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


class HealthcareStrainPredictor:
    """
    Predicts healthcare strain (ICU patients per million) using configurable
    regression models with lagged features to enable genuine forecasting.
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
                 cv_splits=3):
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
        """
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
                # Lag the rolling average too — use min lag to avoid leakage
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
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=self.n_neighbors_imputation)),
            ('scaler', MinMaxScaler()),
            ('regressor', model)
        ])

        if self.use_hyperparameter_tuning:
            print(f"Tuning {self.model_type} hyperparameters...")
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            grid = GridSearchCV(pipeline, param_grid, cv=tscv,
                                scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
            grid.fit(self.X_train_df, self.y_train_series)
            self.model_pipeline = grid.best_estimator_
            self.best_params_ = grid.best_params_
            print(f"Best parameters: {self.best_params_}")
        else:
            print(f"Training {self.model_type} with defaults...")
            self.model_pipeline = pipeline
            self.model_pipeline.fit(self.X_train_df, self.y_train_series)
            self.best_params_ = "default"

        # Feature importances
        regressor = self.model_pipeline.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            self.feature_importances_ = pd.Series(
                regressor.feature_importances_,
                index=self.X_train_df.columns
            ).sort_values(ascending=False)

        print("Training complete.")

    # ------------------------------------------------------------------
    # Prediction & evaluation
    # ------------------------------------------------------------------

    def predict(self, X_df):
        """Make predictions using the trained pipeline."""
        if self.model_pipeline is None:
            raise ValueError("Model not trained.")
        return self.model_pipeline.predict(X_df[self.feature_cols])

    def evaluate_model(self):
        """Evaluate on the held-out test set. Returns (MAE, RMSE)."""
        if self.model_pipeline is None:
            raise ValueError("Model not trained.")
        preds = self.model_pipeline.predict(self.X_test_df)
        mae = mean_absolute_error(self.y_test_series, preds)
        rmse = np.sqrt(mean_squared_error(self.y_test_series, preds))
        return mae, rmse

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
            'best_params': str(self.best_params_),
            'mae': mae,
            'rmse': rmse,
            'train_shape': list(self.X_train_df.shape),
            'test_shape': list(self.X_test_df.shape),
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
            "Healthcare Strain Prediction — Run Summary",
            "=" * 50,
            f"Run ID: {self.run_id}",
            f"Model: {self.model_type}",
            f"Target: {self.target_col}",
            f"Tuned: {self.use_hyperparameter_tuning}",
            f"Best params: {self.best_params_}",
            f"Features: {len(self.feature_cols)}",
            f"Train: {self.X_train_df.shape[0]} rows, Test: {self.X_test_df.shape[0]} rows",
            "",
            "Evaluation:",
            f"  MAE:  {mae:.4f} (avg error in ICU patients per million)",
            f"  RMSE: {rmse:.4f}",
            "",
            "NOTE: Only lagged features are used as predictors (no contemporaneous",
            "dynamic values) to enable genuine forward-looking prediction.",
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
    MODEL_CHOICE = 'GradientBoosting'
    PERFORM_TUNING = False
    LAG_PERIODS = [7, 14]
    ROLLING_WINDOWS = [7, 14]

    target = 'icu_patients_per_million'

    # Dynamic features — will be LAGGED before use as predictors
    base_dynamic_features = [
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
