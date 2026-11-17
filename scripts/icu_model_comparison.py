"""
Compare ICU one-week-ahead models against naive baselines on the exact split used by
healthcare_strain.py. Written after a Nov 2026 check showed the original model
(MAE ~5.6) lost badly to "ICU 7 days ago".

Run from the repo root:  python scripts/icu_model_comparison.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
import healthcare_strain as hs  # noqa: E402

DYNAMIC = ['icu_patients_per_million', 'new_cases_smoothed_per_million',
           'new_deaths_smoothed_per_million', 'reproduction_rate', 'hosp_patients_per_million',
           'new_tests_smoothed_per_thousand', 'positive_rate',
           'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
           'stringency_index']
STATIC = ['population_density', 'median_age', 'aged_65_older', 'gdp_per_capita',
          'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence',
          'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']

p = hs.HealthcareStrainPredictor(DYNAMIC, STATIC, 'icu_patients_per_million',
                                 lag_periods=[7, 14], rolling_avg_windows=[7, 14],
                                 predict_change=True)
os.rmdir(p.run_results_dir)  # this script writes no run folder
p.load_and_preprocess_data('owid-covid-data.csv')
Xtr, ytr, Xte, yte = p.X_train_df, p.y_train_series, p.X_test_df, p.y_test_series
lag7_tr, lag7_te = Xtr[p._lag_col], Xte[p._lag_col]
dates = pd.to_datetime(p.data.loc[p.test_indices, 'date'])

rows = [('Persistence: ICU 7 days earlier', mean_absolute_error(yte, lag7_te)),
        ('Training-period median', mean_absolute_error(yte, np.full(len(yte), ytr.median())))]

# Two-step persistence-with-trend: lag7 + (lag7 - lag14)
trend = lag7_te + (lag7_te - Xte['icu_patients_per_million_lag14']).fillna(0)
rows.append(('Linear trend from last two weeks', mean_absolute_error(yte, trend.clip(lower=0))))

for loss in ['absolute_error', 'squared_error']:
    m = HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05,
                                      min_samples_leaf=50, random_state=42)
    m.fit(Xtr, ytr - lag7_tr)
    pred = (lag7_te + m.predict(Xte)).clip(lower=0)
    rows.append((f'HistGB on weekly change ({loss})', mean_absolute_error(yte, pred)))

print(f"Test rows {len(yte)}, {p.data.loc[p.test_indices, 'location'].nunique()} countries, "
      f"{dates.min().date()} to {dates.max().date()}")
print(f"Mean ICU per million in test: {yte.mean():.2f} (train: {ytr.mean():.2f})")
for name, v in rows:
    print(f"  {name:<42} MAE {v:.3f}")
