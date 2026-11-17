"""
Reproduces why the original pandemic-fatigue model scored ROC AUC ~0.97.

The original label was:
    fatigue(t) = stringency(t) >= 60  AND  14-day case avg rose > 20% vs 14 days earlier
and the features included smoothed stringency, smoothed cases and their
interactions, all measured at the same day t. The data was sorted by country, so
the unshuffled train_test_split held out the last ~20% of countries alphabetically,
not a later time period.

This script fits two small logistic regressions on that original label, using the
original split:
  A) the label's own two ingredients (stringency today, past 14-day case growth);
  B) the same, plus the original model's top feature family (R x stringency).
If A alone gets close to 0.97, the original score measured the label rule, not
any ability to anticipate fatigue.

Run from the repo root:  python scripts/fatigue_leakage_check.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pandemic_fatigue import AGGREGATES

df = pd.read_csv('owid-covid-data.csv', parse_dates=['date'],
                 usecols=['iso_code', 'location', 'date', 'stringency_index',
                          'new_cases_smoothed_per_million', 'reproduction_rate'])
df = df[~df['location'].isin(AGGREGATES)].sort_values(['location', 'date'])
g = df.groupby('location')

# Original label (copied from the pre-Nov-2026 script)
df['case_rolling_avg'] = g['new_cases_smoothed_per_million'].transform(
    lambda x: x.rolling(14, min_periods=7).mean())
df['case_pct_change'] = df.groupby('location')['case_rolling_avg'].transform(
    lambda x: x.pct_change(periods=14))
df['label'] = ((df['stringency_index'] >= 60) & (df['case_pct_change'] > 0.20)).astype(int)

# Label ingredients as features
df['stringency'] = df['stringency_index']
df['case_growth'] = np.log1p(df['case_pct_change'].clip(-0.99, 50))
df['r_x_stringency'] = df['reproduction_rate'] * df['stringency_index']

out = []
for name, cols in [('A: label ingredients only', ['stringency', 'case_growth']),
                   ('B: ingredients + R x stringency', ['stringency', 'case_growth',
                                                        'r_x_stringency'])]:
    d = df.dropna(subset=cols + ['label'])
    d = d[np.isfinite(d[cols]).all(axis=1)]
    Xtr, Xte, ytr, yte = train_test_split(d[cols], d['label'], test_size=0.2, shuffle=False)
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced',
                                                           max_iter=1000))
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    test_countries = d.loc[Xte.index, 'location']
    out.append(f"{name:<34} test AUC {auc:.3f}  (test = {test_countries.nunique()} "
               f"countries, {test_countries.iloc[0]} .. {test_countries.iloc[-1]})")

print("Original label, original split (last 20% of rows, sorted by country):")
print("\n".join(out))
