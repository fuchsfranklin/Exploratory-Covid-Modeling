"""
Pandemic Fatigue: forward-looking prediction (leakage-corrected version)

Question: on a country-day where restrictions are already strict
(stringency_index >= 60), will cases rise by more than 20% over the NEXT
14 days, despite those restrictions?

Why this version exists (Nov 2026 revision)
-------------------------------------------
The earlier version labelled day t as "fatigue" when stringency(t) >= 60 and
cases had risen > 20% over the PREVIOUS 14 days, and then used smoothed
stringency, smoothed cases and their interactions at day t as features. The
model could largely rebuild the label from its own inputs, which produced a
ROC AUC of about 0.97. Its "chronological" split was also a country split,
because the data was sorted by country before an unshuffled train/test split.

This version:
- defines the target over a future window (t+1 .. t+14) of raw daily cases
  that no feature can see;
- uses only information available on day t (trailing averages, lags, static
  country attributes);
- trains on dates before a cutoff, leaving a 14-day gap so no training label
  window overlaps the test period;
- tests on the later period twice: for countries seen in training, and for a
  held-out 20% of countries never seen in training;
- compares against a persistence baseline (past 14-day case growth);
- reports country-cluster bootstrap confidence intervals.

See scripts/fatigue_leakage_check.py for a reproduction of the old setup.
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_BASE_DIR = "results/pandemic_fatigue"

AGGREGATES = {'World', 'Europe', 'European Union', 'Asia', 'Africa',
              'North America', 'South America', 'Oceania', 'High income',
              'Low income', 'Lower middle income', 'Upper middle income',
              'International'}

USECOLS = ['iso_code', 'location', 'date', 'stringency_index',
           'new_cases_per_million',
           'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
           'reproduction_rate', 'positive_rate', 'new_tests_smoothed_per_thousand',
           'people_vaccinated_per_hundred', 'population_density', 'median_age',
           'gdp_per_capita', 'hospital_beds_per_thousand']

STATIC_FEATURES = ['population_density', 'median_age', 'gdp_per_capita',
                   'hospital_beds_per_thousand']

DEFAULT_PARAMS = {
    'stringency_threshold': 60,      # "strict restrictions" on day t
    'horizon_days': 14,              # length of past and future case windows
    'case_increase_threshold': 0.20, # future window avg > 1.2 x past window avg
    'min_case_level': 1.0,           # past 14-day avg cases per million; avoids
                                     # huge % changes from near-zero counts
    'test_start': '2021-07-01',      # first test date
    'holdout_country_frac': 0.20,
    'random_state': 42,
}

MODEL_TYPES = ['LogisticRegression', 'GradientBoosting', 'RandomForest']


# ----------------------------------------------------------------------
# Data and target
# ----------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path, usecols=USECOLS, parse_dates=['date'])
    df = df[~df['location'].isin(AGGREGATES)]
    df = df[~df['iso_code'].astype(str).str.startswith('OWID_')]
    df = df.sort_values(['location', 'date']).reset_index(drop=True)
    return df


def _g(df, col):
    return df.groupby('location', sort=False)[col]


def build_target_and_features(df, params):
    """Add the forward-looking target and day-t features. Rows are country-days."""
    h = params['horizon_days']
    df = df.copy()

    # Target windows use RAW daily cases so they do not overlap:
    #   past   = mean of raw cases over days t-13 .. t   (known on day t)
    #   future = mean of raw cases over days t+1 .. t+14 (never seen by features)
    # (OWID's *_smoothed series is a trailing 7-day mean, so a future window
    # built from it would still contain six days already known on day t.)
    # Two full weeks per window also cancel day-of-week reporting patterns.
    raw = df['new_cases_per_million'].clip(lower=0)
    df['raw_cases'] = raw
    df['case_avg_past'] = _g(df, 'raw_cases').transform(
        lambda x: x.rolling(h, min_periods=h).mean())
    df['case_avg_future'] = _g(df, 'case_avg_past').shift(-h)
    df['future_growth'] = df['case_avg_future'] / df['case_avg_past'] - 1

    df['strict_today'] = df['stringency_index'] >= params['stringency_threshold']
    df['eligible'] = (df['strict_today']
                      & (df['case_avg_past'] >= params['min_case_level'])
                      & df['future_growth'].notna())
    df['target'] = (df['future_growth'] > params['case_increase_threshold']).astype(int)

    # ---- Features: only information available on day t ----
    eps = 0.1
    cases7 = _g(df, 'new_cases_smoothed_per_million').transform(
        lambda x: x.rolling(7, min_periods=4).mean())
    df['f_log_cases_14d'] = np.log1p(df['case_avg_past'])
    df['f_case_growth_7d'] = np.log((cases7 + eps)
                                    / (cases7.groupby(df['location']).shift(7) + eps))
    df['f_case_growth_14d'] = np.log((df['case_avg_past'] + eps)
                                     / (_g(df, 'case_avg_past').shift(h) + eps))
    deaths14 = _g(df, 'new_deaths_smoothed_per_million').transform(
        lambda x: x.rolling(h, min_periods=h // 2).mean())
    df['f_log_deaths_14d'] = np.log1p(deaths14)
    df['f_death_growth_14d'] = np.log((deaths14 + 0.01)
                                      / (deaths14.groupby(df['location']).shift(h) + 0.01))

    df['f_stringency'] = df['stringency_index']
    df['f_stringency_change_7d'] = df['stringency_index'] - _g(df, 'stringency_index').shift(7)
    df['f_stringency_change_28d'] = df['stringency_index'] - _g(df, 'stringency_index').shift(28)
    df['f_stringency_mean_90d'] = _g(df, 'stringency_index').transform(
        lambda x: x.rolling(90, min_periods=30).mean())

    # OWID's reproduction-rate estimate is smoothed and may draw on data after
    # day t, so it is lagged by 14 days to keep it clearly in the past.
    df['f_reproduction_rate_lag14'] = _g(df, 'reproduction_rate').shift(14)
    for col in ['positive_rate', 'new_tests_smoothed_per_thousand',
                'people_vaccinated_per_hundred']:
        df[f'f_{col}'] = _g(df, col).transform(lambda x: x.ffill())
    pre_rollout = df['date'] < pd.Timestamp('2020-12-08')
    df.loc[pre_rollout, 'f_people_vaccinated_per_hundred'] = (
        df.loc[pre_rollout, 'f_people_vaccinated_per_hundred'].fillna(0.0))
    for col in STATIC_FEATURES:
        df[f'f_{col}'] = df[col]
    doy = df['date'].dt.day_of_year
    df['f_season_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['f_season_cos'] = np.cos(2 * np.pi * doy / 365.25)

    feature_cols = sorted(c for c in df.columns if c.startswith('f_'))
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df, feature_cols


def split(df, params):
    """Time split with a gap, plus a held-out set of countries."""
    h = params['horizon_days']
    test_start = pd.Timestamp(params['test_start'])
    train_end = test_start - pd.Timedelta(days=h)   # label windows end before test_start

    data = df[df['eligible']].copy()
    countries = np.array(sorted(data['location'].unique()))
    rng = np.random.default_rng(params['random_state'])
    n_hold = int(round(params['holdout_country_frac'] * len(countries)))
    holdout = set(rng.choice(countries, size=n_hold, replace=False))
    data['holdout_country'] = data['location'].isin(holdout)

    train = data[(~data['holdout_country']) & (data['date'] < train_end)]
    test_seen = data[(~data['holdout_country']) & (data['date'] >= test_start)]
    test_unseen = data[(data['holdout_country']) & (data['date'] >= test_start)]
    return train, test_seen, test_unseen, sorted(holdout)


# ----------------------------------------------------------------------
# Models and metrics
# ----------------------------------------------------------------------

def make_model(model_type, random_state):
    if model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
        steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                 ('scaler', StandardScaler()), ('model', model)]
    elif model_type == 'GradientBoosting':
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_leaf_nodes=31, min_samples_leaf=100,
                                               class_weight='balanced',
                                               random_state=random_state)
        steps = [('model', model)]   # handles missing values natively
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                       class_weight='balanced', n_jobs=-1,
                                       random_state=random_state)
        steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                 ('model', model)]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline(steps)


def point_metrics(y, score, threshold=0.5):
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int)
    return {
        'n': int(len(y)),
        'n_countries': None,
        'positive_rate': float(np.mean(y)),
        'roc_auc': float(roc_auc_score(y, score)),
        'average_precision': float(average_precision_score(y, score)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'f1_fatigue_class': float(f1_score(y, pred, zero_division=0)),
        'accuracy': float(accuracy_score(y, pred)),
    }


def cluster_bootstrap_ci(y, score, groups, n_boot=500, seed=0):
    """95% CI for ROC AUC and average precision, resampling whole countries."""
    rng = np.random.default_rng(seed)
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    aucs, aps = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
        aps.append(average_precision_score(y[idx], score[idx]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {'roc_auc_95ci': q(aucs), 'average_precision_95ci': q(aps), 'n_boot': len(aucs)}


def evaluate(y, score, groups, threshold=0.5, n_boot=500):
    m = point_metrics(y, score, threshold)
    m['n_countries'] = int(pd.Series(groups).nunique())
    m.update(cluster_bootstrap_ci(y, score, groups, n_boot=n_boot))
    return m


# ----------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------

def run(data_path='owid-covid-data.csv', model_types=None, params=None,
        n_boot=500, results_dir=RESULTS_BASE_DIR):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    model_types = model_types or MODEL_TYPES

    print(f"Loading {data_path} ...")
    raw = load_data(data_path)
    df, feats = build_target_and_features(raw, p)
    train, test_seen, test_unseen, holdout = split(df, p)
    print(f"Train {len(train)} rows / {train['location'].nunique()} countries "
          f"({train['date'].min().date()} to {train['date'].max().date()}), "
          f"positive rate {train['target'].mean():.3f}")
    print(f"Test (seen countries) {len(test_seen)} rows; "
          f"test (unseen countries) {len(test_unseen)} rows")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"{run_id}_forward_target")
    os.makedirs(run_dir, exist_ok=True)

    results = {'run_id': run_id, 'timestamp': datetime.now().isoformat(),
               'design': 'forward 14-day target, day-t features, time split with '
                         '14-day gap, held-out countries, country-cluster bootstrap',
               'params': p, 'features': feats,
               'train': {'n': int(len(train)),
                         'n_countries': int(train['location'].nunique()),
                         'date_min': str(train['date'].min().date()),
                         'date_max': str(train['date'].max().date()),
                         'positive_rate': float(train['target'].mean())},
               'holdout_countries': holdout, 'models': {}}

    tests = [('test_seen_countries', test_seen), ('test_unseen_countries', test_unseen)]

    # Persistence baseline: "cases rose over the past 14 days, so they will keep
    # rising". The score is past 14-day log growth; threshold = 20% growth.
    base = {}
    for name, part in tests:
        s = part['f_case_growth_14d'].fillna(0).values
        base[name] = evaluate(part['target'], s, part['location'],
                              threshold=np.log(1.2), n_boot=n_boot)
    results['models']['Baseline_past_growth'] = base

    X_tr, y_tr = train[feats], train['target']
    for mt in model_types:
        print(f"Fitting {mt} ...")
        model = make_model(mt, p['random_state'])
        model.fit(X_tr, y_tr)
        res = {}
        for name, part in tests:
            score = model.predict_proba(part[feats])[:, 1]
            res[name] = evaluate(part['target'], score, part['location'], n_boot=n_boot)
        results['models'][mt] = res
        joblib.dump(model, os.path.join(run_dir, f'model_{mt}.pkl'))
        print(f"  {mt}: AUC seen {res['test_seen_countries']['roc_auc']:.3f}, "
              f"unseen {res['test_unseen_countries']['roc_auc']:.3f}")

    # Permutation importance for the best model on the seen-country test set
    best = max(model_types,
               key=lambda m: results['models'][m]['test_seen_countries']['roc_auc'])
    model = joblib.load(os.path.join(run_dir, f'model_{best}.pkl'))
    sample = test_seen.sample(min(20000, len(test_seen)), random_state=p['random_state'])
    pi = permutation_importance(model, sample[feats], sample['target'],
                                scoring='roc_auc', n_repeats=5,
                                random_state=p['random_state'], n_jobs=-1)
    imp = (pd.DataFrame({'feature': feats, 'importance': pi.importances_mean,
                         'std': pi.importances_std})
           .sort_values('importance', ascending=False))
    imp.to_csv(os.path.join(run_dir, 'feature_coefficients.csv'), index=False)
    results['best_model'] = best

    # Dashboard reads the 'evaluation' block of the latest run
    ev = results['models'][best]['test_seen_countries']
    results['evaluation'] = {k: ev[k] for k in
                             ['balanced_accuracy', 'roc_auc', 'f1_fatigue_class',
                              'accuracy', 'average_precision']}

    with open(os.path.join(run_dir, 'run_details.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    write_summary(results, imp, run_dir)
    print(f"Results saved to: {run_dir}")
    return results


def write_summary(results, imp, run_dir):
    p = results['params']
    lines = ["Pandemic Fatigue (forward target) - Run Summary", "=" * 60,
             f"Run ID: {results['run_id']}",
             f"Design: {results['design']}",
             f"Target: stringency >= {p['stringency_threshold']} today AND next "
             f"{p['horizon_days']}-day avg cases > {1 + p['case_increase_threshold']:.2f}x "
             f"past {p['horizon_days']}-day avg (past avg >= {p['min_case_level']}/M)",
             f"Train: {results['train']['n']} rows, {results['train']['n_countries']} "
             f"countries, {results['train']['date_min']} to {results['train']['date_max']}, "
             f"positive rate {results['train']['positive_rate']:.3f}",
             f"Test period starts: {p['test_start']}", "",
             f"{'Model':<22}{'Test set':<23}{'n':>7}{'AUC':>7}  {'95% CI':<14}"
             f"{'AP':>6}{'Prev':>6}{'BalAcc':>7}{'F1':>6}"]
    for m, res in results['models'].items():
        for name, ev in res.items():
            ci = ev['roc_auc_95ci']
            lines.append(f"{m:<22}{name:<23}{ev['n']:>7}{ev['roc_auc']:>7.3f}  "
                         f"[{ci[0]:.3f},{ci[1]:.3f}]{ev['average_precision']:>6.3f}"
                         f"{ev['positive_rate']:>6.3f}{ev['balanced_accuracy']:>7.3f}"
                         f"{ev['f1_fatigue_class']:>6.3f}")
    lines += ["", "AP = average precision; Prev = share of positives (the AP of a random guess).",
              "95% CI = country-cluster bootstrap for ROC AUC.",
              f"Top permutation importances ({results['best_model']}, drop in AUC):"]
    for _, r in imp.head(10).iterrows():
        lines.append(f"  {r['feature']}: {r['importance']:.4f} (sd {r['std']:.4f})")
    with open(os.path.join(run_dir, 'run_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Forward-looking pandemic fatigue prediction")
    ap.add_argument('--data', default='owid-covid-data.csv')
    ap.add_argument('--models', nargs='+', default=MODEL_TYPES, choices=MODEL_TYPES)
    ap.add_argument('--test-start', default=DEFAULT_PARAMS['test_start'])
    ap.add_argument('--n-boot', type=int, default=500)
    a = ap.parse_args()
    run(a.data, a.models, {'test_start': a.test_start}, n_boot=a.n_boot)

