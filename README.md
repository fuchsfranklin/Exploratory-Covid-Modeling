# Exploratory COVID-19 modeling

Three questions, answered with the Our World in Data COVID-19 dataset (daily, by country, 2020 to 2024):

1. Can ICU load be predicted a week ahead?
2. On days with strict restrictions, can we tell whether cases will rise over the next two weeks anyway? (This is what people called "pandemic fatigue".)
3. How long does a change in policy stringency take to show up in cases, deaths and the reproduction rate?

In November 2026 I re-audited the project, re-ran everything from a clean environment, and compared each model against a simple baseline. Two of the three original headline numbers didn't survive. This README reports the audited results. [CHANGELOG.md](CHANGELOG.md) has the before-and-after.

## Results

| Question | Model | Simple baseline | Verdict |
|---|---|---|---|
| ICU, one week ahead | MAE 0.61 patients per million | ICU value 7 days earlier: MAE 0.52 | Doesn't beat persistence |
| Cases rising despite strict restrictions, next 14 days | ROC AUC 0.86 (95% CI 0.83 to 0.88) | Past 14-day case growth: AUC 0.77 | Clear gain over the baseline |
| Lag from policy change to effect | Median about 11.5 days (R), 13 days (deaths) | none | Consistent for R only |

### ICU strain

The first version reported MAE 5.6 without a reference point. Just carrying last week's ICU number forward scores 0.52, so that model was about ten times worse than persistence. Adding ICU's own lagged values and fitting the week-over-week change brought it to 0.61. That's much closer, but still not better.

Part of the reason is the test period. It runs from November 2022 to August 2024, when ICU levels averaged 3 per million (versus 19 in training) and moved slowly. A week ahead, there isn't much for a model to add. A version worth using would need a longer horizon or admissions data, which OWID has for only a few countries. `scripts/icu_model_comparison.py` has the comparison: persistence 0.52, a two-week linear trend 0.63, boosted trees on the weekly change 0.61 (absolute loss) and 0.82 (squared loss).

### Pandemic fatigue

The first version claimed AUC 0.97. The label was "stringency ≥ 60 today and cases already up 20% over two weeks", and both of those went into the model as features. The split also held out the last 20% of countries in alphabetical order instead of holding out later dates. A two-feature model using only the label's ingredients reaches 0.96 on that setup (`scripts/fatigue_leakage_check.py`), so the 0.97 told us nothing.

The rebuilt model asks a forward question. On a strict-restriction day, will average daily cases over the next 14 days be more than 20% above the past 14 days? Features are limited to what was known that day. Training covers dates up to 16 June 2021 and testing starts 1 July 2021. A separate test uses 33 countries left out of training entirely.

| Model | AUC, countries seen in training | AUC, held-out countries | Average precision (seen / held out) |
|---|---|---|---|
| Baseline: past 14-day growth | 0.77 | 0.78 | 0.59 / 0.62 |
| Logistic regression | 0.82 | 0.81 | 0.64 / 0.62 |
| Gradient boosting | 0.83 | 0.84 | 0.70 / 0.70 |
| Random forest | 0.86 | 0.87 (CI 0.82 to 0.92) | 0.74 / 0.74 |

About 30% of test days are positive, so random guessing gives 0.30 average precision. Confidence intervals come from resampling whole countries, since days within a country aren't independent. Permutation importance puts recent 7-day case growth far ahead of everything else, and the stringency features barely register. So the model picks up epidemic momentum. It doesn't tell you anything about compliance, and "fatigue" is just a name for the outcome.

### Policy lag

For the reproduction rate, all three methods (cross-correlation, Granger causality, wavelet coherence) agree in 9 of 10 countries, with a median lag around 11 to 12 days. For deaths, only France and Spain have all three agreeing. For cases the estimates range from 0 to 39 days. The script counts a country as "significant" when any one method passes, and the wavelet test (coherence above 0.5) passes for every pair, so the "10/10 significant" in the output overstates things. I'd only lean on the R result.

## How to run

Python 3.11 was used for the numbers above. From the repository root:

```bash
pip install -r requirements.txt
python scripts/get_data.py                # downloads the 98 MB dataset (not kept in git)

python scripts/pandemic_fatigue.py        # ~40 s
python scripts/fatigue_leakage_check.py   # ~5 s, reproduces the original leakage
python scripts/healthcare_strain.py       # ~20 s
python scripts/icu_model_comparison.py    # ~15 s, ICU model vs baselines
python scripts/policy_effectiveness_lag.py  # ~10 s

streamlit run dashboard/app.py            # reads the newest run in results/
```

Each script writes a timestamped folder under `results/` with a `run_summary.txt`. Model `.pkl` files are git-ignored. Rerunning gives the same numbers. Other scikit-learn versions can move the tree-model results slightly (around 0.01 AUC). An older copy of the CSV, rounded to 3 decimals instead of 2, gave ICU MAE 0.57 instead of 0.61, which shows how sensitive that comparison is.

## Data

`owid-covid-data.csv` is the [Our World in Data COVID-19 dataset](https://github.com/owid/covid-19-data/tree/master/public/data). OWID stopped updating it in August 2024 (last date 14 August 2024), so `scripts/get_data.py` always fetches the same file, and it checks the SHA-256 against the copy used here. The CSV is about 98 MB, so it's git-ignored. Its stringency index comes from the Oxford COVID-19 Government Response Tracker, which stops at the end of 2022. Regional and income-group aggregates are removed before modeling.

## Limitations

- These are observational associations. Nothing here supports a causal claim about policy.
- Case counts depend on testing, which fell sharply in 2022. That affects both the fatigue target and the later part of the ICU test set.
- The fatigue hyperparameters were set by hand. The held-out-country estimate rests on one random draw (19 of the 33 countries have strict-restriction days in the test period).
- The policy-lag analysis covers 10 large countries and uses the overall stringency index, not individual policies.

## Layout

```
dashboard/app.py                     Streamlit dashboard
scripts/get_data.py                  downloads the OWID dataset
scripts/healthcare_strain.py         ICU model (one week ahead)
scripts/icu_model_comparison.py      ICU model vs persistence and trend baselines
scripts/pandemic_fatigue.py          forward-looking fatigue classifier
scripts/fatigue_leakage_check.py     shows why the original 0.97 AUC was invalid
scripts/policy_effectiveness_lag.py  CCF, Granger and wavelet lag estimates
example_analysis.ipynb               quick plots for one country
results/                             run outputs (summaries, importances, predictions)
```

MIT license. Franklin Fuchs.
