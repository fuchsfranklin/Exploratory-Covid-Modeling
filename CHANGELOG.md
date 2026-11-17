# Changelog

## November 2026: audit and re-run

I went back through every analysis, re-ran each one from a fresh environment built from `requirements.txt` (Python 3.11.9, pandas 3.1.1, scikit-learn 1.9.1, statsmodels 0.15.0, streamlit 1.67.0), and checked every model against a naive baseline. Three things were wrong.

**1. The fatigue classifier leaked its own label (AUC 0.97 → 0.86).**
The label was built from same-day stringency and past case growth, and both were also features. On top of that, the data was sorted by country before an unshuffled split, so the "test set" was the last 20% of countries alphabetically (Seychelles to Zimbabwe), not a later period. I rewrote `scripts/pandemic_fatigue.py` so the target is the next 14 days of raw cases, features only use what was known that day, training ends 16 June 2021 with a 14-day gap before testing, and 33 countries are held out entirely. It now reports a persistence baseline and country-level bootstrap intervals. `scripts/fatigue_leakage_check.py` reproduces the old setup.

**2. The ICU model lost badly to persistence (MAE 5.6 → 0.61; baseline 0.52).**
Nobody had compared it with "ICU a week ago", which scores 0.52. In `scripts/healthcare_strain.py` I added lagged ICU as an input, had the model predict the weekly change (`predict_change=True`), switched to histogram gradient boosting with absolute-error loss, and made every run print both baselines. It's much better than before but still slightly worse than persistence, and the README says so. `scripts/icu_model_comparison.py` has the variants I tried.

**3. The Granger test was silently failing.**
statsmodels 0.15 removed the `verbose` argument from `grangercausalitytests`. The call raised an error that the script caught and stored, so every Granger result dropped out and the lag estimates shifted by several days. Removing the argument brings back the original numbers exactly. The README line about deaths ("5/10 countries, ~15 days") didn't match any run, so I corrected it.

Smaller fixes:
- The dashboard text matches the new models, and each model is shown next to its baseline.
- Two placeholder `github.com/yourusername` links are fixed.
- The notebook labels its fatigue plot as the descriptive rule, not the model.
- Stale Feb 2026 result folders and committed `__pycache__` files are untracked. They're still in git history.
- The 98 MB `owid-covid-data.csv` is no longer tracked. `scripts/get_data.py` downloads it from OWID and checks a SHA-256. The new file is the official one, rounded to 2 decimals; the old committed copy had 3. That moved the ICU MAE from 0.57 to 0.61 and changed some fatigue AUCs by 0.01 or less. The numbers here come from the downloadable file, so anyone can reproduce them.

### Verification

Last full run: 17 November 2026. A fresh copy of the repo (15 files, 0.11 MB, no dataset) ran `get_data.py` and then all five scripts. A separate run in the working repo also covered the dashboard and the notebook. Both runs gave identical numbers.

| Check | Result |
|---|---|
| `get_data.py` | exit 0; 98.4 MB, checksum matches |
| `pandemic_fatigue.py` | exit 0, ~45 s; random forest AUC 0.857 (seen) / 0.870 (held out) |
| `fatigue_leakage_check.py` | exit 0; label-ingredients-only AUC 0.961 |
| `healthcare_strain.py` | exit 0, ~25 s; MAE 0.605 vs persistence 0.516 |
| `icu_model_comparison.py` | exit 0, ~15 s |
| `policy_effectiveness_lag.py` | exit 0, ~7 s; median lag R 11.5, deaths 13.2, cases 3.0 days |
| Dashboard (Streamlit `AppTest`, all 6 pages) | 0 exceptions; metrics match the run summaries |
| `example_analysis.ipynb` (nbclient) | all cells execute |

Not fixed: pandas `inplace=` deprecation warnings and Streamlit's `use_container_width` warning. Both still work on current versions.
