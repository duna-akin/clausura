# clausura

Early-warning signal for U.S. private not-for-profit college closures, built on public IPEDS data.

## Data

Eleven years of the IPEDS Finance Survey Part F2 (2014 to 2024) plus the 2024 Header file. About 20,000 institution-year rows across ~2,200 schools. The pipeline handles cross-year schema drift, IPEDS imputation flags, and silent closures (institutions that drop off the directory without a `CLOSEDAT` record).

## Baseline result

Cohort: 1,836 private nonprofits that filed F2 in 2019. Label: `closed_by_2024` (7.1% positive base rate). Features (28): most-recent value, 3-year mean, slope, and percent change of F2 measures over 2017 to 2019, strictly before the as-of date so there is no survivorship leakage.

| Model                  | 5-fold CV AUC   | Average Precision |
|------------------------|-----------------|-------------------|
| Logistic regression    | 0.830 ± 0.032   | 0.323 ± 0.064     |
| Gradient-boosted trees | 0.830 ± 0.044   | 0.395 ± 0.112     |

Top features match the standard small-college distress story: declining expendable net assets, leverage, revenue trend, expenses-to-revenue, operating margin, tuition dependence.

## Honest limits

- AUC 0.83 is a useful watchlist, not a closure verdict. Most flagged schools still won't close.
- IPEDS Fall Enrollment data is not yet integrated. That is the next data investment.
- Only the 2024 Header snapshot is in the repo, so time-varying metadata isn't modeled.
- No model is persisted or served. Research codebase.

## Next

Add IPEDS Fall Enrollment 2014 to 2024, fit a calibrated XGBoost model with precision-at-K reporting, and reframe as time-to-event once more post-window closures accumulate.

## Run

```bash
.venv/bin/python scripts/01_combine_f2.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_init.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_attach_metadata_labels.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_baseline_model.ipynb
```
