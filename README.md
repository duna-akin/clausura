# clausura

Early-warning signal for U.S. private not-for-profit college closures, built on public IPEDS data.

## Headline result

Of the 25 highest-risk schools the model flagged in the 2019 watchlist, **15 actually closed within 5 years** (60% hit rate vs. a 7% base rate). The model produces a ranked list, not a verdict, so a board or accreditor reviewing the top 50 schools would catch ~21% of all future closures while reviewing under 3% of institutions.

| Top-N reviewed | Real closures caught | Hit rate | % of all closures |
|----------------|----------------------|----------|-------------------|
| 25  | 15 | 60% | 11% |
| 50  | 27 | 54% | 21% |
| 100 | 44 | 44% | 34% |
| 200 | 62 | 31% | 47% |

Full ranked watchlist with school names, key risk drivers, and actual outcomes: [processed_data/watchlist_2019_cohort.csv](processed_data/watchlist_2019_cohort.csv).

## Data

Eleven years of the IPEDS Finance Survey Part F2 (2014 to 2024), the 12-month Enrollment file (EFFY, 2014 to 2024), and the IPEDS Header file. About 20,000 institution-year rows across ~2,200 private nonprofit schools. The pipeline handles cross-year schema drift, IPEDS imputation flags, and silent closures (institutions that drop off the directory without a `CLOSEDAT` record).

## Models

Two models compared head-to-head on the same data and same cross-validation splits:

| Model                       | AUC               | Average Precision |
|-----------------------------|-------------------|-------------------|
| MLP (Keras, 64-32-1)        | 0.832 ± 0.015     | 0.380 ± 0.048     |
| Gradient-boosted trees      | **0.851** ± 0.011 | **0.446** ± 0.049 |

Trees beat the MLP, which is the expected result on tabular data of this size. The neural network is the course deliverable; the GBT baseline is the comparison point. Both run on a multi-cohort feature matrix of 7,457 institution-year snapshots across 1,991 unique institutions.

## Methodology that matters

- **Prospective, leak-safe cohort design.** Features come strictly from before the as-of date. The same institution never appears in both training and validation folds (GroupKFold by UNITID).
- **Multi-cohort stacking.** Four as-of years (2016 to 2019) with a 5-year forward closure label, expanding the dataset 4x over the single-cohort baseline.
- **Three imbalance strategies compared.** Vanilla BCE, class weighting, and SMOTE produce essentially identical AUC, which is the right finding to report rather than the wrong one to hide.
- **Closure year is derived from two sources.** Parsed `CLOSEDAT` for institutions still in the 2024 directory (~23 closures); last F2 filing year + 1 for institutions that silently dropped off (~353 closures).

## Honest limits

- The model produces a useful watchlist, not a verdict. At any usable threshold, the majority of flagged schools will not close.
- Closure year for ~94% of closed institutions is approximated from "last F2 filing" rather than a structured date, which adds noise to the label.
- The 5-year horizon means the most recent cohort (2019, label = closed by 2024) is the only one we can evaluate against fully observed outcomes.
- No model is persisted or served. Research codebase.

## What's next

Add IPEDS retention rate and student-faculty ratio data (EF{YYYY}D), calibrate the model's predicted probabilities, and reframe as time-to-event once more post-window closures accumulate. A tabular-specialized neural network (TabNet or FT-Transformer) is the natural follow-up to the MLP.

## Pipeline

Run in order from project root:

```bash
.venv/bin/python scripts/01_combine_f2.py
.venv/bin/python scripts/02_combine_effy.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_init.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_attach_metadata_labels.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_baseline_model.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_model_with_enrollment.ipynb
.venv/bin/python scripts/03_build_multicohort_features.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/05_mlp_keras.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/06_watchlist_report.ipynb
```
