# Closure

Early-warning signal for U.S. private not-for-profit college closures, built on public IPEDS data.

## Forward-looking watchlist (closure by 2029)

The 25 schools the model rates highest-risk as of the most recent IPEDS data (filed 2024). Full ranked list of all 1,732 institutions with risk drivers and human-readable context: [processed_data/watchlist_2024_forward.csv](processed_data/watchlist_2024_forward.csv).

| Rank | School | State | Risk | Top driver |
|-----:|--------|:-----:|-----:|------------|
| 1 | Visible Music College | TN | 95% | assets-to-liabilities ratio |
| 2 | Jackson Theological Seminary | AR | 94% | undergrad enrollment |
| 3 | California College of ASU | CA | 80% | 3-yr enrollment % change |
| 4 | Watts College of Nursing | NC | 80% | tuition dependence |
| 5 | Vermont College of Fine Arts | VT | 80% | 3-yr enrollment % change |
| 6 | Redeemers University North America | TX | 80% | 3-yr enrollment % change |
| 7 | Remington College-Online Dallas | TX | 80% | 3-yr enrollment % change |
| 8 | Remington College-Dallas Campus | TX | 78% | 3-yr enrollment % change |
| 9 | Oklahoma Technical College | OK | 78% | tuition dependence |
| 10 | Remington College-Cleveland Campus | OH | 78% | expendable net assets |
| 11 | San Diego Christian College | CA | 78% | 3-yr enrollment % change |
| 12 | Lakewood University | OH | 78% | expendable net assets |
| 13 | Memorial Hospital School of Radiation Therapy Tech. | NY | 78% | 3-yr enrollment % change |
| 14 | St. Joseph's College of Nursing | NY | 66% | expendable net assets |
| 15 | Lawrence Memorial Hospital School of Nursing | MA | 66% | tuition dependence |
| 16 | Sh'or Yoshuv Rabbinical College | NY | 66% | 3-yr enrollment % change |
| 17 | Bryant & Stratton College-Albany | NY | 66% | 3-yr enrollment % change |
| 18 | Dewey University-Hato Rey | PR | 66% | expendable net assets |
| 19 | Polytechnic University of Puerto Rico-Miami | FL | 66% | 3-yr enrollment % change |
| 20 | Montserrat College of Art | MA | 66% | expendable net assets |
| 21 | American Islamic College | IL | 66% | undergrad enrollment |
| 22 | Interdenominational Theological Center | GA | 66% | 3-yr enrollment % change |
| 23 | Portland Actors Conservatory | OR | 66% | undergrad enrollment |
| 24 | Remington College-Fort Worth Campus | TX | 66% | assets-to-liabilities ratio |
| 25 | Springfield College-Regional, Online, and Cont. Ed. | MA | 66% | 3-yr enrollment % change |

This is a ranked list, not a verdict — the majority of schools at any usable threshold will not close. Risk is the calibrated probability that the school will close by 2029.

## Backtest credibility check (2019 cohort, observed by 2024)

Of the 25 highest-risk schools the model flagged in the 2019 watchlist, **21 actually closed within 5 years** (84% hit rate vs. a 7% base rate). A board or accreditor reviewing the top 50 schools would have caught ~25% of all closures while reviewing under 3% of institutions.

| Top-N reviewed | Real closures caught | Hit rate | % of all closures |
|----------------|----------------------|----------|-------------------|
| 25  | 21 | 84% | 16% |
| 50  | 32 | 64% | 25% |
| 100 | 41 | 41% | 32% |
| 200 | 61 | 30% | 47% |

Full ranked 2019 watchlist with actual outcomes: [processed_data/watchlist_2019_backtest.csv](processed_data/watchlist_2019_backtest.csv).

## Data

Eleven years of the IPEDS Finance Survey Part F2 (2014 to 2024), the 12-month Enrollment file (EFFY, 2014 to 2024), and the IPEDS Header file. ~21,000 institution-year rows across ~2,100 private nonprofit schools, narrowing to a multi-cohort training matrix of 7,342 snapshots across 1,960 institutions after labeling.

The pipeline handles cross-year schema drift, IPEDS imputation flags, and silent closures (institutions that drop off the directory without a `CLOSEDAT` record).

Cohort scope is "private nonprofit" by IPEDS classification, implemented as an exclusion: institutions ever classified as public (`CONTROL=1` / `SECTOR ∈ {1,4,7}`) or for-profit (`CONTROL=3` / `SECTOR ∈ {3,6,9}`) are dropped. Several chains commonly thought of as for-profit — Remington College, Bryant & Stratton, Herzing — have legally converted to nonprofit and IPEDS classifies them accordingly, so they remain in the cohort.

## Model

LightGBM trained with 5-fold `GroupKFold` (by `UNITID`) on the multi-cohort matrix. Predictions are isotonic-calibrated on the held-out OOF predictions. Per-school top-3 risk drivers come from SHAP values on the final model.

| Metric | Value |
|--------|------:|
| OOF AUC | 0.827 |
| OOF Average Precision | 0.380 |
| Positive rate | 8.7% |

An MLP (Keras, 64-32-1) was compared head-to-head in the original course iteration; trees beat the MLP on the same splits, which is the expected result on tabular data of this size. The MLP path has been retired and the codebase has standardized on LightGBM.

## Methodology that matters

- **Prospective, leak-safe cohort design.** Features come strictly from before the as-of date. The same institution never appears in both training and validation folds.
- **Multi-cohort stacking.** Four as-of years (2016 to 2019) with a 5-year forward closure label, expanding the dataset 4x over the single-cohort baseline.
- **Closure year derived from two sources.** Parsed `CLOSEDAT` for institutions still in the 2024 directory (~23 closures); last F2 filing year + 1 for institutions that silently dropped off (~352 closures).
- **Cohort filter is exclusion-based, not include-based.** Closed institutions disproportionately lose their HD record (we only have HD files for 2014 and 2024), so a strict "include only `SECTOR ∈ {2,5,8}`" filter would silently drop ~95% of positive examples — selection bias on the outcome.

## Honest limits

- The model produces a watchlist, not a verdict. At any usable threshold, the majority of flagged schools will not close.
- Closure year for ~94% of closed institutions is approximated from "last F2 filing" rather than a structured close date, which adds noise to the label.
- The 5-year horizon means the 2019 cohort is the most recent one we can evaluate against fully observed outcomes; the 2024 forward list won't have ground truth until ~2029.
- "Private nonprofit" follows the IPEDS `CONTROL`/`SECTOR` classification, which includes institutions that have legally converted from for-profit status.
- No model artifact is persisted or served. Research codebase.

## What's next

Add annual IPEDS Header files (currently only 2014 and 2024 are available, which forces most closure-year labels to come from "last F2 filing" rather than a structured date); pull in IPEDS Admissions (`ADM`) for leading-indicator signals; consider the ED Financial Responsibility Composite Scores as both a feature and an external baseline to beat.

## Pipeline

Run in order from project root:

```bash
.venv/bin/python scripts/01_combine_f2.py
.venv/bin/python scripts/02_combine_effy.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_init.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_attach_metadata_labels.ipynb
.venv/bin/python scripts/03_build_multicohort_features.py
.venv/bin/python scripts/04_build_watchlist.py
```

Outputs land under `processed_data/`: `watchlist_2024_forward.csv` (the live list) and `watchlist_2019_backtest.csv` (the credibility check against observed outcomes).
