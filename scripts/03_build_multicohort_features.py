from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed_data"

AS_OF_YEARS = [2016, 2017, 2018, 2019]
HORIZON_YEARS = 5
WINDOW_LEN = 3  # look-back including the as-of year itself

NUM = [
    "F2D01", "F2D16", "F2A02", "F2A03",
    "total_expenses", "change_in_net_assets", "expendable_net_assets",
    "endowment_eoy",
    "operating_margin", "expenses_to_revenue", "assets_to_liabilities",
]
MEAN_FIELDS = ["operating_margin", "expenses_to_revenue", "assets_to_liabilities", "reported_field_count"]
TREND_FIELDS = ["F2D16", "total_expenses", "expendable_net_assets"]
LOG_FIELDS = ["F2D16", "F2A02", "F2A03", "total_expenses", "expendable_net_assets", "endowment_eoy"]


def slope(g: pd.DataFrame, col: str) -> float:
    s = g[["year", col]].dropna()
    if len(s) < 2:
        return np.nan
    y = s[col].to_numpy()
    if np.std(y) == 0:
        return 0.0
    return float(np.polyfit(s["year"].to_numpy(), y, 1)[0])


def pct_window(g: pd.DataFrame, col: str) -> float:
    s = g[["year", col]].dropna().sort_values("year")
    if len(s) < 2 or s[col].iloc[0] == 0:
        return np.nan
    return (s[col].iloc[-1] - s[col].iloc[0]) / abs(s[col].iloc[0])


def compute_closure_years(labeled: pd.DataFrame) -> pd.Series:
    per_inst = labeled.groupby("UNITID").agg(
        closed=("closed_by_2024", "max"),
        closed_date=("closed_date", "first"),
        last_f2_year=("year", "max"),
    )

    closure_year = pd.Series(np.nan, index=per_inst.index, dtype=float)

    parsed = pd.to_datetime(per_inst["closed_date"], errors="coerce")
    has_date = parsed.notna()
    closure_year.loc[has_date] = parsed[has_date].dt.year.astype(float)

    inferred = per_inst["closed"].fillna(False) & ~has_date
    closure_year.loc[inferred] = per_inst.loc[inferred, "last_f2_year"].astype(float) + 1.0

    return closure_year


def build_f2_features(panel: pd.DataFrame, window: list[int], cohort: set[int]) -> pd.DataFrame:
    sub = panel[panel.UNITID.isin(cohort) & panel.year.isin(window)].copy()
    for c in NUM:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    last = (
        sub.sort_values(["UNITID", "year"])
        .groupby("UNITID")[NUM].last().add_suffix("__last")
    )
    mean3 = sub.groupby("UNITID")[MEAN_FIELDS].mean().add_suffix("__mean3")
    slopes = sub.groupby("UNITID").apply(
        lambda g: pd.Series({f"{c}__slope": slope(g, c) for c in TREND_FIELDS}),
        include_groups=False,
    )
    pct = sub.groupby("UNITID").apply(
        lambda g: pd.Series({f"{c}__pct3": pct_window(g, c) for c in TREND_FIELDS}),
        include_groups=False,
    )

    feat = last.join(mean3).join(slopes).join(pct)
    for c in LOG_FIELDS:
        s = feat[f"{c}__last"]
        feat[f"log_{c}"] = np.sign(s) * np.log1p(np.abs(s))
    feat["tuition_dependence"] = feat["F2D01__last"] / feat["F2D16__last"].replace({0: np.nan})
    return feat.replace([np.inf, -np.inf], np.nan)


def build_effy_features(effy: pd.DataFrame, window: list[int], cohort: set[int]) -> pd.DataFrame:
    sub = effy[effy.UNITID.isin(cohort) & effy.year.isin(window)].copy()
    for c in ["total_enrollment", "undergrad_enrollment"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    last = (
        sub.sort_values(["UNITID", "year"])
        .groupby("UNITID")[["total_enrollment", "undergrad_enrollment"]]
        .last().add_suffix("__last")
    )
    slopes = sub.groupby("UNITID").apply(
        lambda g: pd.Series({"total_enrollment__slope": slope(g, "total_enrollment")}),
        include_groups=False,
    )
    pct = sub.groupby("UNITID").apply(
        lambda g: pd.Series({"total_enrollment__pct3": pct_window(g, "total_enrollment")}),
        include_groups=False,
    )

    feat = last.join(slopes).join(pct)
    feat["undergrad_share__last"] = (
        feat["undergrad_enrollment__last"]
        / feat["total_enrollment__last"].replace({0: np.nan})
    )
    return feat.replace([np.inf, -np.inf], np.nan)


def build_one_cohort(
    labeled: pd.DataFrame,
    effy: pd.DataFrame,
    closure_year: pd.Series,
    as_of_year: int,
) -> pd.DataFrame:
    window = list(range(as_of_year - WINDOW_LEN + 1, as_of_year + 1))

    alive = set(labeled.loc[labeled.year == as_of_year, "UNITID"].unique())
    not_yet_closed = set(
        closure_year[closure_year.isna() | (closure_year > as_of_year)].index
    )
    cohort = alive & not_yet_closed

    f2_feat = build_f2_features(labeled, window, cohort)
    ef_feat = build_effy_features(effy, window, cohort)

    combined = f2_feat.join(ef_feat, how="left")
    denom = combined["total_enrollment__last"].replace({0: np.nan})
    combined["endowment_per_student"] = combined["endowment_eoy__last"] / denom
    combined["revenue_per_student"] = combined["F2D16__last"] / denom
    combined["tuition_per_student"] = combined["F2D01__last"] / denom
    s = combined["total_enrollment__last"]
    combined["log_total_enrollment"] = np.sign(s) * np.log1p(np.abs(s))

    cy = closure_year.reindex(combined.index)
    label = ((cy.notna()) & (cy <= as_of_year + HORIZON_YEARS)).astype(int)
    combined["closed_within_horizon"] = label
    combined["as_of_year"] = as_of_year
    return combined


def main() -> None:
    print(f"loading {PROC/'f2_labeled.csv'} and {PROC/'effy_combined.csv'}")
    labeled = pd.read_csv(PROC / "f2_labeled.csv", low_memory=False)
    effy = pd.read_csv(PROC / "effy_combined.csv")

    closure_year = compute_closure_years(labeled)
    n_closed = int(closure_year.notna().sum())
    print(f"institutions with derivable closure year: {n_closed} / {len(closure_year)}")

    frames = []
    for as_of in AS_OF_YEARS:
        cohort_df = build_one_cohort(labeled, effy, closure_year, as_of)
        pos = int(cohort_df["closed_within_horizon"].sum())
        print(
            f"as_of={as_of}: {len(cohort_df):>5} institutions  "
            f"positives={pos} ({pos/len(cohort_df):.2%})  "
            f"window=[{as_of-WINDOW_LEN+1}-{as_of}]  label=closed by {as_of+HORIZON_YEARS}"
        )
        cohort_df.index.name = "UNITID"
        cohort_df = cohort_df.reset_index().set_index(["UNITID", "as_of_year"])
        frames.append(cohort_df)

    combined = pd.concat(frames)
    print(f"\ncombined matrix: {combined.shape}")
    print(
        f"total positives: {int(combined['closed_within_horizon'].sum())} / "
        f"{len(combined)} ({combined['closed_within_horizon'].mean():.2%})"
    )
    print(f"unique UNITIDs across cohorts: {combined.index.get_level_values('UNITID').nunique()}")

    out_path = PROC / "f2_effy_features_multicohort.csv"
    combined.to_csv(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
