"""Combine 11 years of IPEDS F2 (private nonprofit finance) into one long panel.

Cross-year column rules are documented in raw_data/README.md. We keep both the
late-year primary fields (F2I03 / F2I05 / F2I07) and their early-year proxies
(F2B02, F2B04, F2A04 + F2A05B); downstream feature code coalesces them into the
canonical derived columns total_expenses, change_in_net_assets, expendable_net_assets.
"""
from pathlib import Path

import pandas as pd

YEAR_FILES = {
    2014: "f1314_f2_rv.csv",
    2015: "f1415_f2_rv.csv",
    2016: "f1516_f2_rv.csv",
    2017: "f1617_f2_rv.csv",
    2018: "f1718_f2_rv.csv",
    2019: "f1819_f2_rv.csv",
    2020: "f1920_f2_rv.csv",
    2021: "f2021_f2_rv.csv",
    2022: "f2122_f2_rv.csv",
    2023: "f2223_f2_rv.csv",
    2024: "f2324_f2_rv.csv",
}

KEEP = [
    "UNITID",
    "F2D01", "F2D16",                   # tuition & fees, total revenues
    "F2A02", "F2A03",                   # total assets, total liabilities
    "F2B02", "F2B04",                   # total expenses proxy, change in net assets proxy
    "F2A04", "F2A05B",                  # F2I05 proxy = unrestricted + temp restricted
    "F2I03", "F2I05", "F2I07",          # later-year primary fields
    "F2H01", "F2H02", "F2FHA",          # endowment BOY/EOY + endowment presence flag
]

# IPEDS pairs each value F2XXX with an imputation flag XF2XXX. F2FHA has no paired flag.
KEEP_FLAGS = [f"X{c}" for c in KEEP[1:] if c != "F2FHA"]
KEEP_ALL = KEEP + KEEP_FLAGS


def load_year(raw_dir: Path, year: int) -> pd.DataFrame:
    path = raw_dir / str(year) / YEAR_FILES[year]
    df = pd.read_csv(path, low_memory=False)
    # 2014 has trailing whitespace in some header names (e.g. 'F2H02   ').
    df.columns = df.columns.str.strip()
    present = [c for c in KEEP_ALL if c in df.columns]
    missing = [c for c in KEEP_ALL if c not in df.columns]
    df = df[present].copy()
    for c in missing:
        df[c] = pd.NA
    df["year"] = year
    return df[["UNITID", "year"] + KEEP[1:] + KEEP_FLAGS]


def combine(raw_dir: Path) -> pd.DataFrame:
    """Stitch all years into one long DataFrame. Returns; does not write."""
    frames = []
    for year in sorted(YEAR_FILES):
        df = load_year(raw_dir, year)
        have = [c for c in KEEP[1:] if df[c].notna().any()]
        print(f"{year}: {len(df):>5} rows  |  cols with data: {len(have)}/{len(KEEP)-1}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(raw_dir: Path | None = None, out_dir: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = raw_dir or root / "raw_data"
    out_dir = out_dir or root / "processed_data"

    combined = combine(raw_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "f2_combined_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  shape={combined.shape}")
    print(f"unique UNITIDs: {combined['UNITID'].nunique()}")
    print(f"year counts:\n{combined['year'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
