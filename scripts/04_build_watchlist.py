"""Train LightGBM on the multicohort matrix and produce two watchlists.

Outputs (both under processed_data/):
    watchlist_2019_backtest.csv  — OOF predictions for the 2019 cohort with actual outcomes
    watchlist_2024_forward.csv   — forward-looking list (predicting closure by 2029)

Each row: rank, INSTNM, state, risk_score, top-3 SHAP-derived risk drivers (plain English),
and a few human-readable context columns (operating margin, endowment-per-student, etc.).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed_data"

N_SPLITS = 5
SEED = 42
TOP_DRIVERS = 3

# Plain-English labels for the SHAP top-driver columns. Anything not in this map
# falls back to the raw feature name.
DRIVER_LABEL = {
    "operating_margin__last": "operating margin",
    "operating_margin__mean3": "3-yr avg operating margin",
    "expenses_to_revenue__last": "expenses-to-revenue ratio",
    "expenses_to_revenue__mean3": "3-yr avg expenses-to-revenue",
    "assets_to_liabilities__last": "assets-to-liabilities ratio",
    "assets_to_liabilities__mean3": "3-yr avg assets-to-liabilities",
    "log_endowment_eoy": "endowment size (log)",
    "endowment_per_student": "endowment per student",
    "log_total_enrollment": "enrollment size (log)",
    "total_enrollment__pct3": "3-yr enrollment % change",
    "total_enrollment__slope": "enrollment trend (slope)",
    "tuition_dependence": "tuition dependence (tuition / total revenue)",
    "F2D16__pct3": "3-yr revenue % change",
    "F2D16__slope": "revenue trend (slope)",
    "expendable_net_assets__pct3": "3-yr expendable net assets % change",
    "expendable_net_assets__slope": "expendable net assets trend",
    "log_expendable_net_assets": "expendable net assets (log)",
    "reported_field_count__mean3": "data completeness (avg fields reported)",
    "undergrad_share__last": "undergraduate share of enrollment",
    "revenue_per_student": "revenue per student",
    "tuition_per_student": "tuition per student",
    "total_expenses__slope": "expenses trend (slope)",
    "total_expenses__pct3": "3-yr expenses % change",
    "log_total_expenses": "total expenses (log)",
    "log_F2D16": "total revenue (log)",
    "log_F2A02": "total assets (log)",
    "log_F2A03": "total liabilities (log)",
    "F2D01__last": "tuition & fees",
    "F2D16__last": "total revenue",
    "F2A02__last": "total assets",
    "F2A03__last": "total liabilities",
    "total_expenses__last": "total expenses",
    "change_in_net_assets__last": "change in net assets",
    "expendable_net_assets__last": "expendable net assets",
    "endowment_eoy__last": "endowment EOY",
    "total_enrollment__last": "total enrollment",
    "undergrad_enrollment__last": "undergrad enrollment",
}


def lgbm_params() -> dict:
    return dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def load_metadata() -> pd.DataFrame:
    """Latest INSTNM and STABBR per UNITID, taken from the labeled F2 panel."""
    df = pd.read_csv(
        PROC / "f2_labeled.csv",
        low_memory=False,
        usecols=["UNITID", "year", "INSTNM", "STABBR"],
    )
    return (
        df.sort_values(["UNITID", "year"])
        .groupby("UNITID")
        .last()[["INSTNM", "STABBR"]]
    )


def train_and_oof(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> tuple[np.ndarray, lgb.LGBMClassifier]:
    """Returns (uncalibrated OOF probs, model fit on all rows)."""
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMClassifier(**lgbm_params())
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"  fold {fold}: train={len(tr):>5}  val={len(va):>5}")
    final = lgb.LGBMClassifier(**lgbm_params())
    final.fit(X, y)
    return oof, final


def shap_for(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return per-row SHAP values for the positive class. Handles old/new shap output shapes."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # old API: [neg, pos]
        sv = sv[1]
    if sv.ndim == 3:  # newer API: (n, k, n_classes)
        sv = sv[:, :, 1]
    return sv


def top_driver_labels(shap_row: np.ndarray, feature_names: list[str], k: int) -> list[str]:
    """k features with the largest positive SHAP value (push prediction toward closure)."""
    top = np.argsort(-shap_row)[:k]
    return [DRIVER_LABEL.get(feature_names[i], feature_names[i]) for i in top]


def build_watchlist(
    scores: pd.Series,
    feature_df: pd.DataFrame,
    shap_matrix: np.ndarray,
    metadata: pd.DataFrame,
    actual: pd.Series | None = None,
) -> pd.DataFrame:
    feature_names = list(feature_df.columns)
    unitids = feature_df.index.get_level_values("UNITID")

    rows = []
    for i, unitid in enumerate(unitids):
        meta = metadata.loc[unitid] if unitid in metadata.index else None
        f = feature_df.iloc[i]
        drivers = top_driver_labels(shap_matrix[i], feature_names, TOP_DRIVERS)

        row = {
            "UNITID": int(unitid),
            "INSTNM": meta["INSTNM"] if meta is not None else None,
            "state": meta["STABBR"] if meta is not None else None,
            "risk_score": float(scores.iloc[i]),
            "driver_1": drivers[0] if len(drivers) > 0 else None,
            "driver_2": drivers[1] if len(drivers) > 1 else None,
            "driver_3": drivers[2] if len(drivers) > 2 else None,
            "operating_margin": float(f.get("operating_margin__last", np.nan)),
            "endowment_per_student": float(f.get("endowment_per_student", np.nan)),
            "total_enrollment": float(f.get("total_enrollment__last", np.nan)),
            "enrollment_3yr_pct": float(f.get("total_enrollment__pct3", np.nan)),
        }
        if actual is not None:
            row["actual_closed_by_2024"] = bool(actual.iloc[i])
        rows.append(row)

    out = (
        pd.DataFrame(rows)
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "rank", out.index + 1)
    return out


def main() -> None:
    print("Loading data...")
    labeled = pd.read_csv(PROC / "f2_effy_features_multicohort.csv").set_index(
        ["UNITID", "as_of_year"]
    )
    inference = pd.read_csv(PROC / "f2_effy_features_asof2024_inference.csv").set_index(
        ["UNITID", "as_of_year"]
    )
    metadata = load_metadata()

    label_col = "closed_within_horizon"
    feature_cols = [c for c in labeled.columns if c != label_col]

    X = labeled[feature_cols]
    y = labeled[label_col].astype(int)
    groups = labeled.index.get_level_values("UNITID").to_numpy()

    print(f"\nTraining LightGBM ({len(X)} rows × {len(feature_cols)} features, "
          f"positive rate {y.mean():.2%})...")
    oof, final_model = train_and_oof(X, y, groups)

    print(f"\nOOF AUC = {roc_auc_score(y, oof):.4f}")
    print(f"OOF AP  = {average_precision_score(y, oof):.4f}")

    print("\nFitting isotonic calibrator on OOF predictions...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof, y)
    oof_cal = calibrator.predict(oof)

    # ---- 2019 backtest watchlist (held-out OOF predictions) ----
    print("\n[2019 backtest watchlist]")
    is_2019 = labeled.index.get_level_values("as_of_year") == 2019
    X_2019 = X[is_2019]
    y_2019 = y[is_2019]
    scores_2019 = pd.Series(oof_cal[is_2019], index=X_2019.index)
    # SHAP from the final model — same view of "what drives risk" for both lists.
    shap_2019 = shap_for(final_model, X_2019)
    watchlist_2019 = build_watchlist(scores_2019, X_2019, shap_2019, metadata, actual=y_2019.astype(bool))
    out_2019 = PROC / "watchlist_2019_backtest.csv"
    watchlist_2019.to_csv(out_2019, index=False)
    print(f"wrote {out_2019}  shape={watchlist_2019.shape}")
    for k in [25, 50, 100, 200]:
        catches = int(watchlist_2019.head(k)["actual_closed_by_2024"].sum())
        print(f"  top-{k}: {catches} real closures caught ({catches/k:.0%})")

    # ---- 2024 forward watchlist (predict from final model) ----
    print("\n[2024 forward watchlist]")
    X_2024 = inference[feature_cols]
    raw_2024 = final_model.predict_proba(X_2024)[:, 1]
    scores_2024 = pd.Series(calibrator.predict(raw_2024), index=X_2024.index)
    shap_2024 = shap_for(final_model, X_2024)
    watchlist_2024 = build_watchlist(scores_2024, X_2024, shap_2024, metadata, actual=None)
    out_2024 = PROC / "watchlist_2024_forward.csv"
    watchlist_2024.to_csv(out_2024, index=False)
    print(f"wrote {out_2024}  shape={watchlist_2024.shape}")
    print("\nTop 10 forward-looking risks:")
    print(
        watchlist_2024[["rank", "INSTNM", "state", "risk_score", "driver_1"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
