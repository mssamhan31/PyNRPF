"""Build the reference slices the package tests are pinned to.

Reads the journal datasets and the reference run under publication/2_journal_article and
writes two parquet files here:

    beta_slice.parquet    every reading of station beta_D (gap-adjacent days, days where no
                          correction wins, days with missing readings) with the reference
                          evidence, windows, probability and outcome of the reference run
    alpha_slice.parquet   thirty days of station alpha_C, likewise

Both carry the calibration pair and floor the reference used for that station, so the tests
compare like with like. Run once from the repository root; the outputs are committed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "publication" / "2_journal_article"
RESULTS = ARTICLE / "results"
HERE = Path(__file__).resolve().parent


def build(cohort: str, station: str, fold_id: str, phi: float, dates: list[str] | None = None) -> pd.DataFrame:
    readings = pd.read_parquet(ARTICLE / "dataset" / "final" / f"dataset_{cohort}.parquet")
    readings = readings[readings["substation_id"] == station].copy()
    readings["date"] = pd.to_datetime(readings["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    if dates is not None:
        readings = readings[readings["date"].isin(dates)]
    scores = pd.read_parquet(RESULTS / "03_m9" / f"scores_{cohort}.parquet")
    keep = ["date", "input_ok", "n_admissible", "best_start", "best_end", "runner_start", "runner_end", "r_best"]
    scores = scores[scores["station"] == station][keep]
    days = pd.read_parquet(RESULTS / "04_metrics" / "site_days.parquet")
    days = days[(days["method"] == "m9") & (days["station"] == station)]
    days = days[["date", "prob_day", "outcome", "candidate_mwh"]]
    fits = pd.read_csv(RESULTS / "03_m9" / "calibration_fits.csv", float_precision="round_trip").set_index("fold_id")
    readings = readings[["substation_id", "timestamp", "date", "net_load_MW", "solar_MW"]]
    out = readings.merge(scores, on="date").merge(days, on="date")
    out["cal_intercept"] = float(fits.loc[fold_id, "cal_intercept"])
    out["cal_slope"] = float(fits.loc[fold_id, "cal_slope"])
    out["phi"] = phi
    return out.reset_index(drop=True)


def main() -> None:
    beta = build("beta", "beta_D", "beta_beta_D", 8.03130184579004e-09)
    beta.to_parquet(HERE / "beta_slice.parquet", index=False)
    alpha_scores = pd.read_parquet(RESULTS / "03_m9" / "scores_alpha.parquet")
    alpha_dates = sorted(alpha_scores.loc[alpha_scores["station"] == "alpha_C", "date"])[200:230]
    alpha = build("alpha", "alpha_C", "alpha_alpha_C", 1.400908788973254e-07, alpha_dates)
    alpha.to_parquet(HERE / "alpha_slice.parquet", index=False)
    print("beta days", beta["date"].nunique(), "alpha days", alpha["date"].nunique())


if __name__ == "__main__":
    main()
