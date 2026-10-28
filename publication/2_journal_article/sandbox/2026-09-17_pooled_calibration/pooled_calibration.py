"""Sandbox: does pooling Alpha and Beta for the M9 calibration change the held-out result?

Reads only committed Phase 3 outputs (label-free scores per site-day, candidate energies,
labels). Refits the two calibration coefficients per fold on all 17 other stations of both
cohorts (variant 'pooled') and on the other stations of the same cohort (variant 'within',
the frozen protocol, as a check that this script reproduces the committed numbers), then
pools held-out decisions at c = 0.7. Nothing frozen is modified.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ARTICLE = pathlib.Path(__file__).resolve().parents[2]
OUT = ARTICLE / "sandbox" / "2026-09-16_phase3_release" / "outputs"
sys.path.insert(0, str(ARTICLE / "m9_dev"))
import m9_metrics as mx  # noqa: E402
import m9_scorer as ms  # noqa: E402

C = 0.7
HERE = pathlib.Path(__file__).resolve().parent

scores = pd.concat([pd.read_parquet(OUT / "05_m9" / f"scores_{c}.parquet") for c in ("alpha", "beta")], ignore_index=True)
sd = pd.read_parquet(OUT / "06_site_days" / "site_days.parquet")
sd = sd[sd["method"] == "m9"][["cohort", "station", "date", "candidate_mwh", "candidate_correct_mwh", "required_mwh"]]
t = scores.merge(sd, on=["cohort", "station", "date"], how="left", validate="one_to_one")
eligible = t["input_ok"] & t["headline"]


def run(variant: str, weighted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, fits = [], []
    for sid in sorted(t["station"].unique()):
        cohort = t.loc[t["station"] == sid, "cohort"].iloc[0]
        train = t[(t["station"] != sid) & eligible]
        if variant == "within":
            train = train[train["cohort"] == cohort]
        z = ms.signed_log(train["r_best"].to_numpy(float)).reshape(-1, 1)
        y = train["rpf"].to_numpy(int)
        from sklearn.linear_model import LogisticRegression
        w = None
        if weighted:  # give each cohort equal total weight so Alpha's 5x more days do not dominate
            w = np.where(train["cohort"] == "alpha", 1.0 / (train["cohort"] == "alpha").sum(), 1.0 / (train["cohort"] == "beta").sum())
        m = LogisticRegression(max_iter=2000).fit(z, y, sample_weight=w)
        cal = ms.Calibrator(float(m.intercept_[0]), float(m.coef_[0, 0]))
        test = t[t["station"] == sid].copy()
        p = np.where(test["input_ok"], cal.probability(test["r_best"].fillna(0.0).to_numpy(float)), np.nan)
        test["p"] = p
        test["outcome"] = [ms.decide(v, C) if np.isfinite(v) else ms.UNCERTAIN for v in p]
        no_win = test["best_start"] < 0
        test.loc[no_win, "outcome"] = test.loc[no_win, "outcome"].replace(ms.AUTO_CORRECT, ms.UNCERTAIN)
        applied = test["outcome"] == ms.AUTO_CORRECT
        test["proposed_mwh"] = np.where(applied, test["candidate_mwh"], 0.0)
        test["correct_mwh"] = np.where(applied, test["candidate_correct_mwh"], 0.0)
        preds.append(test)
        fits.append(dict(held_out=sid, cohort=cohort, n_train=len(train), cal_intercept=cal.alpha, cal_slope=cal.beta, r_at_c=cal.raw_threshold(C)))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(fits)


rows, station_rows, fit_rows = [], [], []
for name, kw in (("within (frozen)", dict(variant="within")), ("pooled 17", dict(variant="pooled")), ("pooled 17, cohort-balanced", dict(variant="pooled", weighted=True))):
    pred, fits = run(**kw)
    fits.insert(0, "variant", name)
    fit_rows.append(fits)
    for cohort in ("alpha", "beta"):
        head = pred[(pred["cohort"] == cohort) & pred["headline"]]
        s = mx.summarise(head)
        rows.append(dict(variant=name, cohort=cohort, **{k: s[k] for k in ("n_days", "n_rpf", "energy_iou", "energy_precision", "day_f1", "day_precision", "sure_day_recall", "rate_uncertain", "fp", "fn")}))
        for sid, g in head.groupby("station"):
            ss = mx.summarise(g)
            station_rows.append(dict(variant=name, station=sid, energy_iou=ss["energy_iou"], energy_precision=ss["energy_precision"], sure_day_recall=ss["sure_day_recall"], fp=ss["fp"]))
pooled = pd.DataFrame(rows)
stations = pd.DataFrame(station_rows)
fits = pd.concat(fit_rows, ignore_index=True)
pooled.to_csv(HERE / "pooled_vs_within.csv", index=False)
stations.to_csv(HERE / "per_station.csv", index=False)
fits.to_csv(HERE / "fits.csv", index=False)
pd.set_option("display.width", 220)
print(pooled.round(4).to_string(index=False))
print()
print(fits[fits["variant"] != "within (frozen)"].round(3).to_string(index=False))
print()
w = stations.pivot(index="station", columns="variant", values="energy_iou").round(3)
print(w.to_string())
