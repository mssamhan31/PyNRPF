"""Round diagnosis: why are human-obvious (Beta 'sure') RPF days not corrected?

Purpose: take a run's held-out predictions, isolate every 'sure' RPF day whose
outcome is not AUTO_CORRECT, and classify the cause so a remedy can be named:
  input      - a required slot was missing, so the day abstained by rule;
  scorer     - the best window scored at or below zero (no positive evidence);
  calibration- positive evidence but p below c (score ordered right, threshold wrong);
  window     - corrected but the window carries little of the reference energy.
It then plots the highest-energy examples per station so the failure can be seen.

Inputs:  runs/<run>/predictions_beta.csv and calibration_fits.csv; the frozen datasets.
Outputs: weak_sites/<run>/missed_sure_breakdown.csv, missed_sure_days.csv and PNGs.

Usage: python weak_sites/diagnose_missed_sure.py --run phase2_baseline --per-station 4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEV = HERE.parent
sys.path.insert(0, str(DEV))

DATA = DEV.parent / "dataset" / "final"


def classify(row: pd.Series, c: float) -> str:
    if not row["input_ok"]:
        return "input"
    if row["outcome"] == "AUTO_CORRECT":
        iou = row["correct_mwh"] / (row["proposed_mwh"] + row["required_mwh"] - row["correct_mwh"])
        return "window" if iou < 0.5 else "ok"
    if row["r_best"] <= 0:
        return "scorer"
    return "calibration"


def plot_day(ax, y, s, truth, row):
    t = np.arange(96) / 4
    u0 = s + y
    uc = u0.copy()
    a, b = int(row["best_start"]), int(row["best_end"])
    if a >= 0:
        uc[a : b + 1] = s[a : b + 1] - y[a : b + 1]
    ax.plot(t, y, color="black", lw=1.2, label="recorded net load")
    ax.plot(t, s, color="orange", lw=1, label="solar")
    ax.plot(t, u0, color="grey", lw=1, ls="--", label="demand, keep sign")
    if a >= 0:
        ax.plot(t, uc, color="tab:blue", lw=1, label="demand, corrected")
        ax.axvspan(a / 4, (b + 1) / 4, color="tab:blue", alpha=0.12, label="proposed window")
    idx = np.flatnonzero(truth)
    if idx.size:
        ax.axvspan(idx[0] / 4, (idx[-1] + 1) / 4, color="tab:red", alpha=0.10, label="labelled RPF")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_title(f"{row['station']} {row['date']}  {row['outcome']}  r={row['r_best']:.0f} p={row['p']:.2f} "
                 f"sigma={row['sigma']:.3f}  req={row['required_mwh']:.1f} MWh  cause={row['cause']}", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-station", type=int, default=4)
    a = ap.parse_args()
    run_dir = DEV / "runs" / a.run
    out = HERE / a.run
    out.mkdir(parents=True, exist_ok=True)
    c = pd.read_json(run_dir / "config.json", typ="series")["c"]

    pred = pd.read_csv(run_dir / "predictions_beta.csv")
    sure = pred[(pred["confidence"] == "sure") & (pred["rpf"] == 1)].copy()
    sure["cause"] = sure.apply(classify, axis=1, c=c)
    missed = sure[sure["cause"] != "ok"].copy()

    bd = (missed.groupby(["station", "cause"]).agg(days=("date", "size"), mwh=("required_mwh", "sum"))
          .reset_index())
    tot = sure.groupby("station").agg(sure_days=("date", "size"), sure_mwh=("required_mwh", "sum")).reset_index()
    bd = bd.merge(tot, on="station")
    bd["share_of_station_days"] = bd["days"] / bd["sure_days"]
    bd.to_csv(out / "missed_sure_breakdown.csv", index=False)
    missed.sort_values("required_mwh", ascending=False).to_csv(out / "missed_sure_days.csv", index=False)

    print(f"=== {a.run}: Beta 'sure' RPF days not corrected, by cause ===")
    print(f"sure RPF days {len(sure)}, not corrected {len(missed)} ({len(missed)/len(sure):.1%}), "
          f"reference energy not corrected {missed.required_mwh.sum():.0f} of {sure.required_mwh.sum():.0f} MWh")
    print(missed.groupby("cause").agg(days=("date", "size"), mwh=("required_mwh", "sum")).round(1).to_string())
    print("\nby station and cause:")
    print(bd.round(3).to_string(index=False))

    # Where do calibration-class misses sit relative to the raw threshold that was applied?
    fits = pd.read_csv(run_dir / "calibration_fits.csv")
    fits = fits[fits.cohort == "beta"].set_index("held_out")
    cal = missed[missed.cause == "calibration"].copy()
    cal["raw_thr"] = cal.station.map(fits["raw_threshold_correct"])
    cal["ratio_to_thr"] = cal.r_best / cal.raw_thr
    print("\ncalibration-class misses: raw score as a fraction of the applied raw threshold")
    print(cal.groupby("station").ratio_to_thr.describe()[["count", "25%", "50%", "75%"]].round(2).to_string())
    print("\nraw thresholds applied per held-out station:")
    print(fits[["raw_threshold_correct", "alpha", "beta"]].round(3).to_string())

    # Figures: top examples per station by reference energy.
    df = pd.read_parquet(DATA / "dataset_beta.parquet")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["ts"].dt.date.astype(str)
    df = df.sort_values(["substation_id", "ts"])
    for sid, g in missed.groupby("station"):
        top = g.sort_values("required_mwh", ascending=False).head(a.per_station)
        fig, axes = plt.subplots(len(top), 1, figsize=(11, 2.6 * len(top)), squeeze=False)
        for ax, (_, row) in zip(axes[:, 0], top.iterrows(), strict=False):
            day = df[(df.substation_id == sid) & (df.d == row["date"])]
            plot_day(ax, day.net_load_MW.to_numpy(float), day.solar_MW.to_numpy(float),
                     day.label_interval.to_numpy(bool), row)
        axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / f"missed_sure_{sid}.png", dpi=110)
        plt.close(fig)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
