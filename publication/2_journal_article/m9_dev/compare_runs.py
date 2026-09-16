"""Apply the Phase 0 adoption rules mechanically to a candidate run against the incumbent.

Rule 2 (consistency): no station with at least one 'sure' RPF day may lose sure-day
recall or energy precision; the pooled metric or a named failure class must improve.
Rule 5 (Alpha non-regression): Alpha pooled Energy IoU and energy precision within
0.01 of the incumbent. Rule 1 (gate): energy precision >= 0.90 on both cohorts.

Usage: python compare_runs.py --incumbent phase2_baseline --candidate r1_fullday
Prints the verdict and the per-station deltas; writes notes/compare_<cand>.csv.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
GATE = 0.90
ALPHA_TOL = 0.01
PREC_TOL = 0.01   # sampling-noise tolerance on station energy precision (round-1 note)
EPS = 1e-9


def load(run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(RUNS / run / "summary_pooled.csv")
    s = pd.read_csv(RUNS / run / "summary_station.csv")
    return p[p.group == "headline"].set_index("cohort"), s.set_index(["cohort", "station"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args()
    ip, ist = load(a.incumbent)
    cp, cst = load(a.candidate)

    print(f"=== {a.candidate} vs {a.incumbent} ===")
    cols = ["energy_iou", "energy_precision", "sure_day_recall", "sure_day_uncertain_rate", "day_precision", "day_f1"]
    pooled = pd.DataFrame({"incumbent": ip[cols].stack(), "candidate": cp[cols].stack()})
    pooled["delta"] = pooled.candidate - pooled.incumbent
    print("\npooled:")
    print(pooled.round(4).to_string())

    st = ist[cols].join(cst[cols], lsuffix="_inc", rsuffix="_cand")
    for c in cols:
        st[f"d_{c}"] = st[f"{c}_cand"] - st[f"{c}_inc"]
    st = st[st.index.get_level_values("cohort") == "beta"]
    show = st[[f"{c}_inc" for c in ("sure_day_recall", "energy_precision", "energy_iou")]
              + [f"{c}_cand" for c in ("sure_day_recall", "energy_precision", "energy_iou")]
              + [f"d_{c}" for c in ("sure_day_recall", "energy_precision", "energy_iou")]]
    print("\nBeta per station (sure days):")
    print(show.round(3).to_string())
    st.to_csv(HERE / "notes" / f"compare_{a.candidate}.csv")

    # --- verdict
    # One sure day of recall per station is the tolerance: 1 / n_sure_rpf_days (round-1 note).
    n_sure = ist.loc[st.index, "sure_rpf_days"].replace(0, np.nan)
    lose_recall = st["d_sure_day_recall"] < -(1.0 / n_sure) - EPS
    prec_pairs = st[["energy_precision_inc", "energy_precision_cand"]].dropna()
    lose_prec = (prec_pairs.energy_precision_cand - prec_pairs.energy_precision_inc) < -PREC_TOL
    alpha_ok = (cp.loc["alpha", "energy_iou"] >= ip.loc["alpha", "energy_iou"] - ALPHA_TOL
                and cp.loc["alpha", "energy_precision"] >= ip.loc["alpha", "energy_precision"] - ALPHA_TOL)
    gate_ok = bool(cp.loc["beta", "energy_precision"] >= GATE)  # Alpha sits below the gate by label construction
    improves = (cp.loc["beta", "energy_iou"] > ip.loc["beta", "energy_iou"] + EPS
                or cp.loc["beta", "sure_day_recall"] > ip.loc["beta", "sure_day_recall"] + EPS)
    print("\n--- adoption rules ---")
    print(f"rule 2a no station loses sure-day recall : {'PASS' if not lose_recall.any() else 'FAIL ' + str(list(lose_recall[lose_recall].index.get_level_values('station')))}")
    print(f"rule 2b no station loses energy precision: {'PASS' if not lose_prec.any() else 'FAIL ' + str(list(lose_prec[lose_prec].index.get_level_values('station')))}")
    print(f"rule 2c pooled Beta IoU or sure recall improves: {'PASS' if improves else 'FAIL'}")
    print(f"rule 5  Alpha within {ALPHA_TOL} on IoU and precision: {'PASS' if alpha_ok else 'FAIL'}")
    print(f"rule 1  Beta energy precision >= {GATE}: {'PASS' if gate_ok else 'FAIL'}   (Alpha {cp.loc['alpha', 'energy_precision']:.3f}, informational)")
    verdict = "ADOPT" if (not lose_recall.any() and not lose_prec.any() and improves and alpha_ok and gate_ok) else "REJECT"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
