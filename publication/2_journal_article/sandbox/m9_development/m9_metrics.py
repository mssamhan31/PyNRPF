"""Locked evaluation metrics for M9 development: energy-weighted and site-day.

Purpose: implement the PRD's provisional correction-energy definitions and the
four headline metrics, pooled across site-days by summing energies, with per-station
breakdowns and the calibration reliability checks.

Inputs:  a site-day prediction table with, per row, the proposed window, the
         labelled slots, recorded net load and the decision outcome.
Outputs: pooled and per-station tables of Reference Energy IoU, reference energy
         precision, site-day precision / recall / F1, sure-day recall, and outcome
         rates; expected calibration error, Brier score, calibration-in-the-large.
Key steps: per slot the correction changes the reading by 2*y, so correction energy
         on a slot is 2*y*0.25 MWh; intersect proposed and labelled slots; sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_SLOT = 0.25


def slot_energy(y: np.ndarray) -> np.ndarray:
    """Correction energy per slot if that slot were flipped, MWh. Missing readings carry none."""
    return 2.0 * np.nan_to_num(y, nan=0.0) * HOURS_PER_SLOT


def energy_terms(y: np.ndarray, truth: np.ndarray, start: int, end: int) -> tuple[float, float, float]:
    """(proposed, required, correctly corrected) MWh for a window; start < 0 means no window."""
    e = slot_energy(y)
    prop = np.zeros_like(truth, dtype=bool)
    if start >= 0:
        prop[start : end + 1] = True
    return float(e[prop].sum()), float(e[truth].sum()), float(e[prop & truth].sum())


def pooled_energy(t: pd.DataFrame, applied: np.ndarray) -> tuple[float, float]:
    """Pooled (Energy IoU, energy precision) over site-days; `applied` marks days corrected."""
    p = float(t.loc[applied, "proposed_mwh"].sum())
    c = float(t.loc[applied, "correct_mwh"].sum())
    r = float(t["required_mwh"].sum())
    union = p + r - c
    return (c / union if union > 0 else 0.0), (c / p if p > 0 else np.nan)


def day_metrics(t: pd.DataFrame, applied: np.ndarray) -> dict:
    """Site-day precision, recall and F1 treating 'applied' as the positive prediction."""
    pos = t["rpf"].to_numpy().astype(bool)
    tp = int((applied & pos).sum())
    fp = int((applied & ~pos).sum())
    fn = int((~applied & pos).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and np.isfinite(prec + rec)) else 0.0
    return dict(day_precision=prec, day_recall=rec, day_f1=f1, tp=tp, fp=fp, fn=fn)


def summarise(t: pd.DataFrame) -> dict:
    """All headline numbers for one group of site-days (a cohort or one station)."""
    applied = (t["outcome"] == "AUTO_CORRECT").to_numpy()
    iou, eprec = pooled_energy(t, applied)
    out = dict(n_days=len(t), n_rpf=int(t["rpf"].sum()), required_mwh=float(t["required_mwh"].sum()),
               applied_mwh=float(t.loc[applied, "proposed_mwh"].sum()),
               energy_iou=iou, energy_precision=eprec)
    out.update(day_metrics(t, applied))
    for o in ("AUTO_CORRECT", "AUTO_KEEP", "UNCERTAIN"):
        out[f"rate_{o.lower()}"] = float((t["outcome"] == o).mean())
    # Human-parity number: share of 'sure' RPF days that were corrected. On Alpha
    # every RPF day counts as sure.
    sure = t[(t["rpf"] == 1) & (t["confidence"].isin(["sure", "controlled"]))]
    out["sure_rpf_days"] = len(sure)
    out["sure_day_recall"] = float((sure["outcome"] == "AUTO_CORRECT").mean()) if len(sure) else np.nan
    out["sure_day_uncertain_rate"] = float((sure["outcome"] == "UNCERTAIN").mean()) if len(sure) else np.nan
    return out


def station_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per-station summary, one row each, ordered by station id."""
    rows = [dict(station=sid, **summarise(g)) for sid, g in t.groupby("station", sort=True)]
    return pd.DataFrame(rows)


def calibration_reliability(p: np.ndarray, label: np.ndarray, bins: int = 10) -> dict:
    """Expected calibration error (equal-count bins), Brier score and calibration-in-the-large."""
    p = np.asarray(p, dtype=float)
    label = np.asarray(label, dtype=float)
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, bins)
    ece = 0.0
    for idx in chunks:
        if idx.size:
            ece += idx.size / p.size * abs(p[idx].mean() - label[idx].mean())
    return dict(ece=float(ece), brier=float(((p - label) ** 2).mean()),
                calibration_in_the_large=float(abs(p.mean() - label.mean())))
