"""Tables for the bridge-anchor experiment: pooled and per-station metrics under the three
anchor rules, the reproduction check of the frozen reference, the beta_D detail table of
gap-adjacent labelled reverse power flow (RPF) days, and the days whose outcome changed.

Inputs:  ../../m9_dev/runs/phase5_final_rev2/     frozen reference (nearest anchors, commit ffa1d92)
         runs/{nearest,edge,gap_edge}/            this experiment, frozen settings, one anchor rule each
         ../../dataset/final/dataset_beta.parquet  labelled slots and missing readings for beta_D
Outputs: tables/pooled.csv and .md, tables/per_station.csv and .md, tables/reproduction_check.md,
         tables/beta_D_gap_days.csv and .md, tables/outcome_changes.csv and .md.
         Every path written into a table is repository-relative.
Key steps: read the run summaries; diff nearest against the frozen reference row by row;
         find beta_D labelled days whose span abuts a missing reading; join each rule's
         held-out prediction for those days; count outcome changes per station.

Run from the repository root:
    python publication/2_journal_article/sandbox/2026-09-22_bridge_anchors/make_tables.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
FROZEN_RUN = ARTICLE / "m9_dev" / "runs" / "phase5_final_rev2"
RUNS = HERE / "runs"
TABLES = HERE / "tables"
RULES = ("nearest", "edge", "gap_edge")
SLOTS = 96
KEY = ["cohort", "station", "date"]
OUTCOME_SHORT = {"AUTO_CORRECT": "AC", "AUTO_KEEP": "AK", "UNCERTAIN": "UNC"}


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """A GitHub-flavoured markdown table; floats rounded, NaN shown as a dash."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("—" if np.isnan(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def predictions(run_dir: pathlib.Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(run_dir / f"predictions_{c}.csv") for c in ("alpha", "beta")], ignore_index=True)


# --------------------------------------------------------------------------- pooled and per station

def pooled_tables() -> None:
    cols = ["n_days", "n_rpf", "energy_iou", "energy_precision", "sure_day_recall", "sure_day_uncertain_rate",
            "day_precision", "day_recall", "day_f1", "rate_auto_correct", "rate_uncertain", "ece"]
    rows = []
    for rule in RULES:
        p = pd.read_csv(RUNS / rule / "summary_pooled.csv")
        p.insert(0, "anchors", rule)
        rows.append(p)
    pooled = pd.concat(rows, ignore_index=True)
    pooled.to_csv(TABLES / "pooled.csv", index=False)
    parts = []
    for cohort, group, title in (("beta", "headline", "Beta `sure`"), ("alpha", "headline", "Alpha"), ("beta", "unsure_sensitivity", "Beta `unsure` (sensitivity only)")):
        t = pooled[(pooled.cohort == cohort) & (pooled.group == group)][["anchors"] + [c for c in cols if c in pooled]]
        parts.append(f"### {title}\n\n" + md_table(t.reset_index(drop=True), 4))
    (TABLES / "pooled.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def per_station_tables() -> None:
    metrics = ["energy_iou", "energy_precision", "sure_day_recall", "day_f1", "rate_auto_correct"]
    rows = []
    for rule in RULES:
        s = pd.read_csv(RUNS / rule / "summary_station.csv")
        s.insert(0, "anchors", rule)
        rows.append(s)
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(TABLES / "per_station.csv", index=False)
    parts = []
    for cohort in ("beta", "alpha"):
        base = long[(long.anchors == "nearest") & (long.cohort == cohort)][["station", "n_days", "n_rpf"]].set_index("station")
        wide = base.copy()
        for m in metrics:
            for rule in RULES:
                col = long[(long.anchors == rule) & (long.cohort == cohort)].set_index("station")[m]
                wide[f"{m} {rule}"] = col
        wide = wide.reset_index()
        parts.append(f"### {cohort}\n\n" + md_table(wide, 3))
    (TABLES / "per_station.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- reproduction check

def reproduction_check() -> str:
    """Row-by-row comparison of the sandbox nearest run against the frozen reference."""
    lines = ["# Reproduction check: sandbox `nearest` against `m9_dev/runs/phase5_final_rev2`", ""]
    for name in ("summary_pooled", "summary_station"):
        f = pd.read_csv(FROZEN_RUN / f"{name}.csv")
        n = pd.read_csv(RUNS / "nearest" / f"{name}.csv")
        num = f.select_dtypes("number").columns
        diff = (n[num] - f[num]).abs().max().max()
        same_shape = f.shape == n.shape
        lines.append(f"- `{name}.csv`: same shape {same_shape}; largest absolute difference over all numeric cells = {diff:.3g}")
    f = predictions(FROZEN_RUN)
    n = predictions(RUNS / "nearest")
    m = f.merge(n, on=KEY, suffixes=("_frozen", "_nearest"))
    lines.append(f"- predictions: {len(f)} frozen rows, {len(n)} sandbox rows, {len(m)} matched on (cohort, station, date)")
    for col in ("r_best", "p", "best_start", "best_end", "runner_start", "runner_end", "n_admissible", "proposed_mwh"):
        a, b = m[f"{col}_frozen"], m[f"{col}_nearest"]
        equal = ((a == b) | (a.isna() & b.isna())).sum()
        lines.append(f"- `{col}`: {equal} of {len(m)} rows identical (exact equality); largest absolute difference {np.nanmax((a - b).abs()):.3g}")
    same_outcome = (m["outcome_frozen"] == m["outcome_nearest"]).sum()
    lines.append(f"- `outcome`: {same_outcome} of {len(m)} rows identical")
    text = "\n".join(lines) + "\n"
    (TABLES / "reproduction_check.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- beta_D gap-adjacent days

def beta_d_days() -> pd.DataFrame:
    """beta_D labelled RPF days whose span abuts a missing reading, with the truth slots."""
    df = pd.read_parquet(ARTICLE / "dataset" / "final" / "dataset_beta.parquet")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["d"] = df["ts"].dt.date.astype(str)
    df = df[df.substation_id == "beta_D"].sort_values("ts")
    rows = []
    for d, g in df.groupby("d"):
        if len(g) != SLOTS:
            continue
        truth = g["label_interval"].to_numpy(bool)
        if not truth.any():
            continue
        finite = np.isfinite(g["net_load_MW"].to_numpy(float)) & np.isfinite(g["solar_MW"].to_numpy(float))
        idx = np.flatnonzero(truth)
        a, b = int(idx[0]), int(idx[-1])
        before, after = (a > 0 and not finite[a - 1]), (b < SLOTS - 1 and not finite[b + 1])
        if not (before or after):
            continue
        rows.append(dict(date=d, confidence=g["confidence"].iloc[0], truth_start=a, truth_end=b,
                         gap="both" if before and after else ("before" if before else "after"), truth=truth))
    out = pd.DataFrame(rows)
    order = out["confidence"].map({"sure": 0, "unsure": 1})
    return out.assign(_o=order).sort_values(["_o", "date"]).drop(columns="_o").reset_index(drop=True)


def window_iou(start: int, end: int, truth: np.ndarray) -> float:
    if start < 0:
        return 0.0
    w = np.zeros(SLOTS, dtype=bool)
    w[start : end + 1] = True
    return float((w & truth).sum() / (w | truth).sum())


def beta_d_table() -> tuple[pd.DataFrame, str]:
    days = beta_d_days()
    preds = {rule: pd.read_csv(RUNS / rule / "predictions_beta.csv").query("station == 'beta_D'").set_index("date") for rule in RULES}
    long_rows, wide_rows = [], []
    for _, day in days.iterrows():
        wide = dict(date=day.date, conf=day.confidence, truth=f"{day.truth_start}–{day.truth_end}", gap=day.gap)
        for rule in RULES:
            r = preds[rule].loc[day.date]
            null_won = r.best_start < 0
            # When the null wins the best window is kept as the runner-up; show it in brackets.
            ws, we = (r.runner_start, r.runner_end) if null_won else (r.best_start, r.best_end)
            iou = window_iou(int(ws), int(we), day.truth)
            exact = bool(ws == day.truth_start and we == day.truth_end)
            long_rows.append(dict(date=day.date, confidence=day.confidence, truth_start=day.truth_start, truth_end=day.truth_end, gap=day.gap,
                                  anchors=rule, window_start=int(ws), window_end=int(we), null_won=null_won, r=r.r_best, p=r.p, outcome=r.outcome,
                                  exact=exact, iou=iou, iou_ge_0_8=bool(iou >= 0.8)))
            window = f"({int(ws)}–{int(we)})" if null_won else f"{int(ws)}–{int(we)}"
            match = "exact" if exact else ("IoU≥0.8" if iou >= 0.8 else f"IoU {iou:.2f}")
            wide.update({f"{rule} window": window, f"{rule} r": r.r_best, f"{rule} p": r.p, f"{rule} dec": OUTCOME_SHORT[r.outcome], f"{rule} match": match})
        wide_rows.append(wide)
    long = pd.DataFrame(long_rows)
    long.to_csv(TABLES / "beta_D_gap_days.csv", index=False)
    wide = pd.DataFrame(wide_rows)
    # Summary counts per rule, sure and unsure separately.
    summ = []
    for conf in ("sure", "unsure"):
        for rule in RULES:
            t = long[(long.confidence == conf) & (long.anchors == rule)]
            summ.append(dict(confidence=conf, anchors=rule, days=len(t), auto_correct=int((t.outcome == "AUTO_CORRECT").sum()),
                             uncertain=int((t.outcome == "UNCERTAIN").sum()), exact=int(t.exact.sum()), iou_ge_0_8=int(t.iou_ge_0_8.sum()),
                             median_r=float(t.r.median()), median_p=float(t.p.median())))
    summ = pd.DataFrame(summ)
    text = ("### beta_D labelled RPF days whose span abuts a missing reading\n\n"
            "`truth` is the labelled span (slots, inclusive); `gap` says which side of the span has a missing reading next to it. "
            "Under each rule: the best window, or in brackets the best window when the null won; evidence `r`; held-out `p`; "
            "decision (AC = AUTO_CORRECT, AK = AUTO_KEEP, UNC = UNCERTAIN); and whether the shown window equals the truth exactly, "
            "has slot IoU ≥ 0.8, or its IoU otherwise.\n\n" + md_table(wide, 1) + "\n\n### Summary over those days\n\n" + md_table(summ, 2))
    (TABLES / "beta_D_gap_days.md").write_text(text + "\n", encoding="utf-8")
    return long, text


# --------------------------------------------------------------------------- what changed elsewhere

def outcome_changes() -> str:
    """Per station, days whose held-out outcome differs from nearest (headline days only)."""
    base = predictions(RUNS / "nearest")
    base = base[base.confidence.isin(["sure", "controlled"])]
    rows = []
    for rule in ("edge", "gap_edge"):
        cand = predictions(RUNS / rule)
        m = base.merge(cand, on=KEY, suffixes=("_n", "_c"))
        ac_n, ac_c = m.outcome_n == "AUTO_CORRECT", m.outcome_c == "AUTO_CORRECT"
        pos = m.rpf_n == 1
        for (cohort, station), g in m.groupby(["cohort", "station"]):
            i = g.index
            rows.append(dict(anchors=rule, cohort=cohort, station=station, n_days=len(g),
                             outcome_changed=int((g.outcome_n != g.outcome_c).sum()),
                             tp_gained=int((~ac_n[i] & ac_c[i] & pos[i]).sum()), tp_lost=int((ac_n[i] & ~ac_c[i] & pos[i]).sum()),
                             fp_gained=int((~ac_n[i] & ac_c[i] & ~pos[i]).sum()), fp_lost=int((ac_n[i] & ~ac_c[i] & ~pos[i]).sum()),
                             window_changed=int(((g.best_start_n != g.best_start_c) | (g.best_end_n != g.best_end_c)).sum()),
                             correct_mwh_delta=float(g.loc[ac_c[i], "correct_mwh_c"].sum() - g.loc[ac_n[i], "correct_mwh_n"].sum()),
                             proposed_mwh_delta=float(g.loc[ac_c[i], "proposed_mwh_c"].sum() - g.loc[ac_n[i], "proposed_mwh_n"].sum())))
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "outcome_changes.csv", index=False)
    text = "### Days whose outcome or window changed against `nearest` (headline days)\n\n" + md_table(t, 1)
    (TABLES / "outcome_changes.md").write_text(text + "\n", encoding="utf-8")
    return text


def evidence_distribution() -> str:
    """Per rule and cohort: where the evidence sits on RPF and non-RPF headline days, and
    the raw AUTO_CORRECT threshold the folds applied. Explains a collapse or a loss as a
    shift of the false-window evidence rather than of the true-window evidence."""
    rows = []
    for rule in RULES:
        pred = predictions(RUNS / rule)
        pred = pred[pred.confidence.isin(["sure", "controlled"]) & pred.input_ok]
        fits = pd.read_csv(RUNS / rule / "calibration_fits.csv")
        for cohort, g in pred.groupby("cohort"):
            q = lambda t: t.r_best.quantile([0.25, 0.5, 0.75]).to_numpy()  # noqa: E731
            r1, r0 = q(g[g.rpf == 1]), q(g[g.rpf == 0])
            thr = fits[fits.cohort == cohort].raw_threshold_correct
            rows.append(dict(anchors=rule, cohort=cohort, rpf_r_q25=r1[0], rpf_r_median=r1[1], rpf_r_q75=r1[2],
                             non_rpf_r_q25=r0[0], non_rpf_r_median=r0[1], non_rpf_r_q75=r0[2],
                             non_rpf_r_q99=float(g[g.rpf == 0].r_best.quantile(0.99)),
                             threshold_median=float(thr.median()), threshold_min=float(thr.min()), threshold_max=float(thr.max())))
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "evidence.csv", index=False)
    text = ("### Evidence r on headline days and the raw AUTO_CORRECT threshold per rule\n\n"
            "Quartiles of the best-window evidence `r_best` on RPF and non-RPF days (`input_ok` only), "
            "and the raw threshold at p = 0.7 across the leave-one-station-out folds.\n\n" + md_table(t, 1))
    (TABLES / "evidence.md").write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # the tables carry "≥" and "—"; a cp1252 console cannot print them
    TABLES.mkdir(exist_ok=True)
    pooled_tables()
    per_station_tables()
    print(reproduction_check())
    print((TABLES / "pooled.md").read_text(encoding="utf-8"))
    print((TABLES / "per_station.md").read_text(encoding="utf-8"))
    _, text = beta_d_table()
    print(text)
    print(outcome_changes())
    print(evidence_distribution())
