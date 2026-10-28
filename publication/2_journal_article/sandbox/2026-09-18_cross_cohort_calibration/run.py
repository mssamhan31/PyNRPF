"""Cross-cohort calibration experiment for M9 (sandbox, reads committed outputs only).

Purpose: test whether one shared calibration layer, fitted without cohort or station
identifiers, can replace the frozen within-cohort calibration while the M9 scorer and its
evidence r stay unchanged. Nested, station-grouped validation on the 18 Phase 3 folds.

Inputs (committed, read-only):
    sandbox/2026-09-16_phase3_release/outputs/05_m9/scores_{alpha,beta}.parquet   scorer outputs and labels
    sandbox/2026-09-16_phase3_release/outputs/06_site_days/site_days.parquet      candidate energies (m9 rows)
    sandbox/2026-09-16_phase3_release/outputs/01_folds/fold_manifest.csv          held-out and training stations
Outputs (this folder): predictions_outer.parquet, fits.csv, selection.csv, metrics_pooled.csv,
    metrics_station.csv, reliability.csv, bootstrap.csv, LEAKAGE_AUDIT.md, results_summary.json.

Candidates (see PLAN.md): R0 reference, B1/B2/B3 benchmarks, C1..C4 cohort-blind candidates,
D1 transductive diagnostic. Selection among C1..C4 (and B1) per outer fold by station-macro
inner log loss subject to the inner Beta energy-precision gate; the outer result of the
selected candidate is the nested result N.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
OUT = ARTICLE / "sandbox" / "2026-09-16_phase3_release" / "outputs"
sys.path.insert(0, str(ARTICLE / "m9_dev"))
import m9_metrics as mx  # noqa: E402
import m9_scorer as ms  # noqa: E402

C = 0.7
GATE = 0.90
SEED = 9
DRAWS = 1000
CANDIDATES = ("C1", "C2", "C3", "C4", "B1")        # eligible for inner selection
ALL = ("R0", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "D1")
EPS = 1e-6


# ----------------------------------------------------------------------------- data

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.concat([pd.read_parquet(OUT / "05_m9" / f"scores_{c}.parquet") for c in ("alpha", "beta")], ignore_index=True)
    sd = pd.read_parquet(OUT / "06_site_days" / "site_days.parquet")
    sd = sd[sd["method"] == "m9"][["cohort", "station", "date", "candidate_mwh", "candidate_correct_mwh", "required_mwh"]]
    t = scores.merge(sd, on=["cohort", "station", "date"], how="left", validate="one_to_one")
    folds = pd.read_csv(OUT / "01_folds" / "fold_manifest.csv")
    # Label-free per-day quantities, computed once for every day from the frozen scorer outputs.
    # Days with input_ok = False have no evidence; their features are zero-filled and their
    # probability is masked to NaN (always UNCERTAIN), exactly as in the frozen harness.
    t["r_best"] = t["r_best"].fillna(0.0)
    t["margin_window"] = t["margin_window"].fillna(0.0)
    t["z"] = ms.signed_log(t["r_best"].to_numpy(float))
    best_len = (t["best_end"] - t["best_start"] + 1).where(t["best_start"] >= 0)
    runner_len = (t["runner_end"] - t["runner_start"] + 1).where(t["runner_start"] >= 0)
    t["L"] = best_len.fillna(runner_len).fillna(1).clip(lower=1).astype(float)
    t["z_per_slot"] = ms.signed_log(t["r_best"].to_numpy(float) / t["L"].to_numpy(float))
    t["logL"] = np.log(t["L"].to_numpy(float))
    t["m"] = ms.signed_log(t["margin_window"].fillna(0.0).to_numpy(float))
    t["is_alpha"] = (t["cohort"] == "alpha").astype(float)
    t["eligible"] = t["input_ok"] & t["headline"]
    return t, folds


# ----------------------------------------------------------------------------- candidates

class Candidate:
    """One calibration layer: a feature map fitted on training days, then a logistic.

    ``fit_transform`` may estimate label-free constants on the training days (C4, D1);
    labels are used only by the logistic.
    """

    def __init__(self, name: str):
        self.name = name
        self.model: LogisticRegression | None = None
        self.constants: dict = {}

    def features(self, t: pd.DataFrame, training: bool) -> np.ndarray:
        n = self.name
        if n in ("R0", "B1", "B2"):
            return t[["z"]].to_numpy(float)
        if n == "B3":
            return t[["z", "is_alpha"]].to_numpy(float)
        if n == "C1":
            return t[["z_per_slot"]].to_numpy(float)
        if n == "C2":
            return t[["z", "logL"]].to_numpy(float)
        if n == "C3":
            return t[["z", "m"]].to_numpy(float)
        if n == "C4":
            # Logit of the empirical null CDF of r, the null being training days where the null
            # wins (r <= 0). Pooled over training stations, cohort-blind, no labels.
            if training:
                null = np.sort(t.loc[t["r_best"] <= 0, "r_best"].to_numpy(float))
                self.constants = {"null_sorted": null}
            null = self.constants["null_sorted"]
            u = (np.searchsorted(null, t["r_best"].to_numpy(float), side="right") + 0.5) / (null.size + 1)
            u = np.clip(u, 1.0 / (null.size + 1), null.size / (null.size + 1))
            q = np.log(u / (1 - u))
            # Above the null range every day maps to the same cap; keep the evidence ordering there.
            above = t["r_best"].to_numpy(float) > null[-1]
            q = np.where(above, q + ms.signed_log(t["r_best"].to_numpy(float) - null[-1]), q)
            return q.reshape(-1, 1)
        if n == "D1":
            # Transductive diagnostic: z standardised by the station's own null-day median and MAD.
            zs = np.empty(len(t))
            for station, idx in t.groupby("station").indices.items():
                z = t["z"].to_numpy(float)[idx]
                nullz = z[t["r_best"].to_numpy(float)[idx] <= 0]
                med = np.median(nullz) if nullz.size >= 20 else 0.0
                mad = np.median(np.abs(nullz - med)) if nullz.size >= 20 else 1.0
                zs[idx] = (z - med) / max(mad, 0.05)
            return zs.reshape(-1, 1)
        raise ValueError(n)

    def fit(self, train: pd.DataFrame) -> "Candidate":
        x = self.features(train, training=True)
        y = train["rpf"].to_numpy(int)
        w = None
        if self.name == "B2":
            a = train["cohort"].to_numpy() == "alpha"
            w = np.where(a, 1.0 / a.sum(), 1.0 / (~a).sum())
        self.model = LogisticRegression(max_iter=5000).fit(x, y, sample_weight=w)
        return self

    def predict(self, t: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(self.features(t, training=False))[:, 1]
        return np.where(t["input_ok"].to_numpy(), p, np.nan)

    def coefficients(self) -> dict:
        d = {"intercept": float(self.model.intercept_[0])}
        for k, c in enumerate(self.model.coef_[0]):
            d[f"coef_{k}"] = float(c)
        if "null_sorted" in self.constants:
            d["null_n"] = int(self.constants["null_sorted"].size)
            d["null_max_r"] = float(self.constants["null_sorted"][-1])
        return d


def decide_frame(t: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    out = t.copy()
    out["p"] = p
    out["outcome"] = [ms.decide(v, C) if np.isfinite(v) else ms.UNCERTAIN for v in p]
    no_window = out["best_start"] < 0
    out.loc[no_window, "outcome"] = out.loc[no_window, "outcome"].replace(ms.AUTO_CORRECT, ms.UNCERTAIN)
    applied = out["outcome"] == ms.AUTO_CORRECT
    out["proposed_mwh"] = np.where(applied, out["candidate_mwh"], 0.0)
    out["correct_mwh"] = np.where(applied, out["candidate_correct_mwh"], 0.0)
    return out


# ----------------------------------------------------------------------------- metrics

def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def station_macro_log_loss(pred: pd.DataFrame) -> float:
    vals = []
    for _, g in pred[pred["eligible"] & pred["p"].notna()].groupby("station"):
        vals.append(log_loss(g["rpf"].to_numpy(float), g["p"].to_numpy(float)))
    return float(np.mean(vals))


def reliability(pred: pd.DataFrame) -> dict:
    g = pred[pred["eligible"] & pred["p"].notna()]
    rel = mx.calibration_reliability(g["p"].to_numpy(float), g["rpf"].to_numpy(float))
    rel["log_loss"] = log_loss(g["rpf"].to_numpy(float), g["p"].to_numpy(float))
    rel["n_days"] = int(len(g))
    return rel


def summarise(pred: pd.DataFrame) -> dict:
    g = pred[pred["headline"]]
    s = mx.summarise(g)
    return {k: s[k] for k in ("n_days", "n_rpf", "energy_iou", "energy_precision", "day_precision", "day_recall", "day_f1",
                              "sure_day_recall", "rate_uncertain", "tp", "fp", "fn")}


def station_bootstrap(pred: pd.DataFrame, rng: np.random.Generator) -> dict:
    """2.5–97.5 percentile intervals over station resampling (each cohort resampled within itself)."""
    g = pred[pred["headline"]]
    by = {k: v for k, v in g.groupby(["cohort", "station"])}
    cohorts = {c: sorted(s for (cc, s) in by if cc == c) for c in g["cohort"].unique()}
    samples = {k: [] for k in ("energy_iou", "energy_precision", "sure_day_recall")}
    for _ in range(DRAWS):
        pick = [by[(c, s)] for c, stations in cohorts.items() for s in rng.choice(stations, size=len(stations), replace=True)]
        m = mx.summarise(pd.concat(pick, ignore_index=True))
        for k in samples:
            samples[k].append(m[k])
    return {f"{k}_lo": float(np.nanpercentile(v, 2.5)) for k, v in samples.items()} | {f"{k}_hi": float(np.nanpercentile(v, 97.5)) for k, v in samples.items()}


# ----------------------------------------------------------------------------- the experiment

def training_rows(t: pd.DataFrame, stations: list[str], same_cohort_as: str | None = None) -> pd.DataFrame:
    train = t[t["station"].isin(stations) & t["eligible"]]
    if same_cohort_as is not None:
        train = train[train["cohort"] == same_cohort_as]
    return train


def run() -> None:
    t0 = time.time()
    t, folds = load()
    fits, selection, preds = [], [], []
    audit_lines = []
    for f in folds.itertuples(index=False):
        held = f.held_out
        others = f.m8_training_stations.split(";")            # the 17 other stations, both cohorts
        same = f.m9_calibration_stations.split(";")            # the other stations of the same cohort
        test = t[t["station"] == held]
        train17 = training_rows(t, others)
        audit_lines.append(f"- Fold {f.fold_id}: outer-training stations {len(others)} ({sum(s.startswith('alpha') for s in others)} Alpha, "
                           f"{sum(s.startswith('beta') for s in others)} Beta), {len(train17):,} eligible days with labels. "
                           f"Held-out {held}: {len(test):,} days, scorer outputs and label-free features only; labels used only to score the result.")
        # Inner loop over the 17 outer-training stations, candidates fitted on 16, scored on the 17th.
        inner_pred = {c: [] for c in CANDIDATES}
        for s in others:
            inner_train = train17[train17["station"] != s]
            inner_test = t[(t["station"] == s) & t["eligible"]]
            for c in CANDIDATES:
                cand = Candidate(c).fit(inner_train)
                inner_pred[c].append(decide_frame(inner_test, cand.predict(inner_test)))
        rows = []
        for c in CANDIDATES:
            ip = pd.concat(inner_pred[c], ignore_index=True)
            beta_ip = ip[ip["cohort"] == "beta"]
            beta_prec = mx.summarise(beta_ip)["energy_precision"] if len(beta_ip) else np.nan
            rows.append(dict(fold_id=f.fold_id, candidate=c, inner_macro_log_loss=station_macro_log_loss(ip),
                             inner_beta_energy_precision=beta_prec, inner_gate_pass=bool(beta_prec >= GATE),
                             inner_combined_energy_iou=mx.summarise(ip)["energy_iou"]))
        table = pd.DataFrame(rows)
        passing = table[table["inner_gate_pass"]]
        pool = passing if len(passing) else table
        chosen = pool.sort_values("inner_macro_log_loss").iloc[0]["candidate"]
        table["selected"] = table["candidate"] == chosen
        table["gate_fallback"] = not len(passing)
        selection.append(table)
        # Outer: every candidate refitted on the 17 (R0 on the same cohort only), applied once to the held-out station.
        for c in ALL:
            train = training_rows(t, same) if c == "R0" else train17
            cand = Candidate(c).fit(train)
            p = cand.predict(test)
            pr = decide_frame(test, p)
            pr["candidate"] = c
            pr["fold_id"] = f.fold_id
            pr["selected"] = (c == chosen)
            preds.append(pr)
            fits.append(dict(fold_id=f.fold_id, cohort=f.cohort, held_out=held, candidate=c, n_train=len(train), **cand.coefficients()))
        nested = [p for p in preds if p["fold_id"].iloc[0] == f.fold_id and p["candidate"].iloc[0] == chosen][0].copy()
        nested["candidate"] = "N"
        preds.append(nested)
        print(f"{f.fold_id}: selected {chosen} ({'gate fallback' if not len(passing) else 'gate met'})", flush=True)

    pred = pd.concat(preds, ignore_index=True)
    keep = ["candidate", "fold_id", "cohort", "station", "date", "confidence", "headline", "eligible", "rpf", "r_best", "L", "z", "p", "outcome",
            "candidate_mwh", "candidate_correct_mwh", "proposed_mwh", "required_mwh", "correct_mwh", "selected"]
    pred[keep].to_parquet(HERE / "predictions_outer.parquet", index=False)
    pd.DataFrame(fits).to_csv(HERE / "fits.csv", index=False)
    pd.concat(selection, ignore_index=True).to_csv(HERE / "selection.csv", index=False)

    # Pooled, per-station, reliability, bootstrap.
    rng = np.random.default_rng(SEED)
    pooled, stations, rel, boot = [], [], [], []
    for c in list(ALL) + ["N"]:
        pc = pred[pred["candidate"] == c]
        for group in ("combined", "alpha", "beta"):
            g = pc if group == "combined" else pc[pc["cohort"] == group]
            pooled.append(dict(candidate=c, group=group, **summarise(g)))
            rel.append(dict(candidate=c, group=group, **reliability(g)))
            boot.append(dict(candidate=c, group=group, **station_bootstrap(g, rng)))
        for (cohort, station), g in pc[pc["headline"]].groupby(["cohort", "station"]):
            s = mx.summarise(g)
            stations.append(dict(candidate=c, cohort=cohort, station=station, n_days=s["n_days"], n_rpf=s["n_rpf"], energy_iou=s["energy_iou"],
                                 energy_precision=s["energy_precision"], sure_day_recall=s["sure_day_recall"], fp=s["fp"], fn=s["fn"],
                                 rate_uncertain=s["rate_uncertain"]))
    pd.DataFrame(pooled).to_csv(HERE / "metrics_pooled.csv", index=False)
    pd.DataFrame(stations).to_csv(HERE / "metrics_station.csv", index=False)
    pd.DataFrame(rel).to_csv(HERE / "reliability.csv", index=False)
    pd.DataFrame(boot).to_csv(HERE / "bootstrap.csv", index=False)

    audit = ["# Leakage audit\n", "What each outer fold was permitted to read. Every number below is derived from committed Phase 3 outputs; "
             "the scorer was not re-run and no dataset was opened.\n",
             "Global, before any fold: the scorer outputs (r, window, runner-up, margin) for every site-day, computed in Phase 3 with a "
             "cohort-level label-free floor φ (a pre-existing property of the frozen scores, recorded in the Phase 0 decision). "
             "The per-day features z, L, z_per_slot, logL and m are label-free functions of those outputs.\n",
             "Per outer fold:\n"] + audit_lines + [
             "\nInner loop: within the 17 outer-training stations only; each inner held-out station's labels are used only to score "
             "the inner prediction; C4's null distribution and every logistic are fitted on the 16 inner-training stations.\n",
             "Selection: by station-macro inner log loss under the inner Beta energy-precision gate, fixed in PLAN.md before any outer result was computed.\n",
             "D1 (diagnostic only): standardises each station's z by that station's own null-day median and MAD, including the held-out "
             "station's unlabelled days; transductive by design and excluded from selection and from the success result.\n",
             "B3 (diagnostic only): receives the cohort indicator; excluded from selection and from the success result.\n",
             "R0 (reference): fitted on the same-cohort outer-training stations only, as in Phase 3.\n"]
    (HERE / "LEAKAGE_AUDIT.md").write_text("\n".join(audit), encoding="utf-8")

    summary = {"elapsed_s": round(time.time() - t0, 1), "selection_counts": pd.concat(selection)[lambda d: d["selected"]]["candidate"].value_counts().to_dict(),
               "gate_fallback_folds": pd.concat(selection)[lambda d: d["selected"] & d["gate_fallback"]]["fold_id"].tolist()}
    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 240)
    show = pd.DataFrame(pooled)[["candidate", "group", "energy_iou", "energy_precision", "sure_day_recall", "day_precision", "day_f1", "fp", "fn", "rate_uncertain"]]
    print(show.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
