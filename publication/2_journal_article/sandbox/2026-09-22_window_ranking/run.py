"""Within-day window ranking of M9: where does the true window sit among the 1,176 candidates?

Purpose: the Phase 3 release reports M9 by the one window it proposes. This sandbox asks
a finer question. When M9 scores every candidate window of a labelled reverse power flow
(RPF) day with the frozen revision-2 settings, at what rank does the true window, or a
window close to it, appear; how good is the rank-1 window; and how much would a reviewer
gain by choosing among the top three or five? Scoring is label-free; labels enter only
when the ranked windows are measured against the truth.

Inputs (read-only):
    config/final_evaluation.yaml and the frozen datasets it names   hash-checked by final_eval.config
    m9_dev/m9_scorer.py                                             the frozen scorer, imported, never copied
    sandbox/2026-09-16_phase3_release/outputs/05_m9/scores_{alpha,beta}.parquet
                                                     committed best windows, scores and admissible counts (self-check)
    sandbox/2026-09-16_phase3_release/outputs/06_site_days/site_days.parquet
                                                     committed per-day M9 candidate energies (energy check)
Outputs (this folder):
    tables/*.csv, tables/*.md   self-check, rank distributions (per population, per station), rank-1
                                quality, top-k oracles, failure geometry, truth admissibility, per-day rows
    figures/*.png               rank distribution bars, rank-1 IoU cumulative distribution, per-station
                                exact-match share
Key steps:
    1. load the Phase 3 population exactly as final_eval does (complete days; cohort sigma floor);
    2. for every labelled RPF day recompute the full 48 x 48 evidence matrix and the frozen
       admissibility mask (missing = mask_windows, edges = inwardx) without labels;
    3. self-check the masked argmax, its score and the admissible count against the committed
       Phase 3 scores, and the rank-1 energies against the committed site-day table;
    4. rank the windows (score descending, then shorter, then earlier) among admissible windows
       and among all scoreable windows; locate the exact truth span and the first window with
       interval IoU >= 0.8; measure the rank-1 window and the best of the top 3 and top 5;
    5. aggregate per population and per station, classify rank-1 failures by overlap geometry,
       write the tables and figures.
Units: net load and solar in MW; energies in MWh (2 * |y| * 0.25 per flipped slot, missing
readings carry none); slots are 15-minute intervals indexed 0-95 from midnight; a window is
an inclusive [start, end] slot pair inside slots 24-71.
Runtime: 20-40 seconds. Deterministic; no random numbers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
PHASE3 = ARTICLE / "sandbox" / "2026-09-16_phase3_release" / "outputs"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
sys.path.insert(0, str(ARTICLE))
sys.path.insert(0, str(ARTICLE / "m9_dev"))
import m9_scorer as ms  # noqa: E402  the frozen scorer, imported, never copied
from final_eval import config, data  # noqa: E402
from final_eval import m9 as m9_engine  # noqa: E402

# The frozen revision-2 settings this sandbox reproduces. The run refuses to start if the
# configuration says anything else, so a changed config cannot be mistaken for Phase 3.
FROZEN = dict(variant="sq", stat="llr", p_exp=0.0, missing="mask_windows", edges="inwardx")
HOURS_PER_SLOT = 0.25
IOU_TARGET = 0.8
TOP_K = (1, 3, 5)
KEY = ["cohort", "station", "date"]

RANK_BINS = ("1", "2", "3", "4-10", ">10")
NOT_ADMISSIBLE = "not admissible"       # truth span fails the frozen admissibility rules
NOT_SCOREABLE = "not scoreable"         # truth span holds a missing reading or lies outside slots 24-71
NO_WINDOW_REACHES = "none >= 0.8"       # no ranked window reaches the IoU target
ALL_BINS = RANK_BINS + (NOT_ADMISSIBLE, NOT_SCOREABLE, NO_WINDOW_REACHES)
POPULATIONS = ("Alpha", "Beta sure", "Beta unsure")
HEADLINE = ("Alpha", "Beta sure")
GEOMETRY = ("exact", "too long (start)", "too long (end)", "too long (both)", "too short",
            "shifted earlier", "shifted later", "disjoint")

# Journal palette, as in final_eval/figures.py.
COLORS = {"orange": "#eb932c", "dark_blue": "#22303d", "grey": "#2F4D67", "light_grey": "#5C7D99",
          "light_white": "#ebe3e3"}
POP_COLORS = {"Alpha": COLORS["grey"], "Beta sure": COLORS["orange"], "Beta unsure": COLORS["light_grey"]}
POP_STYLES = {"Alpha": "-", "Beta sure": "--", "Beta unsure": ":"}

# Window index grids: window (i, j) is slots a = 24 + i .. b = 24 + j, valid when j >= i.
_I = np.arange(ms.N_WINDOWS)[:, None]
_J = np.arange(ms.N_WINDOWS)[None, :]
UPPER = _J >= _I
LENGTH = np.where(UPPER, _J - _I + 1, 0)


# ----------------------------------------------------------------------------- inputs

def load_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Load the Phase 3 population and the committed reference tables.

    Returns:
        intervals: complete-day interval frame per cohort (final_eval.data.load_population).
        floors: the cohort sigma floor, MW, from final_eval.m9.sigma_floor (label-free).
        scores: committed Phase 3 M9 score rows of both cohorts (05_m9/scores_*.parquet).
        site_days: committed Phase 3 M9 site-day rows with candidate energies in MWh.
    """
    settings = config.load()
    m9cfg = settings["m9"]
    found = dict(variant=m9cfg["variant"], stat=m9cfg["stat"], p_exp=float(m9cfg["p_exp"]),
                 missing=m9cfg["missing"], edges=m9cfg["edges"])
    if found != FROZEN:
        raise ValueError(f"Configuration m9 block {found} is not the frozen revision 2 {FROZEN}.")
    _, intervals = data.load_population(settings)
    floors = {cohort: m9_engine.sigma_floor(frame, settings) for cohort, frame in intervals.items()}
    scores = pd.concat([pd.read_parquet(PHASE3 / "05_m9" / f"scores_{c}.parquet") for c in intervals], ignore_index=True)
    site_days = pd.read_parquet(PHASE3 / "06_site_days" / "site_days.parquet")
    site_days = site_days[site_days["method"] == "m9"].reset_index(drop=True)
    return intervals, floors, scores, site_days


def population_name(cohort: str, confidence: str) -> str:
    """Alpha (all controlled days), Beta sure or Beta unsure."""
    return "Alpha" if cohort == "alpha" else f"Beta {confidence}"


def labelled_rpf_days(intervals: dict[str, pd.DataFrame]):
    """Yield one record per complete site-day with at least one labelled slot.

    Each record carries cohort, station, date, confidence and the 96-slot arrays y (MW),
    s (MW) and truth (bool). Days without a labelled slot are skipped.
    """
    for cohort, frame in intervals.items():
        for (_, station, date), g in frame.groupby(KEY, sort=True):
            truth = g["truth"].to_numpy(bool)
            if not truth.any():
                continue
            yield dict(cohort=cohort, station=station, date=date, confidence=str(g["confidence"].iloc[0]),
                       y=g["y"].to_numpy(float), s=g["s"].to_numpy(float), truth=truth)


# ----------------------------------------------------------------------------- frozen scoring (label-free)

def frozen_edge_ok(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end edge admissibility over slots 24-71 under the frozen edges = 'inwardx' rule.

    Transcribed from the ``edges.startswith("inward")`` and ``edges.endswith("x")`` branches of
    ``m9_scorer.score_siteday``, which builds the mask internally and does not return it. The
    start may sit at a local minimum of y or one slot after one; the end at a minimum or one
    slot before one; an edge whose outside neighbour is a missing reading is exempt.

    Args:
        y, s: 96-slot recorded net load and solar estimate, MW.

    Returns:
        (start_ok, end_ok), boolean arrays of length 48 indexed by window start / end.
    """
    m0 = ms.local_minimum_edges(y, tolerance=0)
    start_at_cusp = m0.copy()
    start_at_cusp[1:] |= m0[:-1]          # start one slot after a cusp is accepted (inward)
    end_at_cusp = m0.copy()
    end_at_cusp[:-1] |= m0[1:]            # end one slot before a cusp is accepted (inward)
    finite = np.isfinite(y) & np.isfinite(s)
    gap_left = np.zeros(ms.SLOTS, dtype=bool)
    gap_right = np.zeros(ms.SLOTS, dtype=bool)
    gap_left[1:] = ~finite[:-1]           # a missing reading just before the slot
    gap_right[:-1] = ~finite[1:]          # a missing reading just after the slot
    scan = slice(ms.SCAN_START, ms.SCAN_END)
    return (start_at_cusp | gap_left)[scan], (end_at_cusp | gap_right)[scan]


def frozen_admissibility(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The 48 x 48 admissible-window mask exactly as score_siteday applies it (missing = mask_windows, edges = inwardx)."""
    start_ok, end_ok = frozen_edge_ok(y, s)
    return ms.admissible_windows(y, s) & start_ok[:, None] & end_ok[None, :]


def frozen_score_matrix(y: np.ndarray, s: np.ndarray, floor: float) -> np.ndarray:
    """The unmasked 48 x 48 evidence matrix r_W under the frozen statistic (llr, p_exp = 0).

    Args:
        y, s: 96-slot arrays, MW.
        floor: the cohort sigma floor, MW (the RSS floor lambda_L = L * floor^2).

    Returns:
        r_W per window; -inf where the window cannot be scored (missing interior or no anchor)
        and below the diagonal.
    """
    u0 = ms.reconstruct_uncorrected(y, s)
    rss_u, rss_c, length = ms.bridge_residual_matrices(u0, y, FROZEN["variant"])
    return ms.llr_matrix(rss_u, rss_c, length, floor, FROZEN["p_exp"])


def ranking(sc: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank the candidate windows in the scorer's order: score descending, then shorter, then earlier.

    Args:
        sc: 48 x 48 evidence matrix.
        candidates: boolean 48 x 48 mask of the windows to rank.

    Returns:
        rank: 48 x 48 integer matrix, 1 = best, 0 for windows outside the candidate set;
        ii, jj: the candidate windows' (start, end) indices in rank order.
    """
    ii, jj = np.nonzero(candidates)
    order = np.lexsort((ii, jj - ii, -sc[ii, jj]))   # keys from last to first: score, length, start
    rank = np.zeros(sc.shape, dtype=int)
    rank[ii[order], jj[order]] = np.arange(1, order.size + 1)
    return rank, ii[order], jj[order]


# ----------------------------------------------------------------------------- geometry against the labels

def truth_span(truth: np.ndarray) -> tuple[int, int, int]:
    """(first labelled slot, last labelled slot, number of labelled slots)."""
    idx = np.flatnonzero(truth)
    return int(idx[0]), int(idx[-1]), int(idx.size)


def intersection_matrix(truth: np.ndarray) -> np.ndarray:
    """Number of labelled slots inside every window, via a cumulative count."""
    cum = np.concatenate([[0], np.cumsum(truth.astype(int))])
    a = ms.SCAN_START + _I
    b = ms.SCAN_START + _J
    return np.where(UPPER, cum[b + 1] - cum[a], 0)


def energy_matrices(y: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Candidate and correctly corrected energy of every window, and the required energy, MWh.

    A flipped slot changes the reading by 2|y|, so its correction energy is 2 * |y| * 0.25 MWh;
    missing readings carry none. Candidate = sum over the window; correct = sum over
    window and truth; required = sum over the truth.
    """
    e = 2.0 * np.abs(np.nan_to_num(y, nan=0.0)) * HOURS_PER_SLOT
    cum_e = np.concatenate([[0.0], np.cumsum(e)])
    cum_et = np.concatenate([[0.0], np.cumsum(e * truth)])
    a = ms.SCAN_START + _I
    b = ms.SCAN_START + _J
    candidate = np.where(UPPER, cum_e[b + 1] - cum_e[a], 0.0)
    correct = np.where(UPPER, cum_et[b + 1] - cum_et[a], 0.0)
    return candidate, correct, float((e * truth).sum())


def first_rank_reaching(ii: np.ndarray, jj: np.ndarray, iou: np.ndarray, target: float) -> int:
    """Rank (1-based) of the first window in rank order whose interval IoU reaches the target; 0 if none."""
    hits = np.flatnonzero(iou[ii, jj] >= target)
    return int(hits[0]) + 1 if hits.size else 0


def classify_geometry(a: int, b: int, at: int, bt: int) -> str:
    """How the rank-1 window [a, b] sits against the truth span [at, bt] (slots, inclusive)."""
    if a == at and b == bt:
        return "exact"
    if b < at or a > bt:
        return "disjoint"
    if a <= at and b >= bt:
        if b == bt:
            return "too long (start)"
        if a == at:
            return "too long (end)"
        return "too long (both)"
    if a >= at and b <= bt:
        return "too short"
    return "shifted earlier" if a < at else "shifted later"


def truth_exclusion_reason(y: np.ndarray, s: np.ndarray, at: int, bt: int) -> str:
    """Why the truth span is not an admissible window; '' when it is admissible.

    Reasons, in the order tested: outside the scan range (slots 24-71); a missing reading
    inside the span or no finite anchor beside it; the edge rule at the start, the end or both.
    """
    if at < ms.SCAN_START or bt >= ms.SCAN_END:
        return "outside scan range"
    i, j = at - ms.SCAN_START, bt - ms.SCAN_START
    if not ms.admissible_windows(y, s)[i, j]:
        return "missing reading"
    start_ok, end_ok = frozen_edge_ok(y, s)
    if start_ok[i] and end_ok[j]:
        return ""
    if not start_ok[i] and not end_ok[j]:
        return "edge rule (both)"
    return "edge rule (start)" if not start_ok[i] else "edge rule (end)"


def bin_label(rank: int, exclusion: str) -> str:
    """Rank bin: 1, 2, 3, 4-10, >10, or the exclusion label when the rank is 0."""
    if rank == 0:
        return exclusion
    if rank <= 3:
        return str(rank)
    return "4-10" if rank <= 10 else ">10"


# ----------------------------------------------------------------------------- one site-day

def window_measures(i: int, j: int, inter: np.ndarray, iou: np.ndarray, candidate: np.ndarray, correct: np.ndarray, n_true: int) -> dict:
    """Interval and energy agreement of window (i, j) with the truth: IoU, tp/fp/fn slots, MWh."""
    tp = int(inter[i, j])
    return dict(start=int(ms.SCAN_START + i), end=int(ms.SCAN_START + j), iou=float(iou[i, j]),
                tp=tp, fp=int(LENGTH[i, j]) - tp, fn=n_true - tp,
                candidate_mwh=float(candidate[i, j]), correct_mwh=float(correct[i, j]))


def analyse_day(day: dict, floor: float) -> dict:
    """Score one labelled RPF day label-free, then rank and measure its windows against the truth.

    Args:
        day: record from labelled_rpf_days.
        floor: cohort sigma floor, MW.

    Returns:
        One flat row: population, truth span and contiguity, admissibility of the truth,
        the ranks of the exact span and of the first window with IoU >= 0.8 (among admissible
        and among all scoreable windows; 0 = not ranked, with the reason), the rank-1 window's
        agreement with the truth, its overlap geometry, and the best of the top 3 and top 5.
    """
    y, s, truth = day["y"], day["s"], day["truth"]
    # Label-free part: exactly what the frozen scorer computes.
    adm = frozen_admissibility(y, s)
    sc = frozen_score_matrix(y, s, floor)
    scoreable = UPPER & np.isfinite(sc)
    rank_adm, ii_adm, jj_adm = ranking(sc, adm)
    rank_all, ii_all, jj_all = ranking(sc, scoreable)

    # Labels enter here.
    at, bt, n_true = truth_span(truth)
    inter = intersection_matrix(truth)
    iou = np.where(UPPER, inter / np.maximum(LENGTH + n_true - inter, 1), 0.0)
    candidate, correct, required = energy_matrices(y, truth)
    energy_iou = np.where(UPPER, correct / np.maximum(candidate + required - correct, 1e-12), 0.0)

    in_scan = ms.SCAN_START <= at and bt < ms.SCAN_END
    ti, tj = (at - ms.SCAN_START, bt - ms.SCAN_START) if in_scan else (-1, -1)
    exclusion = truth_exclusion_reason(y, s, at, bt)
    truth_admissible = exclusion == ""
    truth_scoreable = in_scan and bool(scoreable[ti, tj])
    rank_exact_adm = int(rank_adm[ti, tj]) if truth_admissible else 0
    rank_exact_all = int(rank_all[ti, tj]) if truth_scoreable else 0
    rank_iou_adm = first_rank_reaching(ii_adm, jj_adm, iou, IOU_TARGET)
    rank_iou_all = first_rank_reaching(ii_all, jj_all, iou, IOU_TARGET)

    row = dict(cohort=day["cohort"], station=day["station"], date=day["date"], confidence=day["confidence"],
               population=population_name(day["cohort"], day["confidence"]), sigma_floor=floor,
               input_ok=bool(adm.any()), n_admissible=int(adm.sum()), n_scoreable=int(scoreable.sum()),
               true_start=at, true_end=bt, true_slots=n_true, contiguous=(bt - at + 1) == n_true,
               truth_admissible=truth_admissible, truth_exclusion=exclusion, truth_scoreable=truth_scoreable,
               rank_exact_adm=rank_exact_adm, rank_iou_adm=rank_iou_adm,
               rank_exact_all=rank_exact_all, rank_iou_all=rank_iou_all,
               bin_exact_adm=bin_label(rank_exact_adm, NOT_ADMISSIBLE),
               bin_iou_adm=bin_label(rank_iou_adm, NOT_ADMISSIBLE if not adm.any() else NO_WINDOW_REACHES),
               bin_exact_all=bin_label(rank_exact_all, NOT_SCOREABLE),
               bin_iou_all=bin_label(rank_iou_all, NO_WINDOW_REACHES),
               max_iou_adm=float(iou[adm].max()) if adm.any() else np.nan,
               max_iou_all=float(iou[scoreable].max()) if scoreable.any() else np.nan,
               required_mwh=required)
    if not adm.any():
        return row

    # The rank-1 window (the scorer's proposal) and the top-k oracles, admissible ranking.
    # The scorer's own best_window (tie tolerance included) is kept beside it for the self-check.
    i1, j1 = int(ii_adm[0]), int(jj_adm[0])
    scorer_best = ms.best_window(np.where(adm, sc, -np.inf))
    best = window_measures(i1, j1, inter, iou, candidate, correct, n_true)
    row.update({f"rank1_{k}": v for k, v in best.items()})
    row.update(rank1_score=float(sc[i1, j1]), rank1_length=int(LENGTH[i1, j1]),
               scorer_best_start=scorer_best.start, scorer_best_end=scorer_best.end,
               null_wins=bool(sc[i1, j1] <= 0.0 + ms.TIE_TOL),
               geometry=classify_geometry(best["start"], best["end"], at, bt),
               start_offset=best["start"] - at, end_offset=best["end"] - bt)
    for k in TOP_K:
        top_i, top_j = ii_adm[:k], jj_adm[:k]
        pick = int(np.argmax(iou[top_i, top_j]))                       # first window on ties = higher rank
        by_iou = window_measures(int(top_i[pick]), int(top_j[pick]), inter, iou, candidate, correct, n_true)
        pick = int(np.argmax(energy_iou[top_i, top_j]))
        by_energy = window_measures(int(top_i[pick]), int(top_j[pick]), inter, iou, candidate, correct, n_true)
        row.update({f"top{k}_iou_{m}": by_iou[m] for m in ("iou", "tp", "fp", "fn")})
        row.update({f"top{k}_energy_{m}": by_energy[m] for m in ("candidate_mwh", "correct_mwh")})
    return row


# ----------------------------------------------------------------------------- self-check against Phase 3

def self_check(per_day: pd.DataFrame, scores: pd.DataFrame, site_days: pd.DataFrame) -> pd.DataFrame:
    """Compare the recomputed scoring with the committed Phase 3 outputs, per cohort.

    Checks, on every labelled RPF day: the sigma floor; the admissible count; the masked
    argmax against the committed best window (or, when the null won, against the committed
    runner-up, which holds the best window in that case); its score against r_best; the
    ranking's top window against the scorer's own best_window (tie tolerance included); and
    the rank-1 energies against the committed candidate energies on days with a committed window.
    """
    ref = scores.merge(site_days[KEY + ["candidate_mwh", "candidate_correct_mwh", "required_mwh"]], on=KEY, how="left", validate="one_to_one")
    t = per_day.merge(ref, on=KEY, how="left", suffixes=("", "_p3"), validate="one_to_one")
    if t["input_ok_p3"].isna().any():
        raise ValueError("A labelled RPF day is missing from the committed Phase 3 scores.")
    rows = []
    for cohort, g in t.groupby("cohort", sort=True):
        committed_start = np.where(g["best_start"] >= 0, g["best_start"], g["runner_start"])
        committed_end = np.where(g["best_start"] >= 0, g["best_end"], g["runner_end"])
        window_ok = (g["rank1_start"] == committed_start) & (g["rank1_end"] == committed_end)
        null_ok = (g["best_start"] < 0) == g["null_wins"]
        score_diff = (g["rank1_score"] - g["r_best"]).abs()
        has_window = g["best_start"] >= 0
        e = g[has_window]
        e_diff = pd.DataFrame({"candidate": (e["rank1_candidate_mwh"] - e["candidate_mwh"]).abs(),
                               "correct": (e["rank1_correct_mwh"] - e["candidate_correct_mwh"]).abs(),
                               "required": (e["required_mwh"] - e["required_mwh_p3"]).abs()})
        tie_ok = (g["scorer_best_start"] == g["rank1_start"]) & (g["scorer_best_end"] == g["rank1_end"])
        rows.append(dict(
            cohort=cohort, days_checked=len(g), days_input_ok=int(g["input_ok"].sum()),
            sigma_floor_matches=bool(np.allclose(g["sigma_floor"], g["sigma_floor_p3"], rtol=0, atol=1e-18)),
            n_admissible_mismatches=int((g["n_admissible"] != g["n_admissible_p3"]).sum()),
            best_window_mismatches=int((~window_ok).sum()),
            null_decision_mismatches=int((~null_ok).sum()),
            tie_break_mismatches=int((~tie_ok).sum()),
            score_max_abs_diff=float(score_diff.max()),
            days_with_committed_window=int(has_window.sum()),
            energy_days_compared=len(e),
            candidate_mwh_max_abs_diff=float(e_diff["candidate"].max()),
            correct_mwh_max_abs_diff=float(e_diff["correct"].max()),
            required_mwh_max_abs_diff=float(e_diff["required"].max()),
            energy_days_differing=int((e_diff > 1e-6).any(axis=1).sum()),
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- aggregation

def rank_distribution(per_day: pd.DataFrame, by: str) -> pd.DataFrame:
    """Counts and shares of rank bins per group, for both rankings and both targets.

    Args:
        per_day: per-day rows.
        by: grouping column, 'population' or 'station'.

    Returns:
        One row per (group, ranking, target) with n_days and n_<bin>, share_<bin> columns.
    """
    rows = []
    specs = [("admissible", "exact span", "bin_exact_adm"), ("admissible", f"IoU >= {IOU_TARGET}", "bin_iou_adm"),
             ("all windows", "exact span", "bin_exact_all"), ("all windows", f"IoU >= {IOU_TARGET}", "bin_iou_all")]
    for group, g in per_day.groupby(by, sort=True):
        for ranking_name, target, column in specs:
            counts = g[column].value_counts()
            row = {by: group, "ranking": ranking_name, "target": target, "n_days": len(g)}
            for b in ALL_BINS:
                row[f"n_{b}"] = int(counts.get(b, 0))
            for b in ALL_BINS:
                row[f"share_{b}"] = row[f"n_{b}"] / len(g) if len(g) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_quality(g: pd.DataFrame, prefix: str = "rank1") -> dict:
    """Pooled agreement of one chosen window per day with the truth, over days with an admissible window.

    Interval precision, recall and F1 pool the tp/fp/fn slot counts; energy IoU and energy
    precision pool the MWh (summed before dividing, as the locked metrics do).
    """
    d = g[g["input_ok"]]
    tp, fp, fn = (float(d[f"{prefix}_{c}"].sum()) for c in ("tp", "fp", "fn"))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    cand, corr, req = (float(d[c].sum()) for c in (f"{prefix}_candidate_mwh", f"{prefix}_correct_mwh", "required_mwh"))
    return dict(
        n_days=len(g), n_days_with_window=len(d), n_null_wins=int(d["null_wins"].sum()),
        mean_iou=float(d[f"{prefix}_iou"].mean()), median_iou=float(d[f"{prefix}_iou"].median()),
        share_iou_ge_target=float((d[f"{prefix}_iou"] >= IOU_TARGET).mean()),
        interval_precision=prec, interval_recall=rec,
        interval_f1=(2 * prec * rec / (prec + rec)) if (prec and rec and np.isfinite(prec + rec)) else 0.0,
        energy_iou=corr / (cand + req - corr) if cand + req - corr > 0 else np.nan,
        energy_precision=corr / cand if cand > 0 else np.nan,
        candidate_mwh=cand, correct_mwh=corr, required_mwh=req,
    )


def rank1_quality(per_day: pd.DataFrame, by: str) -> pd.DataFrame:
    """Rank-1 window quality per group: exact share, IoU summary, pooled interval and energy metrics."""
    rows = []
    for group, g in per_day.groupby(by, sort=True):
        q = pooled_quality(g)
        q["exact_share"] = float((g["rank_exact_adm"] == 1).mean())
        q["mean_best_admissible_iou"] = float(g["max_iou_adm"].mean())
        rows.append({by: group, **q})
    cols = [by, "n_days", "n_days_with_window", "n_null_wins", "exact_share", "mean_iou", "median_iou",
            "share_iou_ge_target", "mean_best_admissible_iou", "interval_precision", "interval_recall",
            "interval_f1", "energy_iou", "energy_precision", "candidate_mwh", "correct_mwh", "required_mwh"]
    return pd.DataFrame(rows)[cols]


def oracle_table(per_day: pd.DataFrame) -> pd.DataFrame:
    """Best-of-top-k per population: the window among the top k that a reviewer would pick.

    Two selection rules per day: the top-k window with the highest interval IoU (reported
    with pooled interval metrics) and the one with the highest energy IoU (reported with
    pooled energy metrics). k = 1 is the rank-1 window itself.
    """
    rows = []
    for pop, g in per_day.groupby("population", sort=True):
        d = g[g["input_ok"]]
        for k in TOP_K:
            tp, fp, fn = (float(d[f"top{k}_iou_{c}"].sum()) for c in ("tp", "fp", "fn"))
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            cand, corr, req = (float(d[c].sum()) for c in (f"top{k}_energy_candidate_mwh", f"top{k}_energy_correct_mwh", "required_mwh"))
            rows.append(dict(
                population=pop, k=k, n_days_with_window=len(d),
                exact_within_top_k=float(((g["rank_exact_adm"] >= 1) & (g["rank_exact_adm"] <= k)).mean()),
                iou_target_within_top_k=float(((g["rank_iou_adm"] >= 1) & (g["rank_iou_adm"] <= k)).mean()),
                mean_iou=float(d[f"top{k}_iou_iou"].mean()),
                interval_precision=prec, interval_recall=rec, interval_f1=2 * prec * rec / (prec + rec),
                energy_iou=corr / (cand + req - corr), energy_precision=corr / cand,
            ))
    return pd.DataFrame(rows)


def geometry_table(per_day: pd.DataFrame) -> pd.DataFrame:
    """Rank-1 overlap geometry per population, with Alpha split by label contiguity.

    Counts and shares of each category, and the mean slot offsets of the start and end
    (rank-1 minus truth; negative = earlier) over the overlapping non-exact days.
    """
    groups = [(pop, g) for pop, g in per_day.groupby("population", sort=True)]
    alpha = per_day[per_day["population"] == "Alpha"]
    groups += [("Alpha (contiguous)", alpha[alpha["contiguous"]]), ("Alpha (non-contiguous)", alpha[~alpha["contiguous"]])]
    rows = []
    for name, g in groups:
        d = g[g["input_ok"]]
        counts = d["geometry"].value_counts()
        row = {"population": name, "n_days_with_window": len(d)}
        for cat in GEOMETRY:
            row[f"n_{cat}"] = int(counts.get(cat, 0))
        for cat in GEOMETRY:
            row[f"share_{cat}"] = row[f"n_{cat}"] / len(d) if len(d) else np.nan
        off = d[(d["geometry"] != "exact") & (d["geometry"] != "disjoint")]
        row["mean_start_offset_slots"] = float(off["start_offset"].mean()) if len(off) else np.nan
        row["mean_end_offset_slots"] = float(off["end_offset"].mean()) if len(off) else np.nan
        # Edge jitter: share of days whose rank-1 edges sit within one slot of the truth span's.
        row["share_start_within_1_slot"] = float((d["start_offset"].abs() <= 1).mean()) if len(d) else np.nan
        row["share_end_within_1_slot"] = float((d["end_offset"].abs() <= 1).mean()) if len(d) else np.nan
        row["share_both_edges_within_1_slot"] = float(((d["start_offset"].abs() <= 1) & (d["end_offset"].abs() <= 1)).mean()) if len(d) else np.nan
        too_long = d[d["geometry"].str.startswith("too long", na=False)]
        extra_slots = too_long["rank1_length"] - (too_long["true_end"] - too_long["true_start"] + 1)
        row["mean_extra_slots_too_long"] = float(extra_slots.mean()) if len(too_long) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def truth_admissibility_table(per_day: pd.DataFrame) -> pd.DataFrame:
    """Why truth spans are excluded from the admissible ranking, per population."""
    reasons = ["outside scan range", "missing reading", "edge rule (start)", "edge rule (end)", "edge rule (both)"]
    rows = []
    for pop, g in per_day.groupby("population", sort=True):
        counts = g["truth_exclusion"].value_counts()
        row = {"population": pop, "n_days": len(g), "n_admissible": int((g["truth_exclusion"] == "").sum()),
               "n_non_contiguous": int((~g["contiguous"]).sum())}
        for r in reasons:
            row[f"n_{r}"] = int(counts.get(r, 0))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tables and figures

def to_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no external dependency)."""
    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{decimals}f}"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, name: str, decimals: int = 3) -> None:
    """Write tables/<name>.csv and tables/<name>.md."""
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(to_markdown(df, decimals), encoding="utf-8")


def compact_distribution(dist: pd.DataFrame, by: str, ranking_name: str) -> pd.DataFrame:
    """Readable form of a rank distribution for one ranking: 'count (share)' per bin."""
    bins = RANK_BINS + ((NOT_ADMISSIBLE,) if ranking_name == "admissible" else (NOT_SCOREABLE,)) + (NO_WINDOW_REACHES,)
    d = dist[dist["ranking"] == ranking_name]
    out = d[[by, "target", "n_days"]].copy()
    for b in bins:
        out[b] = [f"{n} ({s:.1%})" for n, s in zip(d[f"n_{b}"], d[f"share_{b}"], strict=True)]
    return out


def apply_journal_style() -> None:
    """Arial, white background, dark-blue frame, no titles; matches final_eval/figures.py."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": COLORS["dark_blue"],
        "axes.labelcolor": COLORS["dark_blue"], "axes.axisbelow": True, "axes.grid": False, "axes.linewidth": 1.0,
        "font.size": 11, "axes.labelsize": 11, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.color": COLORS["dark_blue"], "ytick.color": COLORS["dark_blue"], "text.color": COLORS["dark_blue"],
        "savefig.facecolor": "white", "legend.frameon": False, "savefig.dpi": 200,
    })


def style_axis(axis) -> None:
    """Light horizontal grid behind the marks."""
    axis.grid(True, axis="y", color=COLORS["light_white"], linewidth=0.9, zorder=0)
    axis.set_axisbelow(True)


def figure_rank_distribution(dist: pd.DataFrame, per_day: pd.DataFrame) -> None:
    """Grouped bars of the rank bins (admissible ranking) per population: exact span and IoU >= 0.8."""
    bins = RANK_BINS + (NOT_ADMISSIBLE, NO_WINDOW_REACHES)
    labels = ["1", "2", "3", "4–10", ">10", "not\nadm.", f"none\n≥ {IOU_TARGET}"]
    fig, axes = plt.subplots(1, len(POPULATIONS), figsize=(13, 3.9), sharey=True)
    d = dist[dist["ranking"] == "admissible"]
    x = np.arange(len(bins))
    width = 0.38
    for axis, pop in zip(axes, POPULATIONS, strict=True):
        n = int(per_day["population"].eq(pop).sum())
        for offset, target, color in ((-width / 2, "exact span", COLORS["dark_blue"]), (width / 2, f"IoU >= {IOU_TARGET}", COLORS["orange"])):
            row = d[(d["population"] == pop) & (d["target"] == target)].iloc[0]
            shares = np.array([row[f"share_{b}"] for b in bins]) * 100
            bars = axis.bar(x + offset, shares, width, color=color, zorder=2,
                            label=target.replace(">=", "≥"))
            for bar, share in zip(bars, shares, strict=True):
                if share >= 0.5:
                    axis.text(bar.get_x() + bar.get_width() / 2, share + 1, f"{share:.0f}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_xlabel(f"{pop} (n = {n:,} RPF days)")
        style_axis(axis)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Share of days (%)")
    axes[0].legend(loc="upper right")
    fig.supxlabel("Rank of the truth window among admissible windows (not adm. = truth span inadmissible; none = no admissible window reaches the IoU target)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_rank_distribution.png")
    plt.close(fig)


def figure_rank1_iou_cdf(per_day: pd.DataFrame) -> None:
    """Cumulative distribution of the rank-1 window's interval IoU per population."""
    fig, axis = plt.subplots(figsize=(6, 4))
    for pop in POPULATIONS:
        v = np.sort(per_day.loc[per_day["input_ok"] & (per_day["population"] == pop), "rank1_iou"].to_numpy(float))
        axis.step(np.concatenate([[0.0], v]), np.concatenate([[0.0], np.arange(1, v.size + 1) / v.size]), where="post",
                  color=POP_COLORS[pop], linestyle=POP_STYLES[pop], linewidth=2, label=f"{pop} (n = {v.size:,})", zorder=2)
    axis.axvline(IOU_TARGET, color=COLORS["light_white"], linewidth=1.2, zorder=1)
    axis.text(IOU_TARGET - 0.01, 0.03, f"IoU = {IOU_TARGET}", ha="right", va="bottom", fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Interval IoU of the rank-1 window against the labelled slots")
    axis.set_ylabel("Cumulative share of RPF days")
    axis.legend(loc="upper left")
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_rank1_iou_cdf.png")
    plt.close(fig)


def figure_station_exact(station_quality: pd.DataFrame) -> None:
    """Horizontal bars of the exact rank-1 share per station on the headline populations."""
    d = station_quality[station_quality["n_days"] > 0].sort_values(["population", "station"]).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(6.5, 0.32 * len(d) + 1.2))
    y = np.arange(len(d))[::-1]
    colors = [POP_COLORS[p] for p in d["population"]]
    axis.barh(y, d["exact_share"] * 100, color=colors, height=0.7, zorder=2)
    for yi, share, n in zip(y, d["exact_share"] * 100, d["n_days"], strict=True):
        axis.text(share + 1, yi, f"{share:.0f}% (n = {n:,})", va="center", fontsize=8)
    axis.set_yticks(y, d["station"])
    axis.set_xlim(0, 100)
    axis.set_xlabel("RPF days on which the rank-1 window equals the truth span (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=POP_COLORS[p]) for p in HEADLINE]
    axis.legend(handles, HEADLINE, loc="lower right")
    axis.grid(True, axis="x", color=COLORS["light_white"], linewidth=0.9, zorder=0)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_station_exact_rank1.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def run() -> None:
    """Run the whole study end to end and print the headline tables."""
    t0 = time.time()
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    intervals, floors, scores, site_days = load_inputs()
    print(f"loaded in {time.time() - t0:.1f} s; sigma floors {floors}")

    rows = [analyse_day(day, floors[day["cohort"]]) for day in labelled_rpf_days(intervals)]
    per_day = pd.DataFrame(rows)
    print(f"scored {len(per_day)} labelled RPF days in {time.time() - t0:.1f} s")

    check = self_check(per_day, scores, site_days)
    write_table(check, "self_check", decimals=9)
    print("\nSelf-check against Phase 3\n" + check.to_string(index=False))
    if check[["n_admissible_mismatches", "best_window_mismatches", "null_decision_mismatches", "tie_break_mismatches"]].to_numpy().any():
        raise RuntimeError("Recomputed scoring does not reproduce the committed Phase 3 best windows; see tables/self_check.csv.")

    per_day.to_csv(TABLES / "per_day.csv", index=False)
    headline = per_day[per_day["population"].isin(HEADLINE)]

    dist_pop = rank_distribution(per_day, "population")
    dist_station = rank_distribution(headline, "station")
    write_table(dist_pop, "rank_distribution_population")
    write_table(dist_station, "rank_distribution_station")
    for ranking_name, tag in (("admissible", "admissible"), ("all windows", "all_windows")):
        (TABLES / f"rank_distribution_population_{tag}.md").write_text(to_markdown(compact_distribution(dist_pop, "population", ranking_name)), encoding="utf-8")
        (TABLES / f"rank_distribution_station_{tag}.md").write_text(to_markdown(compact_distribution(dist_station, "station", ranking_name)), encoding="utf-8")

    # Alpha split by label contiguity: the outer span of a non-contiguous label is not a window
    # any single-window method can match, so the two halves are reported side by side.
    alpha = per_day[per_day["population"] == "Alpha"].assign(
        contiguity=lambda d: np.where(d["contiguous"], "Alpha (contiguous)", "Alpha (non-contiguous)"))
    dist_contiguity = rank_distribution(alpha, "contiguity")
    write_table(dist_contiguity, "rank_distribution_alpha_contiguity")
    (TABLES / "rank_distribution_alpha_contiguity_admissible.md").write_text(to_markdown(compact_distribution(dist_contiguity, "contiguity", "admissible")), encoding="utf-8")
    write_table(rank1_quality(alpha, "contiguity"), "rank1_quality_alpha_contiguity")

    quality_pop = rank1_quality(per_day, "population")
    quality_station = rank1_quality(headline, "station")
    quality_station.insert(0, "population", quality_station["station"].map(lambda s: "Alpha" if s.startswith("alpha") else "Beta sure"))
    write_table(quality_pop, "rank1_quality_population")
    write_table(quality_station, "rank1_quality_station")

    oracles = oracle_table(per_day)
    geometry = geometry_table(per_day)
    admissibility = truth_admissibility_table(per_day)
    write_table(oracles, "oracle_top_k")
    write_table(geometry, "failure_geometry")
    write_table(admissibility, "truth_admissibility")

    apply_journal_style()
    figure_rank_distribution(dist_pop, per_day)
    figure_rank1_iou_cdf(per_day)
    figure_station_exact(quality_station)

    pd.set_option("display.width", 250)
    for ranking_name in ("admissible", "all windows"):
        print(f"\nRank distribution, {ranking_name} ranking\n" + compact_distribution(dist_pop, "population", ranking_name).to_string(index=False))
    print("\nRank distribution, admissible ranking, Alpha by contiguity\n" + compact_distribution(dist_contiguity, "contiguity", "admissible").to_string(index=False))
    print("\nRank-1 quality\n" + quality_pop.round(3).to_string(index=False))
    print("\nOracles\n" + oracles.round(3).to_string(index=False))
    print("\nGeometry\n" + geometry.round(3).to_string(index=False))
    print("\nTruth admissibility\n" + admissibility.to_string(index=False))
    print(f"\ndone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    run()
