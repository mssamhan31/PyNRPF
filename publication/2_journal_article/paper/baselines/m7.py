"""M7, the deterministic threshold rule, scored on every held-out station.

The rule looks for the solar-peak window of each day, finds the pair of net-load minima
on either side of a local maximum inside that window, and flags the day when a minimum
falls below a fraction of the day's maximum. Two paths run on every day: the strict
path with the threshold gates gives the day flag; the relaxed path without them gives
the interval correction, the slots strictly between the two minima. The two can
disagree on the same day, exactly as in the conference codebase.

Inputs:  the complete-day interval frame of a cohort and the folds; the ``m7`` block
         of the settings (solar_peak_tiebreak_time, peak_window_minutes, min_threshold,
         min_threshold_both). Nothing is fitted.
Outputs: the interval prediction table for M7 in the common schema (``data.INTERVAL_COLUMNS``).
Key steps: build the frame in the configuration's column names for one station; apply
         the rule; carry the flags back positionally (the rule never reorders rows).
"""

from __future__ import annotations

from datetime import time as dt_time
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..data import INTERVAL_COLUMNS, finish_interval_table, model_input
from ..folds import Fold
from . import check_frame

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_MIN = 60_000_000_000
_NS_PER_SEC = 1_000_000_000
_DAY_START_NS = 6 * _NS_PER_HOUR
_DAY_END_NS = 18 * _NS_PER_HOUR

PairResult = tuple[int, int, float, float] | None


def _parse_time(text: str) -> dt_time:
    """Parse ``HH:MM`` or ``HH:MM:SS`` into ``datetime.time``."""
    parts = text.strip().split(":")
    return dt_time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)


def _pick_max(vals: np.ndarray, ts_i64: np.ndarray, ref_i64: int) -> int | None:
    """Index of the maximum value; ties go to the reading nearest ``ref_i64``, then the earliest."""
    if len(vals) == 0:
        return None
    dist = np.abs(ts_i64 - ref_i64)
    order = np.lexsort((ts_i64, dist, -vals))
    return int(order[0])


def _best_candidate_pair(ts_g: np.ndarray, mw_g: np.ndarray, ts_m: np.ndarray, mw_m: np.ndarray,
                         cand: np.ndarray, threshold_both: float | None) -> PairResult:
    """The left and right minima pair with the smallest sum over the candidate peaks."""
    best_sum = np.inf
    best_peak = -np.inf
    best_pair: PairResult = None
    for c in cand:
        pk_ts = int(ts_m[c])
        pk_mw = float(mw_m[c])
        lm_mask = ts_g < pk_ts
        rm_mask = ts_g > pk_ts
        if threshold_both is not None:
            lm_mask &= mw_g < threshold_both
            rm_mask &= mw_g < threshold_both
        if not lm_mask.any() or not rm_mask.any():
            continue
        li = np.where(lm_mask)[0]
        ri = np.where(rm_mask)[0]
        lx = int(np.argmin(mw_g[li]))
        rx = int(np.argmin(mw_g[ri]))
        l_mw = float(mw_g[li[lx]])
        l_ts = int(ts_g[li[lx]])
        r_mw = float(mw_g[ri[rx]])
        r_ts = int(ts_g[ri[rx]])
        pair_sum = l_mw + r_mw
        if pair_sum < best_sum or (pair_sum == best_sum and pk_mw > best_peak):
            best_sum = pair_sum
            best_peak = pk_mw
            best_pair = (l_ts, r_ts, l_mw, r_mw)
    return best_pair


def _pair_is_daytime(left_ts: int, right_ts: int, midnight: int) -> bool:
    """Both minima lie between 06:00 and 18:00."""
    left_delta = left_ts - midnight
    right_delta = right_ts - midnight
    return _DAY_START_NS <= left_delta <= _DAY_END_NS and _DAY_START_NS <= right_delta <= _DAY_END_NS


def apply_rule(frame: pd.DataFrame, thresholds: dict[str, Any], columns: dict[str, str]) -> pd.DataFrame:
    """Run the M7 rule on an interval frame.

    Args:
        frame: interval rows, tz-naive timestamps, net load and solar in MW.
        thresholds: the ``m7.m7_threshold`` block: ``solar_peak_tiebreak_time`` (HH:MM),
            ``peak_window_minutes``, ``min_threshold`` and ``min_threshold_both``
            (fractions of the day's maximum net load).
        columns: the configuration's ``columns`` block.

    Returns:
        A copy of ``frame`` in the same row order with ``m7_rpf_flag`` (the relaxed
        interval correction) and ``m7_rpf_day`` (the strict day flag) added.
    """
    col_site, col_ts = columns["site"], columns["timestamp"]
    col_net, col_solar = columns["net_load"], columns["solar"]
    tiebreak_time = _parse_time(str(thresholds["solar_peak_tiebreak_time"]))
    window_minutes = int(thresholds["peak_window_minutes"])
    win_ns = int(window_minutes * 60 * _NS_PER_SEC)
    min_threshold = float(thresholds["min_threshold"])
    min_threshold_both = float(thresholds["min_threshold_both"])

    df = frame.copy()
    if "date" not in df.columns:
        df["date"] = df[col_ts].dt.date
    ts_i64_all = df[col_ts].values.astype("datetime64[ns]").astype(np.int64)
    mw_all = df[col_net].values.astype(np.float64)
    solar_all = df[col_solar].values.astype(np.float64)
    interval_flag_arr = np.zeros(len(df), dtype=bool)
    day_flag_arr = np.zeros(len(df), dtype=bool)
    tb_offset = tiebreak_time.hour * _NS_PER_HOUR + tiebreak_time.minute * _NS_PER_MIN

    for (_, _date), grp in df.groupby([col_site, "date"], sort=False):
        positions = df.index.get_indexer(grp.index)
        ts_g = ts_i64_all[positions]
        mw_g = mw_all[positions]
        solar_g = solar_all[positions]
        order = np.argsort(ts_g)
        ts_g, mw_g, solar_g = ts_g[order], mw_g[order], solar_g[order]
        positions = positions[order]
        midnight = int(np.datetime64(str(_date), "ns"))

        # A day with a missing or negative reading is left alone.
        if np.any(np.isnan(mw_g)) or np.any(mw_g < 0):
            continue
        secs = (ts_g - midnight) / _NS_PER_SEC
        midday = (secs >= 21_600) & (secs < 64_800)
        if midday.sum() < 3:
            continue
        max_mw = float(np.nanmax(mw_g))
        th = max_mw * min_threshold
        th_both = max_mw * min_threshold_both

        mi = np.where(midday)[0]
        ts_m, mw_m, sol_m = ts_g[mi], mw_g[mi], solar_g[mi]
        n_m = len(mw_m)

        # The solar-peak window: around the solar maximum when solar is known, 10:00 to 15:00 otherwise.
        sol_ok = ~np.isnan(sol_m)
        tb_i64 = midnight + tb_offset
        if sol_ok.any():
            si = np.where(sol_ok)[0]
            sp = _pick_max(sol_m[si], ts_m[si], tb_i64)
            if sp is None:
                continue
            solar_ts = int(ts_m[si][sp])
            wl, wh = solar_ts - win_ns, solar_ts + win_ns
        else:
            wl = midnight + 10 * _NS_PER_HOUR
            wh = midnight + 15 * _NS_PER_HOUR - 1

        lmax = np.zeros(n_m, dtype=bool)
        if n_m >= 3:
            lmax[1:-1] = (mw_m[1:-1] > mw_m[:-2]) & (mw_m[1:-1] > mw_m[2:])
        in_win = (ts_m >= wl) & (ts_m <= wh)
        cand = np.where(in_win & lmax)[0]
        if len(cand) == 0:
            continue

        # Strict path: the day flag needs a minima pair under the 'both' gate, one minimum under the gate, daytime.
        strict_pair = _best_candidate_pair(ts_g, mw_g, ts_m, mw_m, cand, th_both)
        if strict_pair is not None:
            strict_lt, strict_rt, strict_lm, strict_rm = strict_pair
            if (strict_lm < th or strict_rm < th) and _pair_is_daytime(strict_lt, strict_rt, midnight):
                day_flag_arr[positions] = True

        # Relaxed path: the interval correction is the span strictly between the ungated minima pair.
        relaxed_pair = _best_candidate_pair(ts_g, mw_g, ts_m, mw_m, cand, None)
        if relaxed_pair is None:
            continue
        relaxed_lt, relaxed_rt, _, _ = relaxed_pair
        if not _pair_is_daytime(relaxed_lt, relaxed_rt, midnight):
            continue
        interval_mask = (ts_g > relaxed_lt) & (ts_g < relaxed_rt)
        if interval_mask.any():
            interval_flag_arr[positions[interval_mask]] = True

    df["m7_rpf_flag"] = interval_flag_arr
    df["m7_rpf_day"] = day_flag_arr
    return df


def predict_station(intervals: pd.DataFrame, fold: Fold, settings: Settings) -> pd.DataFrame:
    """M7 interval predictions for one held-out station, in the common schema."""
    station = intervals[intervals["station"] == fold.held_out]
    frame = model_input(station, settings)
    check_frame(frame, settings.columns)
    flags = apply_rule(frame, settings["m7"]["m7_threshold"], settings.columns)
    table = station[["cohort", "station", "date", "slot", "ts"]].copy()
    table["pred_interval"] = flags["m7_rpf_flag"].to_numpy(bool)
    table["pred_day"] = flags["m7_rpf_day"].to_numpy(bool)
    table["prob_day"] = np.nan
    # The rule has no graded score: a flagged slot carries confidence 1, any other 0.
    table["prob_interval"] = np.where(flags["m7_rpf_flag"].to_numpy(bool), 1.0, 0.0)
    return finish_interval_table(table, "m7", fold.fold_id)


def predict_cohort(intervals: pd.DataFrame, folds: list[Fold], settings: Settings) -> pd.DataFrame:
    """M7 interval predictions for every held-out station of one cohort."""
    cohort = intervals["cohort"].iloc[0]
    parts = [predict_station(intervals, f, settings) for f in folds if f.cohort == cohort]
    return pd.concat(parts, ignore_index=True)[INTERVAL_COLUMNS]
