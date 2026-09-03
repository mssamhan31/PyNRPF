"""m7_threshold -- Deterministic Threshold Rule (DTR) for reverse power flow (RPF) detection.

Live package behaviour keeps the original threshold-based day detection, while
using a relaxed, threshold-free interval correction path. As a result, strict
day flags and interval corrections can diverge on the same site-day.

Inputs:  an interval-level frame of net load and solar in megawatts, and the
         ``m7_threshold`` configuration block.
Outputs: the frame with strict day flags and relaxed interval corrections.
Key steps: find the solar peak window per site-day, apply the threshold gates,
         then correct the minima span within each flagged day.
"""

from __future__ import annotations

from datetime import time as dt_time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .io import req

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_MIN = 60_000_000_000
_NS_PER_SEC = 1_000_000_000
_DAY_START_NS = 6 * _NS_PER_HOUR
_DAY_END_NS = 18 * _NS_PER_HOUR

PairResult = Optional[tuple[int, int, float, float]]


def _parse_time(s: str) -> dt_time:
    """Parse ``HH:MM`` or ``HH:MM:SS`` into ``datetime.time``."""
    parts = s.strip().split(":")
    return dt_time(
        int(parts[0]),
        int(parts[1]),
        int(parts[2]) if len(parts) > 2 else 0,
    )


def _pick_max(
    vals: np.ndarray,
    ts_i64: np.ndarray,
    ref_i64: int,
) -> Optional[int]:
    """Index of maximum value. Tie-break: nearest ``ref_i64``, then earliest."""
    if len(vals) == 0:
        return None
    dist = np.abs(ts_i64 - ref_i64)
    order = np.lexsort((ts_i64, dist, -vals))
    return int(order[0])


def _best_candidate_pair(
    ts_g: np.ndarray,
    mw_g: np.ndarray,
    ts_m: np.ndarray,
    mw_m: np.ndarray,
    cand: np.ndarray,
    threshold_both: float | None,
) -> PairResult:
    """Return the best left/right minima pair for the candidate peaks."""
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
    """Return whether both minima timestamps are within 06:00 to 18:00."""
    left_delta = left_ts - midnight
    right_delta = right_ts - midnight
    return _DAY_START_NS <= left_delta <= _DAY_END_NS and _DAY_START_NS <= right_delta <= _DAY_END_NS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_m7(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    col_site: str,
    col_ts: str,
    col_net: str,
    col_solar: str,
) -> pd.DataFrame:
    """Run the m7 deterministic threshold (DTR) method.

    Adds three columns:
    - ``m7_rpf_flag``: interval corrections from the relaxed interval path.
    - ``net_load_MW_m7``: sign-corrected net load for relaxed interval flags.
    - ``m7_rpf_day``: strict day detection result from the original threshold path.
    """
    import time as _time

    t0 = _time.perf_counter()

    m7 = req(cfg, "m7_threshold")
    tiebreak_time = _parse_time(str(m7["solar_peak_tiebreak_time"]))
    window_minutes = int(m7["peak_window_minutes"])
    win_ns = int(window_minutes * 60 * _NS_PER_SEC)
    min_threshold = float(req(m7, "min_threshold"))
    min_threshold_both = float(req(m7, "min_threshold_both"))

    df = df.copy()
    if "date" not in df.columns:
        df["date"] = df[col_ts].dt.date

    ts_i64_all = df[col_ts].values.astype("datetime64[ns]").astype(np.int64)
    mw_all = df[col_net].values.astype(np.float64)
    solar_all = df[col_solar].values.astype(np.float64)

    interval_flag_arr = np.zeros(len(df), dtype=bool)
    day_flag_arr = np.zeros(len(df), dtype=bool)

    n_total = n_miss = n_neg = n_mid_fail = 0
    n_no_cand = 0
    n_strict_no_pair = n_strict_thr_fail = n_strict_day_fail = 0
    n_relaxed_no_pair = n_relaxed_day_fail = 0
    n_strict_days = n_relaxed_days = n_relaxed_only_days = 0

    tb_offset = tiebreak_time.hour * _NS_PER_HOUR + tiebreak_time.minute * _NS_PER_MIN

    for (_, _date), grp in df.groupby([col_site, "date"], sort=False):
        n_total += 1
        positions = df.index.get_indexer(grp.index)
        ts_g = ts_i64_all[positions]
        mw_g = mw_all[positions]
        solar_g = solar_all[positions]

        order = np.argsort(ts_g)
        ts_g, mw_g, solar_g = ts_g[order], mw_g[order], solar_g[order]
        positions = positions[order]

        midnight = int(np.datetime64(str(_date), "ns"))

        if np.any(np.isnan(mw_g)):
            n_miss += 1
            continue

        if np.any(mw_g < 0):
            n_neg += 1
            continue

        secs = (ts_g - midnight) / _NS_PER_SEC
        midday = (secs >= 21_600) & (secs < 64_800)
        if midday.sum() < 3:
            n_mid_fail += 1
            continue

        max_mw = float(np.nanmax(mw_g))
        th = max_mw * min_threshold
        th_both = max_mw * min_threshold_both

        mi = np.where(midday)[0]
        ts_m, mw_m, sol_m = ts_g[mi], mw_g[mi], solar_g[mi]
        n_m = len(mw_m)

        sol_ok = ~np.isnan(sol_m)
        tb_i64 = midnight + tb_offset
        if sol_ok.any():
            si = np.where(sol_ok)[0]
            sp = _pick_max(sol_m[si], ts_m[si], tb_i64)
            if sp is None:
                n_no_cand += 1
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
            n_no_cand += 1
            continue

        strict_day_positive = False
        strict_pair = _best_candidate_pair(ts_g, mw_g, ts_m, mw_m, cand, th_both)
        if strict_pair is None:
            n_strict_no_pair += 1
        else:
            strict_lt, strict_rt, strict_lm, strict_rm = strict_pair
            if not (strict_lm < th or strict_rm < th):
                n_strict_thr_fail += 1
            elif not _pair_is_daytime(strict_lt, strict_rt, midnight):
                n_strict_day_fail += 1
            else:
                strict_day_positive = True
                day_flag_arr[positions] = True
                n_strict_days += 1

        relaxed_pair = _best_candidate_pair(ts_g, mw_g, ts_m, mw_m, cand, None)
        if relaxed_pair is None:
            n_relaxed_no_pair += 1
            continue

        relaxed_lt, relaxed_rt, _, _ = relaxed_pair
        if not _pair_is_daytime(relaxed_lt, relaxed_rt, midnight):
            n_relaxed_day_fail += 1
            continue

        interval_mask = (ts_g > relaxed_lt) & (ts_g < relaxed_rt)
        if not interval_mask.any():
            continue

        interval_flag_arr[positions[interval_mask]] = True
        n_relaxed_days += 1
        if not strict_day_positive:
            n_relaxed_only_days += 1

    df["m7_rpf_flag"] = interval_flag_arr
    df["net_load_MW_m7"] = np.where(interval_flag_arr, -df[col_net].values, df[col_net].values)
    if df[col_net].isna().any():
        df.loc[df[col_net].isna(), "net_load_MW_m7"] = np.nan

    df["m7_rpf_day"] = day_flag_arr

    elapsed = _time.perf_counter() - t0
    print(f"\nm7_threshold complete ({elapsed:.1f}s):")
    print(f"  Site-days total:              {n_total:,}")
    print(f"  Skipped (missing data):       {n_miss:,}")
    print(f"  Skipped (negative MW):        {n_neg:,}")
    print(f"  Skipped (midday < 3 pts):     {n_mid_fail:,}")
    print(f"  Skipped (no candidates):      {n_no_cand:,}")
    print(f"  Strict skip (no valid pair):  {n_strict_no_pair:,}")
    print(f"  Strict skip (threshold gate): {n_strict_thr_fail:,}")
    print(f"  Strict skip (daytime gate):   {n_strict_day_fail:,}")
    print(f"  Relaxed skip (no valid pair): {n_relaxed_no_pair:,}")
    print(f"  Relaxed skip (daytime gate):  {n_relaxed_day_fail:,}")
    print(f"  Strict days flagged:          {n_strict_days:,}")
    print(f"  Relaxed interval days:        {n_relaxed_days:,}")
    print(f"  Relaxed-only days:            {n_relaxed_only_days:,}")
    print(f"  Intervals flagged:            {int(interval_flag_arr.sum()):,}")
    print(
        "  Strict thresholds: "
        f"min={min_threshold:.2%}, both={min_threshold_both:.2%}"
    )

    return df
