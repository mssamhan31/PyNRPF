from __future__ import annotations

"""m7_threshold -- Deterministic Threshold Rule (DTR) for RPF detection.

Faithfully mirrors the Databricks PySpark implementation logic.

Per (site, date):
1. **Gates** (early-exit):
   - missing_data : skip if any MW value is NaN.
   - has_negative_any : skip if any MW < 0  (already reversed).
   - midday_ok : require >= 3 data-points with hour in [6, 18).
2. Per-day thresholds:
   - max_MW = max(MW) over *all* hours of the day.
   - MW_min_th      = max_MW * min_threshold       (stricter, 5 %).
   - MW_min_th_both = max_MW * min_threshold_both   (looser, 25 %).
3. Solar peak from midday data (hour in [6, 18)).
4. Candidate MW peaks: **strict** 3-point local maxima (>, boundaries excluded)
   computed on *all* midday data, then intersected with the solar window.
5. Left / right minima searched from **all hours** of the day, pre-filtered
   to MW < MW_min_th_both.  Lowest MW wins.
6. Best candidate: both minima must exist.  Rank by lowest raw sum
   (left_mw + right_mw), then highest peak MW.
7. Day-decision gates:
   - either_minima_is_below_th : at least one min < MW_min_th.
   - left_right_ts_in_day_time : both timestamps in [06:00, 18:00].
8. Flag intervals **strictly between** left_min_ts and right_min_ts (exclusive).
"""

from datetime import time as dt_time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.io import req

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_MIN = 60_000_000_000
_NS_PER_SEC = 1_000_000_000


def _parse_time(s: str) -> dt_time:
    """Parse ``"HH:MM"`` or ``"HH:MM:SS"`` into a ``datetime.time``."""
    parts = s.strip().split(":")
    return dt_time(int(parts[0]), int(parts[1]),
                   int(parts[2]) if len(parts) > 2 else 0)


def _pick_max(
    vals: np.ndarray,
    ts_i64: np.ndarray,
    ref_i64: int,
) -> Optional[int]:
    """Index of maximum value.  Tie-break: nearest *ref_i64*, then earliest."""
    if len(vals) == 0:
        return None
    dist = np.abs(ts_i64 - ref_i64)
    order = np.lexsort((ts_i64, dist, -vals))
    return int(order[0])


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
    - ``m7_rpf_flag``   : bool -- True for intervals classified as RPF.
    - ``net_load_MW_m7``: float -- sign-corrected net load (negated where flagged).
    - ``m7_rpf_day``    : bool -- True if any interval in that (site, date) is flagged.
    """
    import time as _time
    t0 = _time.perf_counter()

    # ---- Config ----
    m7 = req(cfg, "m7_threshold")
    tiebreak_time = _parse_time(str(m7["solar_peak_tiebreak_time"]))
    window_minutes = int(m7["peak_window_minutes"])
    win_ns = int(window_minutes * 60 * _NS_PER_SEC)
    min_threshold = float(req(m7, "min_threshold"))
    min_threshold_both = float(req(m7, "min_threshold_both"))

    df = df.copy()
    if "date" not in df.columns:
        df["date"] = df[col_ts].dt.date

    # ---- Pre-extract arrays ----
    ts_i64_all = df[col_ts].values.astype("datetime64[ns]").astype(np.int64)
    mw_all = df[col_net].values.astype(np.float64)
    solar_all = df[col_solar].values.astype(np.float64)

    flag_arr = np.zeros(len(df), dtype=bool)

    # ---- Counters ----
    n_total = n_miss = n_neg = n_mid_fail = 0
    n_no_cand = n_no_pair = n_thr_fail = n_day_fail = n_rpf = 0

    tb_offset = (tiebreak_time.hour * _NS_PER_HOUR
                 + tiebreak_time.minute * _NS_PER_MIN)

    # ---- Iterate over ALL (site, date) groups ----
    for (site, _date), grp in df.groupby([col_site, "date"], sort=False):
        n_total += 1
        positions = df.index.get_indexer(grp.index)
        ts_g = ts_i64_all[positions]
        mw_g = mw_all[positions]
        solar_g = solar_all[positions]

        # Sort by timestamp
        order = np.argsort(ts_g)
        ts_g, mw_g, solar_g = ts_g[order], mw_g[order], solar_g[order]
        positions = positions[order]

        midnight = int(np.datetime64(str(_date), "ns"))

        # ── Gate 1: missing_data ──
        if np.any(np.isnan(mw_g)):
            n_miss += 1
            continue

        # ── Gate 2: has_negative_any ──
        if np.any(mw_g < 0):
            n_neg += 1
            continue

        # ── Midday mask: hour ∈ [6, 18) ──
        secs = (ts_g - midnight) / _NS_PER_SEC
        midday = (secs >= 21_600) & (secs < 64_800)  # 6 h .. 18 h

        # ── Gate 3: midday_ok (>= 3 points) ──
        if midday.sum() < 3:
            n_mid_fail += 1
            continue

        # ── Per-day thresholds ──
        max_mw = float(np.nanmax(mw_g))
        th = max_mw * min_threshold
        th_both = max_mw * min_threshold_both

        # ── Midday arrays ──
        mi = np.where(midday)[0]
        ts_m, mw_m, sol_m = ts_g[mi], mw_g[mi], solar_g[mi]
        n_m = len(mw_m)

        # ── Solar peak (midday only) ──
        sol_ok = ~np.isnan(sol_m)
        has_solar = sol_ok.any()
        tb_i64 = midnight + tb_offset

        if has_solar:
            si = np.where(sol_ok)[0]
            sp = _pick_max(sol_m[si], ts_m[si], tb_i64)
            if sp is None:
                n_no_cand += 1
                continue
            solar_ts = int(ts_m[si][sp])
            wl, wh = solar_ts - win_ns, solar_ts + win_ns
        else:
            # Fallback window: hours 10-14
            wl = midnight + 10 * _NS_PER_HOUR
            wh = midnight + 15 * _NS_PER_HOUR - 1

        # ── Strict local maxima on ALL midday data ──
        lmax = np.zeros(n_m, dtype=bool)
        if n_m >= 3:
            lmax[1:-1] = ((mw_m[1:-1] > mw_m[:-2]) &
                          (mw_m[1:-1] > mw_m[2:]))

        # ── Candidates: in window AND local max ──
        in_win = (ts_m >= wl) & (ts_m <= wh)
        cand = np.where(in_win & lmax)[0]

        if len(cand) == 0:
            n_no_cand += 1
            continue

        # ── Best couple among candidates ──
        best_sum: float = np.inf
        best_peak: float = -np.inf
        best_lt = best_rt = best_lm = best_rm = None

        for c in cand:
            pk_ts = int(ts_m[c])
            pk_mw = float(mw_m[c])

            # Left min: TS < peak, MW < th_both  (ALL hours)
            lm_mask = (ts_g < pk_ts) & (mw_g < th_both)
            if not lm_mask.any():
                continue
            li = np.where(lm_mask)[0]
            lx = int(np.argmin(mw_g[li]))
            l_mw, l_ts = float(mw_g[li[lx]]), int(ts_g[li[lx]])

            # Right min: TS > peak, MW < th_both  (ALL hours)
            rm_mask = (ts_g > pk_ts) & (mw_g < th_both)
            if not rm_mask.any():
                continue
            ri = np.where(rm_mask)[0]
            rx = int(np.argmin(mw_g[ri]))
            r_mw, r_ts = float(mw_g[ri[rx]]), int(ts_g[ri[rx]])

            s = l_mw + r_mw  # raw sum (not abs)
            if s < best_sum or (s == best_sum and pk_mw > best_peak):
                best_sum, best_peak = s, pk_mw
                best_lt, best_lm = l_ts, l_mw
                best_rt, best_rm = r_ts, r_mw

        if best_lt is None:
            n_no_pair += 1
            continue

        # ── either_minima_is_below_th ──
        if not (best_lm < th or best_rm < th):
            n_thr_fail += 1
            continue

        # ── left_right_ts_in_day_time [06:00:00, 18:00:00] ──
        ls = (best_lt - midnight) / _NS_PER_SEC
        rs = (best_rt - midnight) / _NS_PER_SEC
        if not (21_600 <= ls <= 64_800 and 21_600 <= rs <= 64_800):
            n_day_fail += 1
            continue

        # ── Flag: exclusive boundaries (TS > left, TS < right) ──
        n_rpf += 1
        m = (ts_g > best_lt) & (ts_g < best_rt)
        flag_arr[positions[m]] = True

    # ---- Apply results ----
    df["m7_rpf_flag"] = flag_arr
    df["net_load_MW_m7"] = np.where(flag_arr, -df[col_net].values,
                                    df[col_net].values)
    if df[col_net].isna().any():
        df.loc[df[col_net].isna(), "net_load_MW_m7"] = np.nan

    day_flags = df.groupby([col_site, "date"])["m7_rpf_flag"].any()
    df["m7_rpf_day"] = (df.set_index([col_site, "date"])
                          .index.map(day_flags).values)

    elapsed = _time.perf_counter() - t0
    print(f"\nm7_threshold complete ({elapsed:.1f}s):")
    print(f"  Site-days total:           {n_total:,}")
    print(f"  Skipped (missing data):    {n_miss:,}")
    print(f"  Skipped (negative MW):     {n_neg:,}")
    print(f"  Skipped (midday < 3 pts):  {n_mid_fail:,}")
    print(f"  Skipped (no candidates):   {n_no_cand:,}")
    print(f"  Skipped (no valid pair):   {n_no_pair:,}")
    print(f"  Skipped (threshold gate):  {n_thr_fail:,}")
    print(f"  Skipped (daytime gate):    {n_day_fail:,}")
    print(f"  RPF days flagged:          {n_rpf:,}")
    print(f"  Intervals flagged:         {int(flag_arr.sum()):,}")
    print(f"  Thresholds: min={min_threshold:.2%}, both={min_threshold_both:.2%}")

    return df
