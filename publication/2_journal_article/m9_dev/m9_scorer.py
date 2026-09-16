"""M9 reference scorer: counterfactual bridge plausibility for reverse power flow (RPF) sign errors.

Purpose: for one site-day of 96 fifteen-minute readings, compare two reconstructions
of underlying demand — keep the recorded net-load sign, or flip it inside one
contiguous candidate window — and score every window by how much better a straight
line through the window's two outside anchors explains the corrected reconstruction
than the uncorrected one. NO_CORRECTION is an explicit candidate scored at zero.

Inputs:  y  recorded net load, MW, 96 slots (positive = import)
         s  estimated solar generation, MW, 96 slots
Outputs: best window, its raw evidence score, runner-up and margins; a calibrated
         probability that the day needs correction; the AUTO_CORRECT / AUTO_KEEP /
         UNCERTAIN outcome; the proposed corrected net-load series, MW.
Key steps: reconstruct U0 = s + y; for every window [a, b] in slots 24-71 compute the
         reduction in squared deviation from the anchor bridge when y is negated
         inside the window; divide by the day's overnight noise scale and the window
         length; rank jointly with the null; calibrate; decide.

bridge_gain_matrix_reference is the plain-loop definition; bridge_gain_matrix is the
vectorised version checked against it in tests/.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOTS = 96
SCAN_START = 24          # 06:00 inclusive
SCAN_END = 72            # 18:00 exclusive; slots 23 and 72 are anchors only
N_WINDOWS = SCAN_END - SCAN_START
HOURS_PER_SLOT = 0.25
OVERNIGHT = slice(0, 24)  # 00:00-06:00, used for the noise scale
TIE_TOL = 1e-12

AUTO_CORRECT = "AUTO_CORRECT"
AUTO_KEEP = "AUTO_KEEP"
UNCERTAIN = "UNCERTAIN"
NO_CORRECTION = "NO_CORRECTION"


# --------------------------------------------------------------------------- inputs

def input_ok(y: np.ndarray, s: np.ndarray) -> bool:
    """True when every slot the scorer reads (anchors 23 and 72 included) is finite."""
    lo, hi = SCAN_START - 1, SCAN_END + 1
    return bool(np.isfinite(y[lo:hi]).all() and np.isfinite(s[lo:hi]).all())


def nearest_finite(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every slot k: index of the last finite reading before k and the first after k.

    -1 where none exists. A reading is finite when both y and s are finite. These are
    the bridge anchors under the round-4 rule: the nearest reference level on each
    side of a window, rather than the adjacent slot.
    """
    ok = np.isfinite(y) & np.isfinite(s)
    prev = np.full(SLOTS, -1)
    last = -1
    for k in range(SLOTS):
        prev[k] = last
        if ok[k]:
            last = k
    nxt = np.full(SLOTS, -1)
    first = -1
    for k in range(SLOTS - 1, -1, -1):
        nxt[k] = first
        if ok[k]:
            first = k
    return prev, nxt


def admissible_windows(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Boolean [start, end] matrix: a window is admissible iff its interior is finite in
    both y and s and a finite reading exists somewhere before a and after b. Round-4
    rule: missing readings disqualify the windows that contain them; the anchors are
    the nearest finite readings on each side."""
    bad = ~(np.isfinite(y) & np.isfinite(s))
    cum = np.concatenate([[0], np.cumsum(bad)])          # cum[k] = number of bad slots < k
    a = SCAN_START + np.arange(N_WINDOWS)[:, None]
    b = SCAN_START + np.arange(N_WINDOWS)[None, :]
    n_bad = cum[b + 1] - cum[a]                           # bad slots in a .. b
    prev, nxt = nearest_finite(y, s)
    has_left = prev[a] >= 0
    has_right = nxt[b] >= 0
    return (n_bad == 0) & (b >= a) & has_left & has_right


def reconstruct_uncorrected(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Underlying demand if the recorded sign is kept: U0 = s + y, MW."""
    return s + y


def overnight_scale(u0: np.ndarray, floor: float) -> float:
    """Day-level noise scale, MW: median absolute overnight step of U0, floored.

    The overnight slots carry no solar and no RPF, so their step size is a
    label-free estimate of how rough ordinary demand is for this site on this day.
    Missing overnight values are ignored; if none are usable the floor is returned.
    """
    steps = np.abs(np.diff(u0[OVERNIGHT]))
    steps = steps[np.isfinite(steps)]
    return float(max(np.median(steps), floor)) if steps.size else float(floor)


def fullday_scale(u0: np.ndarray, floor: float) -> float:
    """Day-level noise scale, MW: median absolute step of U0 over the whole day, floored.

    Round-1 remedy candidate. The overnight-only scale assumes night-time roughness
    represents how smooth daytime demand is; on stations with jagged overnight load
    (beta_A, beta_E) it overstates the yardstick by an order of magnitude and
    compresses the evidence on obvious days. The median over all 95 steps is still
    label-free and is barely moved by a smooth RPF hump, whose steps are small.
    """
    steps = np.abs(np.diff(u0))
    steps = steps[np.isfinite(steps)]
    return float(max(np.median(steps), floor)) if steps.size else float(floor)


# --------------------------------------------------------------------------- scoring

def bridge_gain_matrix_reference(u0: np.ndarray, y: np.ndarray, variant: str = "sq") -> tuple[np.ndarray, np.ndarray]:
    """Plain-loop reference for bridge_gain_matrix; kept so the fast version can be checked against it.

    Args:
        u0: uncorrected demand, MW, 96 slots.
        y: recorded net load, MW, 96 slots.
        variant: "sq" squared deviation from the bridge (Candidate A);
                 "abs" absolute deviation (Candidate B);
                 "tv" total variation over the window's edges (Candidate C).

    Returns:
        gain[i, j] for a = 24 + i, b = 24 + j, j >= i, else -inf; and the window
        length matrix. Units: MW^2 for "sq", MW for "abs" and "tv".
    """
    gain = np.full((N_WINDOWS, N_WINDOWS), -np.inf)
    length = np.zeros((N_WINDOWS, N_WINDOWS))
    uc_full = u0 - 2.0 * y  # corrected demand where the sign is flipped
    for i, a in enumerate(range(SCAN_START, SCAN_END)):
        for j in range(i, N_WINDOWS):
            b = SCAN_START + j
            w = slice(a, b + 1)
            n = b - a + 1
            length[i, j] = n
            if variant == "tv":
                # Edges a-1..a through b..b+1; only the window interior differs.
                seg_u = u0[a - 1 : b + 2]
                seg_c = seg_u.copy()
                seg_c[1:-1] = uc_full[w]
                gain[i, j] = np.abs(np.diff(seg_u)).sum() - np.abs(np.diff(seg_c)).sum()
                continue
            # Straight bridge from the anchor before a to the anchor after b. Both
            # reconstructions share it because they agree outside the window.
            line = np.linspace(u0[a - 1], u0[b + 1], n + 2)[1:-1]
            ru, rc = u0[w] - line, uc_full[w] - line
            if variant == "sq":
                gain[i, j] = (ru**2).sum() - (rc**2).sum()
            elif variant == "abs":
                gain[i, j] = np.abs(ru).sum() - np.abs(rc).sum()
            else:
                raise ValueError(f"unknown variant {variant!r}")
    return gain, length


def bridge_residual_matrices(u0: np.ndarray, y: np.ndarray, variant: str = "sq") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window misfit under each counterfactual: (rss_u, rss_c, length), vectorised over ends.

    rss_u is the misfit of the uncorrected reconstruction against the bridge and rss_c
    that of the corrected one; their difference is the gain. Keeping both allows the
    scale-free likelihood-ratio statistic in llr_matrix. Units follow the variant.

    For each start a, every end b is handled at once: the bridge line for [a, b] is
    left + (right_b - left) * (t - (a-1)) / (b - a + 2), masked to t <= b.
    See bridge_gain_matrix_reference for the definition of each variant.
    """
    if variant not in ("sq", "abs", "tv"):
        raise ValueError(f"unknown variant {variant!r}")
    rss_u = np.full((N_WINDOWS, N_WINDOWS), np.nan)
    rss_c = np.full((N_WINDOWS, N_WINDOWS), np.nan)
    length = np.zeros((N_WINDOWS, N_WINDOWS))
    uc_full = u0 - 2.0 * y
    t = np.arange(SCAN_START, SCAN_END)
    # Anchors: nearest finite reading on each side (u0 is finite exactly where y and
    # s are). With no missing data this is a-1 and b+1, identical to the reference.
    prev, nxt = nearest_finite(u0, u0)
    for i, a in enumerate(range(SCAN_START, SCAN_END)):
        bs = np.arange(a, SCAN_END)                       # every end for this start
        la = prev[a]
        rb = nxt[bs]
        if la < 0:
            continue
        rb_safe = np.where(rb >= 0, rb, SLOTS - 1)
        tt = t[i:]                                        # interior slots a .. 71
        mask = tt[None, :] <= bs[:, None]                 # [b, t]: slot belongs to window
        n = mask.sum(1)
        length[i, i:] = n
        if variant == "tv":
            # Total variation keeps adjacent-slot edges; windows whose adjacent slots are
            # missing are excluded by admissible_windows for this variant.
            du_int = np.abs(np.diff(u0[a - 1 : SCAN_END + 1]))[1:]     # edges a..a+1 ... 70..71 -> index k is edge (a+k, a+k+1)
            dc_int = np.abs(np.diff(uc_full[a - 1 : SCAN_END + 1]))[1:]
            kmax = bs - a                                                # number of interior edges for each b
            cum_u = np.concatenate([[0.0], np.cumsum(du_int)])
            cum_c = np.concatenate([[0.0], np.cumsum(dc_int)])
            rss_u[i, i:] = cum_u[kmax] + np.abs(u0[a] - u0[a - 1]) + np.abs(u0[bs + 1] - u0[bs])
            rss_c[i, i:] = cum_c[kmax] + np.abs(uc_full[a] - u0[a - 1]) + np.abs(u0[bs + 1] - uc_full[bs])
            continue
        left = u0[la]
        right = np.where(rb >= 0, u0[rb_safe], np.nan)
        span = (rb_safe - la).astype(float)
        frac = (tt[None, :] - la) / span[:, None]
        line = left + (right[:, None] - left) * frac
        ru = np.where(mask, u0[tt][None, :] - line, 0.0)
        rc = np.where(mask, uc_full[tt][None, :] - line, 0.0)
        if variant == "sq":
            rss_u[i, i:], rss_c[i, i:] = (ru**2).sum(1), (rc**2).sum(1)
        else:
            rss_u[i, i:], rss_c[i, i:] = np.abs(ru).sum(1), np.abs(rc).sum(1)
    return rss_u, rss_c, length


def bridge_gain_matrix(u0: np.ndarray, y: np.ndarray, variant: str = "sq") -> tuple[np.ndarray, np.ndarray]:
    """Vectorised gain = rss_u - rss_c; identical values to bridge_gain_matrix_reference."""
    rss_u, rss_c, length = bridge_residual_matrices(u0, y, variant)
    gain = np.where(np.isfinite(rss_u), rss_u - rss_c, -np.inf)
    return gain, length


def score_matrix(gain: np.ndarray, length: np.ndarray, sigma: float, variant: str = "sq", p_exp: float = 1.0) -> np.ndarray:
    """Dimensionless evidence per window: gain / (scale * length^p_exp).

    The scale is sigma^2 for the squared variant and sigma for the others, so every
    variant is dimensionless. p_exp = 1 is the per-slot mean adopted in Phase 0.
    """
    scale = sigma**2 if variant == "sq" else sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = gain / (scale * np.power(np.maximum(length, 1.0), p_exp))
    return np.where(np.isfinite(sc), sc, -np.inf)


def llr_matrix(rss_u: np.ndarray, rss_c: np.ndarray, length: np.ndarray, floor: float, p_exp: float = 1.0) -> np.ndarray:
    """Scale-free evidence per window: the profile likelihood ratio of the two fits.

    For nested Gaussian fits with unknown variance, profiling the variance out gives
    LLR = (L/2) * log(RSS_u / RSS_c). The corrected reconstruction's own misfit is the
    yardstick, so no external noise scale enters and the statistic is comparable
    across stations by construction. p_exp = 0 returns the full LLR (evidence grows
    with a sustained effect); p_exp = 1 the per-slot value. Each RSS is floored at
    L * floor^2 so a near-perfect fit on a short window cannot produce an unbounded
    ratio; the floor is the label-free resolution proxy fixed in Phase 0.
    """
    lam = np.maximum(length, 1.0) * floor**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log((rss_u + lam) / (rss_c + lam))
        sc = 0.5 * ratio * np.power(np.maximum(length, 1.0), 1.0 - p_exp)
    return np.where(np.isfinite(sc), sc, -np.inf)


@dataclass(frozen=True)
class Candidate:
    """One ranked candidate: a window (start, end inclusive) or the null."""

    start: int
    end: int
    score: float

    @property
    def is_null(self) -> bool:
        return self.start < 0

    @property
    def length(self) -> int:
        return 0 if self.is_null else self.end - self.start + 1


NULL = Candidate(-1, -1, 0.0)


def best_window(sc: np.ndarray) -> Candidate:
    """Highest-scoring window; ties go to the shorter, then the earlier, window."""
    top = sc.max()
    if not np.isfinite(top):
        return NULL
    ii, jj = np.nonzero(sc >= top - TIE_TOL)
    lengths = jj - ii
    order = np.lexsort((ii, lengths))  # primary: length, secondary: start
    i, j = ii[order[0]], jj[order[0]]
    return Candidate(SCAN_START + i, SCAN_START + j, float(sc[i, j]))


def runner_up(sc: np.ndarray, best: Candidate) -> Candidate:
    """Best window not overlapping the best one: a genuinely different explanation."""
    if best.is_null:
        return best_window(sc)
    a = SCAN_START + np.arange(N_WINDOWS)[:, None]
    b = SCAN_START + np.arange(N_WINDOWS)[None, :]
    overlaps = ~((b < best.start) | (a > best.end))
    return best_window(np.where(overlaps, -np.inf, sc))


def rank(sc: np.ndarray) -> tuple[Candidate, Candidate]:
    """Rank the null jointly with all windows. Returns (best, runner_up).

    The null scores zero. If no window beats zero the null wins and the best window
    is returned as the runner-up so it stays available for inspection.
    """
    bw = best_window(sc)
    if bw.is_null or bw.score <= 0.0 + TIE_TOL:
        return NULL, bw
    ru = runner_up(sc, bw)
    return bw, ru


# --------------------------------------------------------------------------- calibration

def signed_log(r: np.ndarray | float) -> np.ndarray | float:
    """Variance-stabilising transform: sign(r) * log(1 + |r|)."""
    return np.sign(r) * np.log1p(np.abs(r))


@dataclass
class Calibrator:
    """Two-coefficient logistic on the signed-log score: p = logistic(alpha + beta * z)."""

    alpha: float = 0.0
    beta: float = 1.0

    def fit(self, r: np.ndarray, label: np.ndarray) -> "Calibrator":
        from sklearn.linear_model import LogisticRegression

        z = signed_log(np.asarray(r, dtype=float)).reshape(-1, 1)
        m = LogisticRegression(max_iter=2000).fit(z, np.asarray(label, dtype=int))
        self.alpha, self.beta = float(m.intercept_[0]), float(m.coef_[0, 0])
        return self

    def probability(self, r: np.ndarray | float) -> np.ndarray | float:
        z = signed_log(r)
        return 1.0 / (1.0 + np.exp(-(self.alpha + self.beta * z)))

    def raw_threshold(self, c: float) -> float:
        """Raw score at which p = c, for reporting the thresholds actually applied."""
        z = (np.log(c / (1 - c)) - self.alpha) / self.beta
        return float(np.sign(z) * (np.expm1(abs(z))))


def decide(p: float, c: float) -> str:
    """Three-way action from a calibrated probability and one public control c in (0.5, 1)."""
    if p >= c:
        return AUTO_CORRECT
    if p <= 1.0 - c:
        return AUTO_KEEP
    return UNCERTAIN


# --------------------------------------------------------------------------- site-day

def corrected_series(y: np.ndarray, cand: Candidate) -> np.ndarray:
    """Net load with the sign flipped inside the candidate window, MW. Null returns a copy."""
    out = y.copy()
    if not cand.is_null:
        out[cand.start : cand.end + 1] = -y[cand.start : cand.end + 1]
    return out


def score_siteday(y: np.ndarray, s: np.ndarray, sigma_floor: float, variant: str = "sq", p_exp: float = 1.0, sigma: float | None = None, scale: str = "overnight", stat: str = "gain", missing: str = "abstain_day") -> dict:
    """Score one site-day end to end, without calibration.

    Args:
        y, s: 96-slot arrays, MW.
        sigma_floor: floor for the overnight scale, MW.
        variant, p_exp: see score_matrix.
        sigma: override the noise scale (used when a station-level scale is supplied).
        scale: "overnight" or "fullday" day-level scale definition when sigma is None.
        stat: "gain" (external scale, score_matrix) or "llr" (scale-free, llr_matrix).
        missing: "abstain_day" (any missing slot in 23-72 abstains the day) or
            "mask_windows" (only windows touching a missing slot are excluded).

    Returns:
        Dict with input_ok, sigma, best/runner-up start, end and score, margins, and
        the raw evidence r_best (0.0 when the null wins).
    """
    adm = admissible_windows(y, s)
    ok = input_ok(y, s) if missing == "abstain_day" else bool(adm.any())
    if not ok:
        return dict(input_ok=False, n_admissible=int(adm.sum()), sigma=np.nan, best_start=-1, best_end=-1, best_score=np.nan,
                    runner_start=-1, runner_end=-1, runner_score=np.nan, margin_window=np.nan, r_best=np.nan)
    u0 = reconstruct_uncorrected(y, s)
    # The tv variant needs adjacent anchors; restrict its admissibility accordingly.
    if variant == "tv":
        bad = ~np.isfinite(u0)
        cum = np.concatenate([[0], np.cumsum(bad)])
        aa = SCAN_START + np.arange(N_WINDOWS)[:, None]
        bb = SCAN_START + np.arange(N_WINDOWS)[None, :]
        adm = adm & ((cum[bb + 2] - cum[aa - 1]) == 0)
        u0 = np.nan_to_num(u0, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
    if sigma is not None:
        sig = float(sigma)
    else:
        sig = overnight_scale(u0, sigma_floor) if scale == "overnight" else fullday_scale(u0, sigma_floor)
    rss_u, rss_c, length = bridge_residual_matrices(u0, y, variant)
    if stat == "llr":
        sc = llr_matrix(rss_u, rss_c, length, sigma_floor, p_exp)
    else:
        gain = np.where(np.isfinite(rss_u), rss_u - rss_c, -np.inf)
        sc = score_matrix(gain, length, sig, variant, p_exp)
    sc = np.where(adm, sc, -np.inf)
    best, ru = rank(sc)
    # The raw evidence is the best WINDOW score even when the null wins, so a day
    # with a weak negative best window is distinguishable from one with none.
    r_best = ru.score if best.is_null else best.score
    return dict(input_ok=True, n_admissible=int(adm.sum()), sigma=sig,
                best_start=best.start, best_end=best.end, best_score=best.score,
                runner_start=ru.start, runner_end=ru.end, runner_score=ru.score,
                margin_window=(best.score - max(ru.score, 0.0)) if not best.is_null else np.nan,
                r_best=float(r_best))
