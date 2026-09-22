"""One-off patch for round 2: add the scale-free likelihood-ratio statistic.

Run once from m9_dev/: python notes/_patch_round2.py. Kept in notes/ as the record
of exactly what changed between round 1 and round 2.
"""

import pathlib

DEV = pathlib.Path(__file__).resolve().parents[1]


def patch(path: str, pairs: list[tuple[str, str]]) -> None:
    p = DEV / path
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in t, f"{path}: anchor not found:\n{old[:80]}"
        t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("patched", path)


patch("m9_scorer.py", [
    (
        'def bridge_gain_matrix(u0: np.ndarray, y: np.ndarray, variant: str = "sq") -> tuple[np.ndarray, np.ndarray]:\n'
        '    """Vectorised bridge_gain_matrix: identical values to the reference, one loop over starts.',
        'def bridge_residual_matrices(u0: np.ndarray, y: np.ndarray, variant: str = "sq") -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n'
        '    """Per-window misfit under each counterfactual: (rss_u, rss_c, length), vectorised over ends.\n'
        '\n'
        '    rss_u is the misfit of the uncorrected reconstruction against the bridge and rss_c\n'
        '    that of the corrected one; their difference is the gain. Keeping both allows the\n'
        '    scale-free likelihood-ratio statistic in llr_matrix. Units follow the variant.',
    ),
    (
        '    gain = np.full((N_WINDOWS, N_WINDOWS), -np.inf)\n'
        '    length = np.zeros((N_WINDOWS, N_WINDOWS))\n'
        '    uc_full = u0 - 2.0 * y\n'
        '    t = np.arange(SCAN_START, SCAN_END)\n',
        '    rss_u = np.full((N_WINDOWS, N_WINDOWS), np.nan)\n'
        '    rss_c = np.full((N_WINDOWS, N_WINDOWS), np.nan)\n'
        '    length = np.zeros((N_WINDOWS, N_WINDOWS))\n'
        '    uc_full = u0 - 2.0 * y\n'
        '    t = np.arange(SCAN_START, SCAN_END)\n',
    ),
    (
        '            inner = cum_u[kmax] - cum_c[kmax]\n'
        '            left_edge = np.abs(u0[a] - u0[a - 1]) - np.abs(uc_full[a] - u0[a - 1])\n'
        '            right_edge = np.abs(u0[bs + 1] - u0[bs]) - np.abs(u0[bs + 1] - uc_full[bs])\n'
        '            gain[i, i:] = inner + left_edge + right_edge\n'
        '            continue\n',
        '            rss_u[i, i:] = cum_u[kmax] + np.abs(u0[a] - u0[a - 1]) + np.abs(u0[bs + 1] - u0[bs])\n'
        '            rss_c[i, i:] = cum_c[kmax] + np.abs(uc_full[a] - u0[a - 1]) + np.abs(u0[bs + 1] - uc_full[bs])\n'
        '            continue\n',
    ),
    (
        '        if variant == "sq":\n'
        '            gain[i, i:] = (ru**2).sum(1) - (rc**2).sum(1)\n'
        '        else:\n'
        '            gain[i, i:] = np.abs(ru).sum(1) - np.abs(rc).sum(1)\n'
        '    return gain, length\n',
        '        if variant == "sq":\n'
        '            rss_u[i, i:], rss_c[i, i:] = (ru**2).sum(1), (rc**2).sum(1)\n'
        '        else:\n'
        '            rss_u[i, i:], rss_c[i, i:] = np.abs(ru).sum(1), np.abs(rc).sum(1)\n'
        '    return rss_u, rss_c, length\n'
        '\n'
        '\n'
        'def bridge_gain_matrix(u0: np.ndarray, y: np.ndarray, variant: str = "sq") -> tuple[np.ndarray, np.ndarray]:\n'
        '    """Vectorised gain = rss_u - rss_c; identical values to bridge_gain_matrix_reference."""\n'
        '    rss_u, rss_c, length = bridge_residual_matrices(u0, y, variant)\n'
        '    gain = np.where(np.isfinite(rss_u), rss_u - rss_c, -np.inf)\n'
        '    return gain, length\n',
    ),
    (
        '    return np.where(np.isfinite(sc), sc, -np.inf)\n'
        '\n'
        '\n'
        '@dataclass(frozen=True)\n',
        '    return np.where(np.isfinite(sc), sc, -np.inf)\n'
        '\n'
        '\n'
        'def llr_matrix(rss_u: np.ndarray, rss_c: np.ndarray, length: np.ndarray, floor: float, p_exp: float = 1.0) -> np.ndarray:\n'
        '    """Scale-free evidence per window: the profile likelihood ratio of the two fits.\n'
        '\n'
        '    For nested Gaussian fits with unknown variance, profiling the variance out gives\n'
        '    LLR = (L/2) * log(RSS_u / RSS_c). The corrected reconstruction\'s own misfit is the\n'
        '    yardstick, so no external noise scale enters and the statistic is comparable\n'
        '    across stations by construction. p_exp = 0 returns the full LLR (evidence grows\n'
        '    with a sustained effect); p_exp = 1 the per-slot value. Each RSS is floored at\n'
        '    L * floor^2 so a near-perfect fit on a short window cannot produce an unbounded\n'
        '    ratio; the floor is the label-free resolution proxy fixed in Phase 0.\n'
        '    """\n'
        '    lam = np.maximum(length, 1.0) * floor**2\n'
        '    with np.errstate(divide="ignore", invalid="ignore"):\n'
        '        ratio = np.log((rss_u + lam) / (rss_c + lam))\n'
        '        sc = 0.5 * ratio * np.power(np.maximum(length, 1.0), 1.0 - p_exp)\n'
        '    return np.where(np.isfinite(sc), sc, -np.inf)\n'
        '\n'
        '\n'
        '@dataclass(frozen=True)\n',
    ),
    (
        'def score_siteday(y: np.ndarray, s: np.ndarray, sigma_floor: float, variant: str = "sq", p_exp: float = 1.0, sigma: float | None = None, scale: str = "overnight") -> dict:',
        'def score_siteday(y: np.ndarray, s: np.ndarray, sigma_floor: float, variant: str = "sq", p_exp: float = 1.0, sigma: float | None = None, scale: str = "overnight", stat: str = "gain") -> dict:',
    ),
    (
        '        scale: "overnight" or "fullday" day-level scale definition when sigma is None.',
        '        scale: "overnight" or "fullday" day-level scale definition when sigma is None.\n'
        '        stat: "gain" (external scale, score_matrix) or "llr" (scale-free, llr_matrix).',
    ),
    (
        '    gain, length = bridge_gain_matrix(u0, y, variant)\n'
        '    sc = score_matrix(gain, length, sig, variant, p_exp)\n',
        '    rss_u, rss_c, length = bridge_residual_matrices(u0, y, variant)\n'
        '    if stat == "llr":\n'
        '        sc = llr_matrix(rss_u, rss_c, length, sigma_floor, p_exp)\n'
        '    else:\n'
        '        gain = np.where(np.isfinite(rss_u), rss_u - rss_c, -np.inf)\n'
        '        sc = score_matrix(gain, length, sig, variant, p_exp)\n',
    ),
])

patch("m9_eval.py", [
    (
        'def score_cohort(days: list[dict], floor: float, variant: str, p_exp: float, sigma_mode: str) -> pd.DataFrame:',
        'def score_cohort(days: list[dict], floor: float, variant: str, p_exp: float, sigma_mode: str, stat: str = "gain") -> pd.DataFrame:',
    ),
    (
        '        r = ms.score_siteday(d["y"], d["s"], floor, variant=variant, p_exp=p_exp, sigma=sig, scale=day_scale)',
        '        r = ms.score_siteday(d["y"], d["s"], floor, variant=variant, p_exp=p_exp, sigma=sig, scale=day_scale, stat=stat)',
    ),
    (
        'def run(run_name: str, variant: str, p_exp: float, sigma_mode: str, c: float) -> None:',
        'def run(run_name: str, variant: str, p_exp: float, sigma_mode: str, c: float, stat: str = "gain") -> None:',
    ),
    (
        '    config = dict(run=run_name, variant=variant, p_exp=p_exp, sigma_mode=sigma_mode, c=c,',
        '    config = dict(run=run_name, variant=variant, p_exp=p_exp, sigma_mode=sigma_mode, c=c, stat=stat,',
    ),
    (
        '        scores = score_cohort(days, floor, variant, p_exp, sigma_mode)',
        '        scores = score_cohort(days, floor, variant, p_exp, sigma_mode, stat)',
    ),
    (
        '    ap.add_argument("--c", type=float, default=0.7, help="public confidence control")',
        '    ap.add_argument("--c", type=float, default=0.7, help="public confidence control")\n'
        '    ap.add_argument("--stat", default="gain", choices=["gain", "llr"], help="evidence statistic")',
    ),
    (
        '        run(a.run, a.variant, a.p, a.sigma, a.c)',
        '        run(a.run, a.variant, a.p, a.sigma, a.c, a.stat)',
    ),
])

patch("compare_runs.py", [
    (
        'EPS = 1e-9  # equality tolerance on recall and precision, so 0.804 vs 0.804 is not a loss',
        'PREC_TOL = 0.01   # sampling-noise tolerance on station energy precision (round-1 note)\n'
        'EPS = 1e-9',
    ),
    (
        '    lose_recall = st.loc[has_sure[has_sure].index.intersection(st.index), "d_sure_day_recall"] < -EPS\n'
        '    prec_pairs = st[["energy_precision_inc", "energy_precision_cand"]].dropna()\n'
        '    lose_prec = (prec_pairs.energy_precision_cand - prec_pairs.energy_precision_inc) < -EPS',
        '    # One sure day of recall per station is the tolerance: 1 / n_sure_rpf_days (round-1 note).\n'
        '    n_sure = ist.loc[st.index, "sure_rpf_days"].replace(0, np.nan)\n'
        '    lose_recall = st["d_sure_day_recall"] < -(1.0 / n_sure) - EPS\n'
        '    prec_pairs = st[["energy_precision_inc", "energy_precision_cand"]].dropna()\n'
        '    lose_prec = (prec_pairs.energy_precision_cand - prec_pairs.energy_precision_inc) < -PREC_TOL',
    ),
    (
        '    gate_ok = bool((cp["energy_precision"] >= GATE).all())',
        '    gate_ok = bool(cp.loc["beta", "energy_precision"] >= GATE)  # Alpha sits below the gate by label construction',
    ),
    (
        '    print(f"rule 1  energy precision >= {GATE} both cohorts: {\'PASS\' if gate_ok else \'FAIL\'}")',
        '    print(f"rule 1  Beta energy precision >= {GATE}: {\'PASS\' if gate_ok else \'FAIL\'}   (Alpha {cp.loc[\'alpha\', \'energy_precision\']:.3f}, informational)")',
    ),
    ("import pandas as pd\n", "import numpy as np\nimport pandas as pd\n"),
])

patch("tests/test_m9_dev.py", [
    (
        'def test_one_slot_and_full_window():',
        'def test_llr_statistic_finds_planted_window_and_null_on_clean_day():\n'
        '    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0, noise=0.05, seed=1)\n'
        '    neg = np.flatnonzero(y_true < 0)\n'
        '    a, b = int(neg[0]), int(neg[-1])\n'
        '    r = scored(plant_error(y_true, a, b), s, stat="llr")\n'
        '    assert r["best_score"] > 0 and a <= r["best_start"] <= a + 1 and b - 1 <= r["best_end"] <= b\n'
        '    y_clean, s_clean = synthetic_day(demand_level=8.0, solar_peak=5.0, noise=0.05, seed=2)\n'
        '    assert scored(y_clean, s_clean, stat="llr")["best_start"] < 0\n'
        '\n'
        '\n'
        'def test_one_slot_and_full_window():',
    ),
])
print("round-2 patch complete")
