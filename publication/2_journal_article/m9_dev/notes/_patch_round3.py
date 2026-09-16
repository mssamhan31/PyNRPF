"""One-off patch for round 3: missing readings disqualify windows, not days.

Phase 0 said 'abstain only on missing values'. The baseline applied that to the
whole site-day: one missing slot anywhere in 23-72 and the day is UNCERTAIN. At
beta_D that hides 21 of 45 obvious days whose missing slots lie hours before the
RPF window. The remedy is to abstain only the windows that need the missing
slots (interior or anchors) and to abstain the day only when no window is left.
No parameter. Run once from m9_dev/: python notes/_patch_round3.py
"""

import pathlib

DEV = pathlib.Path(__file__).resolve().parents[1]


def patch(path: str, pairs: list[tuple[str, str]]) -> None:
    p = DEV / path
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in t, f"{path}: anchor not found:\n{old[:90]}"
        t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("patched", path)


patch("m9_scorer.py", [
    (
        'def reconstruct_uncorrected(y: np.ndarray, s: np.ndarray) -> np.ndarray:',
        'def admissible_windows(y: np.ndarray, s: np.ndarray) -> np.ndarray:\n'
        '    """Boolean [start, end] matrix: a window is admissible iff its interior and both\n'
        '    anchors (a-1, b+1) are finite in both y and s. Round-3 rule: missing readings\n'
        '    disqualify the windows that need them, not the whole day."""\n'
        '    bad = ~(np.isfinite(y) & np.isfinite(s))\n'
        '    cum = np.concatenate([[0], np.cumsum(bad)])          # cum[k] = number of bad slots < k\n'
        '    a = SCAN_START + np.arange(N_WINDOWS)[:, None]\n'
        '    b = SCAN_START + np.arange(N_WINDOWS)[None, :]\n'
        '    n_bad = cum[b + 2] - cum[a - 1]                       # bad slots in a-1 .. b+1\n'
        '    return (n_bad == 0) & (b >= a)\n'
        '\n'
        '\n'
        'def reconstruct_uncorrected(y: np.ndarray, s: np.ndarray) -> np.ndarray:',
    ),
    (
        'def score_siteday(y: np.ndarray, s: np.ndarray, sigma_floor: float, variant: str = "sq", p_exp: float = 1.0, sigma: float | None = None, scale: str = "overnight", stat: str = "gain") -> dict:',
        'def score_siteday(y: np.ndarray, s: np.ndarray, sigma_floor: float, variant: str = "sq", p_exp: float = 1.0, sigma: float | None = None, scale: str = "overnight", stat: str = "gain", missing: str = "abstain_day") -> dict:',
    ),
    (
        '        stat: "gain" (external scale, score_matrix) or "llr" (scale-free, llr_matrix).',
        '        stat: "gain" (external scale, score_matrix) or "llr" (scale-free, llr_matrix).\n'
        '        missing: "abstain_day" (any missing slot in 23-72 abstains the day) or\n'
        '            "mask_windows" (only windows touching a missing slot are excluded).',
    ),
    (
        '    if not input_ok(y, s):\n'
        '        return dict(input_ok=False, sigma=np.nan, best_start=-1, best_end=-1, best_score=np.nan,\n'
        '                    runner_start=-1, runner_end=-1, runner_score=np.nan, margin_window=np.nan, r_best=np.nan)\n'
        '    u0 = reconstruct_uncorrected(y, s)\n',
        '    adm = admissible_windows(y, s)\n'
        '    ok = input_ok(y, s) if missing == "abstain_day" else bool(adm.any())\n'
        '    if not ok:\n'
        '        return dict(input_ok=False, n_admissible=int(adm.sum()), sigma=np.nan, best_start=-1, best_end=-1, best_score=np.nan,\n'
        '                    runner_start=-1, runner_end=-1, runner_score=np.nan, margin_window=np.nan, r_best=np.nan)\n'
        '    # Missing values would poison the day-level scale and the tv cumulative sums;\n'
        '    # zero them here knowing every window that reads them is already inadmissible.\n'
        '    y = np.nan_to_num(y, nan=0.0)\n'
        '    s = np.nan_to_num(s, nan=0.0)\n'
        '    u0 = reconstruct_uncorrected(y, s)\n',
    ),
    (
        '        sc = score_matrix(gain, length, sig, variant, p_exp)\n'
        '    best, ru = rank(sc)\n',
        '        sc = score_matrix(gain, length, sig, variant, p_exp)\n'
        '    sc = np.where(adm, sc, -np.inf)\n'
        '    best, ru = rank(sc)\n',
    ),
    (
        '    return dict(input_ok=True, sigma=sig,\n',
        '    return dict(input_ok=True, n_admissible=int(adm.sum()), sigma=sig,\n',
    ),
])

patch("m9_eval.py", [
    (
        'def score_cohort(days: list[dict], floor: float, variant: str, p_exp: float, sigma_mode: str, stat: str = "gain") -> pd.DataFrame:',
        'def score_cohort(days: list[dict], floor: float, variant: str, p_exp: float, sigma_mode: str, stat: str = "gain", missing: str = "abstain_day") -> pd.DataFrame:',
    ),
    (
        '        r = ms.score_siteday(d["y"], d["s"], floor, variant=variant, p_exp=p_exp, sigma=sig, scale=day_scale, stat=stat)',
        '        r = ms.score_siteday(d["y"], d["s"], floor, variant=variant, p_exp=p_exp, sigma=sig, scale=day_scale, stat=stat, missing=missing)',
    ),
    (
        'def run(run_name: str, variant: str, p_exp: float, sigma_mode: str, c: float, stat: str = "gain") -> None:',
        'def run(run_name: str, variant: str, p_exp: float, sigma_mode: str, c: float, stat: str = "gain", missing: str = "abstain_day") -> None:',
    ),
    (
        '    config = dict(run=run_name, variant=variant, p_exp=p_exp, sigma_mode=sigma_mode, c=c, stat=stat,',
        '    config = dict(run=run_name, variant=variant, p_exp=p_exp, sigma_mode=sigma_mode, c=c, stat=stat, missing=missing,',
    ),
    (
        '        scores = score_cohort(days, floor, variant, p_exp, sigma_mode, stat)',
        '        scores = score_cohort(days, floor, variant, p_exp, sigma_mode, stat, missing)',
    ),
    (
        '    ap.add_argument("--stat", default="gain", choices=["gain", "llr"], help="evidence statistic")',
        '    ap.add_argument("--stat", default="gain", choices=["gain", "llr"], help="evidence statistic")\n'
        '    ap.add_argument("--missing", default="abstain_day", choices=["abstain_day", "mask_windows"])',
    ),
    (
        '        run(a.run, a.variant, a.p, a.sigma, a.c, a.stat)',
        '        run(a.run, a.variant, a.p, a.sigma, a.c, a.stat, a.missing)',
    ),
])

patch("tests/test_m9_dev.py", [
    (
        '@pytest.mark.parametrize("slot", [ms.SCAN_START - 1, ms.SCAN_START + 10, ms.SCAN_END])\n'
        'def test_missing_net_load_abstains(slot):\n'
        '    y, s = synthetic_day()\n'
        '    y[slot] = np.nan\n'
        '    assert not scored(y, s)["input_ok"]\n',
        '@pytest.mark.parametrize("slot", [ms.SCAN_START - 1, ms.SCAN_START + 10, ms.SCAN_END])\n'
        'def test_missing_net_load_abstains_day_under_default_rule(slot):\n'
        '    y, s = synthetic_day()\n'
        '    y[slot] = np.nan\n'
        '    assert not scored(y, s)["input_ok"]\n'
        '\n'
        '\n'
        'def test_mask_windows_rule_scores_around_a_missing_slot():\n'
        '    # A planted error at 44-58 and a missing reading at slot 30, well before it:\n'
        '    # the day must still be scored and the chosen window must not touch slot 30.\n'
        '    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)\n'
        '    neg = np.flatnonzero(y_true < 0)\n'
        '    y = plant_error(y_true, int(neg[0]), int(neg[-1]))\n'
        '    y[30] = np.nan\n'
        '    r = scored(y, s, missing="mask_windows")\n'
        '    assert r["input_ok"] and r["best_score"] > 0\n'
        '    assert r["best_start"] - 1 > 30 or r["best_end"] + 1 < 30\n'
        '    adm = ms.admissible_windows(y, s)\n'
        '    i, j = 30 - ms.SCAN_START, 31 - ms.SCAN_START\n'
        '    assert not adm[i, j] and not adm[j - 1, j] and adm[j + 1, j + 3]\n'
        '\n'
        '\n'
        'def test_mask_windows_rule_abstains_when_nothing_is_scorable():\n'
        '    y, s = synthetic_day()\n'
        '    y[ms.SCAN_START - 1 : ms.SCAN_END + 1] = np.nan\n'
        '    assert not scored(y, s, missing="mask_windows")["input_ok"]\n',
    ),
])
print("round-3 patch complete")
