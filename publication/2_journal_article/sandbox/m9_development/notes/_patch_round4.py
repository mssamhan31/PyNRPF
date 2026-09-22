"""One-off patch for round 4: bridge anchors are the nearest finite readings, not the adjacent slots.

Under round 3, a window needs slots a-1 and b+1 finite. At beta_D the missing
block ends where the RPF begins, so the true window's left anchor is missing and
the window is inadmissible; the scorer then settles for a fragment. The bridge
needs a reference level on each side, and the nearest available reading is the
best label-free estimate of it. A window is now admissible iff its interior is
finite and a finite reading exists somewhere before a and after b within the day.
No parameter. Run once from m9_dev/: python notes/_patch_round4.py
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
    # admissibility: interior finite + a finite reading exists on each side
    (
        'def admissible_windows(y: np.ndarray, s: np.ndarray) -> np.ndarray:\n'
        '    """Boolean [start, end] matrix: a window is admissible iff its interior and both\n'
        '    anchors (a-1, b+1) are finite in both y and s. Round-3 rule: missing readings\n'
        '    disqualify the windows that need them, not the whole day."""\n'
        '    bad = ~(np.isfinite(y) & np.isfinite(s))\n'
        '    cum = np.concatenate([[0], np.cumsum(bad)])          # cum[k] = number of bad slots < k\n'
        '    a = SCAN_START + np.arange(N_WINDOWS)[:, None]\n'
        '    b = SCAN_START + np.arange(N_WINDOWS)[None, :]\n'
        '    n_bad = cum[b + 2] - cum[a - 1]                       # bad slots in a-1 .. b+1\n'
        '    return (n_bad == 0) & (b >= a)\n',
        'def nearest_finite(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:\n'
        '    """For every slot k: index of the last finite reading before k and the first after k.\n'
        '\n'
        '    -1 where none exists. A reading is finite when both y and s are finite. These are\n'
        '    the bridge anchors under the round-4 rule: the nearest reference level on each\n'
        '    side of a window, rather than the adjacent slot.\n'
        '    """\n'
        '    ok = np.isfinite(y) & np.isfinite(s)\n'
        '    prev = np.full(SLOTS, -1)\n'
        '    last = -1\n'
        '    for k in range(SLOTS):\n'
        '        prev[k] = last\n'
        '        if ok[k]:\n'
        '            last = k\n'
        '    nxt = np.full(SLOTS, -1)\n'
        '    first = -1\n'
        '    for k in range(SLOTS - 1, -1, -1):\n'
        '        nxt[k] = first\n'
        '        if ok[k]:\n'
        '            first = k\n'
        '    return prev, nxt\n'
        '\n'
        '\n'
        'def admissible_windows(y: np.ndarray, s: np.ndarray) -> np.ndarray:\n'
        '    """Boolean [start, end] matrix: a window is admissible iff its interior is finite in\n'
        '    both y and s and a finite reading exists somewhere before a and after b. Round-4\n'
        '    rule: missing readings disqualify the windows that contain them; the anchors are\n'
        '    the nearest finite readings on each side."""\n'
        '    bad = ~(np.isfinite(y) & np.isfinite(s))\n'
        '    cum = np.concatenate([[0], np.cumsum(bad)])          # cum[k] = number of bad slots < k\n'
        '    a = SCAN_START + np.arange(N_WINDOWS)[:, None]\n'
        '    b = SCAN_START + np.arange(N_WINDOWS)[None, :]\n'
        '    n_bad = cum[b + 1] - cum[a]                           # bad slots in a .. b\n'
        '    prev, nxt = nearest_finite(y, s)\n'
        '    has_left = prev[a] >= 0\n'
        '    has_right = nxt[b] >= 0\n'
        '    return (n_bad == 0) & (b >= a) & has_left & has_right\n',
    ),
    # vectorised residuals: anchors from nearest finite readings
    (
        '    rss_u = np.full((N_WINDOWS, N_WINDOWS), np.nan)\n'
        '    rss_c = np.full((N_WINDOWS, N_WINDOWS), np.nan)\n'
        '    length = np.zeros((N_WINDOWS, N_WINDOWS))\n'
        '    uc_full = u0 - 2.0 * y\n'
        '    t = np.arange(SCAN_START, SCAN_END)\n'
        '    for i, a in enumerate(range(SCAN_START, SCAN_END)):\n'
        '        bs = np.arange(a, SCAN_END)                       # every end for this start\n',
        '    rss_u = np.full((N_WINDOWS, N_WINDOWS), np.nan)\n'
        '    rss_c = np.full((N_WINDOWS, N_WINDOWS), np.nan)\n'
        '    length = np.zeros((N_WINDOWS, N_WINDOWS))\n'
        '    uc_full = u0 - 2.0 * y\n'
        '    t = np.arange(SCAN_START, SCAN_END)\n'
        '    # Anchors: nearest finite reading on each side (u0 is finite exactly where y and\n'
        '    # s are). With no missing data this is a-1 and b+1, identical to the reference.\n'
        '    prev, nxt = nearest_finite(u0, u0)\n'
        '    for i, a in enumerate(range(SCAN_START, SCAN_END)):\n'
        '        bs = np.arange(a, SCAN_END)                       # every end for this start\n'
        '        la = prev[a]\n'
        '        rb = nxt[bs]\n'
        '        if la < 0:\n'
        '            continue\n'
        '        rb_safe = np.where(rb >= 0, rb, SLOTS - 1)\n',
    ),
    (
        '        if variant == "tv":\n'
        '            # Interior edges t-1..t for t in (a, b]; boundary edges at a and b+1.\n',
        '        if variant == "tv":\n'
        '            # Total variation keeps adjacent-slot edges; windows whose adjacent slots are\n'
        '            # missing are excluded by admissible_windows for this variant.\n',
    ),
    (
        '        left = u0[a - 1]\n'
        '        right = u0[bs + 1]\n'
        '        span = (bs - a + 2).astype(float)\n'
        '        frac = (tt[None, :] - (a - 1)) / span[:, None]\n'
        '        line = left + (right[:, None] - left) * frac\n',
        '        left = u0[la]\n'
        '        right = np.where(rb >= 0, u0[rb_safe], np.nan)\n'
        '        span = (rb_safe - la).astype(float)\n'
        '        frac = (tt[None, :] - la) / span[:, None]\n'
        '        line = left + (right[:, None] - left) * frac\n',
    ),
    # score_siteday: do not zero-fill before residuals (anchors must see NaN); zero-fill only for the scale
    (
        '    # Missing values would poison the day-level scale and the tv cumulative sums;\n'
        '    # zero them here knowing every window that reads them is already inadmissible.\n'
        '    y = np.nan_to_num(y, nan=0.0)\n'
        '    s = np.nan_to_num(s, nan=0.0)\n'
        '    u0 = reconstruct_uncorrected(y, s)\n',
        '    u0 = reconstruct_uncorrected(y, s)\n'
        '    # The tv variant needs adjacent anchors; restrict its admissibility accordingly.\n'
        '    if variant == "tv":\n'
        '        bad = ~np.isfinite(u0)\n'
        '        cum = np.concatenate([[0], np.cumsum(bad)])\n'
        '        aa = SCAN_START + np.arange(N_WINDOWS)[:, None]\n'
        '        bb = SCAN_START + np.arange(N_WINDOWS)[None, :]\n'
        '        adm = adm & ((cum[bb + 2] - cum[aa - 1]) == 0)\n'
        '        u0 = np.nan_to_num(u0, nan=0.0)\n'
        '        y = np.nan_to_num(y, nan=0.0)\n',
    ),
])

patch("tests/test_m9_dev.py", [
    (
        'def test_mask_windows_rule_abstains_when_nothing_is_scorable():',
        'def test_nearest_anchor_reaches_a_window_beside_a_missing_block():\n'
        '    # Slots 40-43 missing and the error starting at 44: under adjacent anchors the\n'
        '    # true window cannot start before 45; under nearest anchors it bridges from 39.\n'
        '    y_true, s = synthetic_day(demand_level=4.0, solar_peak=6.0)\n'
        '    y = plant_error(y_true, 44, 58)\n'
        '    y[40:44] = np.nan\n'
        '    adm = ms.admissible_windows(y, s)\n'
        '    assert adm[44 - ms.SCAN_START, 58 - ms.SCAN_START]\n'
        '    r = scored(y, s, missing="mask_windows")\n'
        '    assert r["input_ok"] and r["best_score"] > 0 and r["best_start"] <= 45 and r["best_end"] >= 57\n'
        '\n'
        '\n'
        'def test_mask_windows_rule_abstains_when_nothing_is_scorable():',
    ),
])
print("round-4 patch complete")
