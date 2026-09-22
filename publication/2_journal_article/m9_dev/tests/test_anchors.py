"""Bridge-anchor rules of the M9 scorer: "nearest" (frozen), "edge" and "gap_edge". Synthetic days only.

(a) "nearest" is unchanged bit for bit. The expectations under fixtures/ were written by
    the scorer at commit ffa1d92 (code sha256 f05c148d...) from the same three synthetic
    days that fixture_days() builds, and are read back here, never recomputed.
(b) On a clean planted sign flip "edge" and "gap_edge" find the planted window.
(c) With a missing reading immediately before the span, "gap_edge" anchors on the
    window's own start slot on that side only, while "nearest" reaches back to the
    reading before the gap. Every rule is also checked, window by window, against a
    plain-loop statement of its definition.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import m9_metrics as mx  # noqa: E402
import m9_scorer as ms  # noqa: E402
from test_m9_dev import FLOOR, plant_error, synthetic_day  # noqa: E402

FIXTURES = HERE / "fixtures"
FROZEN = dict(variant="sq", p_exp=0.0, stat="llr", missing="mask_windows", edges="inwardx")


def fixture_days() -> dict[str, tuple[np.ndarray, np.ndarray, int, int]]:
    """The three synthetic days behind the stored expectations: (y, s, planted start, planted end)."""
    days = {}
    y_true, s = synthetic_day()
    neg = np.flatnonzero(y_true < 0)
    a, b = int(neg[0]), int(neg[-1])
    days["clean"] = (plant_error(y_true, a, b), s, a, b)
    y_true, s = synthetic_day(ramp=0.5, noise=0.1, seed=1)
    neg = np.flatnonzero(y_true < 0)
    a, b = int(neg[0]), int(neg[-1])
    y = plant_error(y_true, a, b)
    y[a - 1] = np.nan                       # net load missing immediately before the span
    days["gap_before"] = (y, s, a, b)
    y_true, s = synthetic_day(noise=0.2, seed=2)
    y = plant_error(y_true, 44, 58)
    y[40:44] = np.nan                       # a block ending where the span starts
    y[30] = np.nan                          # an isolated gap well before it
    s = s.copy()
    s[59] = np.nan                          # solar missing immediately after the span
    days["gaps_both"] = (y, s, 44, 58)
    return days


# --------------------------------------------------------------------------- plain-loop definition

def anchor_slots(y: np.ndarray, s: np.ndarray, a: int, b: int, anchors: str) -> tuple[int, int]:
    """(left, right) anchor slots of window [a, b] under one rule, stated plainly; -1 if none."""
    finite = np.isfinite(y) & np.isfinite(s)
    prev = max((k for k in range(a) if finite[k]), default=-1)
    nxt = min((k for k in range(b + 1, ms.SLOTS) if finite[k]), default=-1)
    left_on_edge = anchors == "edge" or (anchors == "gap_edge" and not finite[a - 1])
    right_on_edge = anchors == "edge" or (anchors == "gap_edge" and not finite[b + 1])
    return (a if left_on_edge else prev), (b if right_on_edge else nxt)


def misfit_between(y: np.ndarray, s: np.ndarray, a: int, b: int, left: int, right: int, variant: str = "sq") -> tuple[float, float, int]:
    """(rss_u, rss_c, residual slots) of window [a, b] bridged between the given anchor slots.

    Each story's line passes through that story's own values at the anchors; story B is
    S - y inside the window and S + y outside. Window slots that are anchors carry no
    misfit. NaN misfit when an anchor is missing or no residual slot remains.
    """
    slots = np.array([t for t in range(a, b + 1) if t not in (left, right)])
    if left < 0 or right < 0 or slots.size == 0:
        return np.nan, np.nan, 0
    inside = np.zeros(ms.SLOTS, dtype=bool)
    inside[a : b + 1] = True
    out = []
    for series in (s + y, np.where(inside, s - y, s + y)):
        line = series[left] + (series[right] - series[left]) * (slots - left) / (right - left)
        dev = series[slots] - line
        out.append(float((dev**2).sum() if variant == "sq" else np.abs(dev).sum()))
    return out[0], out[1], int(slots.size)


# --------------------------------------------------------------------------- (a) nearest is unchanged

@pytest.mark.parametrize("variant", ["sq", "abs"])
def test_nearest_matrices_bit_for_bit_unchanged(variant):
    stored = np.load(FIXTURES / "anchors_nearest_expected.npz")
    for name, (y, s, _, _) in fixture_days().items():
        u0 = ms.reconstruct_uncorrected(y, s)
        by_default = ms.bridge_residual_matrices(u0, y, variant)
        explicit = ms.bridge_residual_matrices(u0, y, variant, anchors="nearest")
        for got, key in zip(by_default, ("rss_u", "rss_c", "length"), strict=True):
            assert np.array_equal(got, stored[f"{name}_{variant}_{key}"], equal_nan=True), (name, key)
        for got, want in zip(explicit, by_default, strict=True):
            assert np.array_equal(got, want, equal_nan=True)


def test_nearest_siteday_unchanged_under_frozen_settings():
    stored = json.loads((FIXTURES / "anchors_nearest_expected.json").read_text(encoding="utf-8"))
    for name, (y, s, _, _) in fixture_days().items():
        for got in (ms.score_siteday(y, s, FLOOR, **FROZEN), ms.score_siteday(y, s, FLOOR, anchors="nearest", **FROZEN)):
            for key, want in stored[name].items():
                if want is None:
                    assert np.isnan(got[key]), (name, key)
                else:
                    assert got[key] == want, (name, key, got[key], want)


def test_nearest_admissibility_unchanged():
    for y, s, _, _ in fixture_days().values():
        assert np.array_equal(ms.admissible_windows(y, s), ms.admissible_windows(y, s, "nearest"))


# --------------------------------------------------------------------------- every rule against its definition

@pytest.mark.parametrize("anchors", ms.ANCHORS)
@pytest.mark.parametrize("variant", ["sq", "abs"])
def test_vectorised_matches_plain_loop_definition(anchors, variant):
    for name, (y, s, _, _) in fixture_days().items():
        u0 = ms.reconstruct_uncorrected(y, s)
        rss_u, rss_c, length = ms.bridge_residual_matrices(u0, y, variant, anchors)
        adm = ms.admissible_windows(y, s, anchors)
        assert not adm[~np.isfinite(rss_u)].any(), name      # no admissible window without a misfit
        for i, j in zip(*np.nonzero(adm), strict=True):
            a, b = ms.SCAN_START + i, ms.SCAN_START + j
            left, right = anchor_slots(y, s, a, b, anchors)
            ru, rc, n = misfit_between(y, s, a, b, left, right, variant)
            assert rss_u[i, j] == pytest.approx(ru, rel=1e-9, abs=1e-9), (name, a, b)
            assert rss_c[i, j] == pytest.approx(rc, rel=1e-9, abs=1e-9), (name, a, b)
            assert length[i, j] == n, (name, a, b)


def test_short_windows_are_inadmissible_under_edge_anchors():
    y, s, _, _ = fixture_days()["clean"]
    i = 50 - ms.SCAN_START
    adm = ms.admissible_windows(y, s, "edge")
    assert not adm[i, i] and not adm[i, i + 1] and adm[i, i + 2]     # both anchors on the edges: three slots needed
    assert np.array_equal(ms.admissible_windows(y, s, "gap_edge"), ms.admissible_windows(y, s))   # no gap: same set
    y2, s2, a2, _ = fixture_days()["gap_before"]
    adm2 = ms.admissible_windows(y2, s2, "gap_edge")
    i2 = a2 - ms.SCAN_START
    assert not adm2[i2, i2] and adm2[i2, i2 + 1]                     # one anchor on the edge: two slots needed
    assert adm2[i2 + 1, i2 + 1]                                      # no gap beside it: a one-slot window stands


def test_anchor_rule_validation():
    y, s, _, _ = fixture_days()["clean"]
    u0 = ms.reconstruct_uncorrected(y, s)
    with pytest.raises(ValueError):
        ms.bridge_residual_matrices(u0, y, "sq", "adjacent")
    with pytest.raises(ValueError):
        ms.bridge_residual_matrices(u0, y, "tv", "edge")


# --------------------------------------------------------------------------- (b) planted flip, no gaps

@pytest.mark.parametrize("anchors", ["edge", "gap_edge"])
def test_edge_rules_find_planted_window_on_clean_day(anchors):
    y, s, a, b = fixture_days()["clean"]
    r = ms.score_siteday(y, s, FLOOR, anchors=anchors, **FROZEN)
    assert r["input_ok"] and r["best_score"] > 0
    assert a <= r["best_start"] <= a + 1 and b - 1 <= r["best_end"] <= b
    truth = np.zeros(ms.SLOTS, dtype=bool)
    truth[a : b + 1] = True
    p, req, c = mx.energy_terms(y, truth, r["best_start"], r["best_end"])
    assert c / (p + req - c) > 0.98 and c / p > 0.999


def test_gap_edge_equals_nearest_without_gaps():
    y, s, _, _ = fixture_days()["clean"]
    u0 = ms.reconstruct_uncorrected(y, s)
    for got, want in zip(ms.bridge_residual_matrices(u0, y, "sq", "gap_edge"), ms.bridge_residual_matrices(u0, y, "sq", "nearest"), strict=True):
        assert np.array_equal(got, want, equal_nan=True)


# --------------------------------------------------------------------------- (c) gap immediately before the span

def test_gap_edge_anchors_only_the_gap_side():
    y, s, a, b = fixture_days()["gap_before"]
    i, j = a - ms.SCAN_START, b - ms.SCAN_START
    # Anchor slots by definition: nearest reaches back past the gap, gap_edge stops at the
    # window's own start, and both keep the adjacent reading on the far side.
    assert anchor_slots(y, s, a, b, "nearest") == (a - 2, b + 1)
    assert anchor_slots(y, s, a, b, "gap_edge") == (a, b + 1)
    assert anchor_slots(y, s, a, b, "edge") == (a, b)
    u0 = ms.reconstruct_uncorrected(y, s)
    rules = {k: ms.bridge_residual_matrices(u0, y, "sq", k) for k in ms.ANCHORS}
    for k, (left, right) in {"nearest": (a - 2, b + 1), "gap_edge": (a, b + 1), "edge": (a, b)}.items():
        ru, rc, n = misfit_between(y, s, a, b, left, right)
        rss_u, rss_c, length = rules[k]
        assert rss_u[i, j] == pytest.approx(ru, rel=1e-9) and rss_c[i, j] == pytest.approx(rc, rel=1e-9)
        assert length[i, j] == n
    assert [rules[k][2][i, j] for k in ("nearest", "gap_edge", "edge")] == [b - a + 1, b - a, b - a - 1]
    # Only the windows that touch the gap differ from nearest: those starting at a and
    # those ending at a - 2. Everything else is bit-identical.
    starts = ms.SCAN_START + np.arange(ms.N_WINDOWS)
    touched = (starts == a)[:, None] | (starts == a - 2)[None, :]
    for near, gap in zip(rules["nearest"], rules["gap_edge"], strict=True):
        assert np.array_equal(near[~touched], gap[~touched], equal_nan=True)
    assert not np.array_equal(rules["nearest"][0][i], rules["gap_edge"][0][i], equal_nan=True)
    # End to end, the planted window is still found beside the gap.
    r = ms.score_siteday(y, s, FLOOR, anchors="gap_edge", **FROZEN)
    assert r["input_ok"] and r["best_score"] > 0
    assert a <= r["best_start"] <= a + 1 and b - 1 <= r["best_end"] <= b
