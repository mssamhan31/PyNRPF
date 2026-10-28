"""One test per step of M9 on planted days: each step keeps its meaning."""

import numpy as np
import pytest
from fixtures.planted_days import SLOTS, base_day, clean_day, clean_flip, flip_with_gap_before

from pynrpf.m9 import bridge, edges, evidence, misfit, stories, windows, winner
from pynrpf.m9.winner import NO_CORRECTION, Candidate


def test_step1_stories_agree_outside_and_flip_inside():
    y, s, truth = clean_flip()
    kept, flipped = stories.demand_if_kept(y, s), stories.demand_if_flipped(y, s)
    assert np.allclose(kept, s + y)
    assert np.allclose(flipped, s - y)
    # inside the export span the flipped story recovers the smooth demand, the kept story does not
    demand, _ = base_day()
    assert np.allclose(flipped[truth], demand[truth])
    assert not np.allclose(kept[truth], demand[truth])


def test_step2_candidate_count_and_admissibility():
    y, s, _ = clean_flip()
    adm = windows.admissible(y, s)
    assert adm.shape == (48, 48)
    assert adm.sum() == windows.N_CANDIDATES == 1176      # every window admissible on a clean day
    a, b = windows.grid()
    assert not adm[b < a].any()
    y_gap = y.copy()
    y_gap[40] = np.nan                                     # slot 40 missing: windows containing it drop out
    adm_gap = windows.admissible(y_gap, s)
    assert not adm_gap[(a <= 40) & (b >= 40)].any()
    assert adm_gap[((b < 40) | (a > 40)) & (b >= a)].all()


def test_step2_nearest_finite_skips_gaps():
    y, s, _ = clean_flip()
    y[50:53] = np.nan
    before, after = windows.nearest_finite(y, s)
    assert before[51] == 49 and after[51] == 53
    assert before[0] == -1 and after[95] == -1


def test_step3_edges_sit_at_local_minima_with_inward_tolerance_and_gap_exemption():
    y, s, truth = clean_flip()
    start, end = int(np.flatnonzero(truth)[0]), int(np.flatnonzero(truth)[-1])
    minima = edges.local_minima(y)
    assert minima[start] or minima[start - 1] or minima[start + 1]   # the cusp sits at the crossing
    start_ok, end_ok = edges.valid_edges(y, s)
    assert start_ok[start - windows.SCAN_START]
    assert end_ok[end - windows.SCAN_START]
    assert not start_ok[start + 6 - windows.SCAN_START]              # the rising flank is not a minimum
    y_gap, s_gap, _ = flip_with_gap_before()
    start_ok_gap, _ = edges.valid_edges(y_gap, s_gap)
    assert start_ok_gap[start - windows.SCAN_START]                  # exempt beside the gap


def test_step4_bridge_is_a_straight_line_between_the_anchors():
    kept = np.linspace(0.0, 95.0, SLOTS)                             # demand already a straight line
    before, after = bridge.anchors(kept)
    slots = np.arange(30, 40)
    ends = np.array([35, 39])
    line = bridge.line(29, kept[29], after[ends], kept[after[ends]], slots)
    assert np.allclose(line[0], kept[slots])                         # the bridge of a line is the line
    assert line.shape == (2, 10)


def test_step5_flipped_story_fits_the_bridge_on_the_planted_window():
    y, s, truth = clean_flip()
    kept, flipped = stories.demand_if_kept(y, s), stories.demand_if_flipped(y, s)
    before, after = bridge.anchors(kept)
    rss_u, rss_c, length = misfit.residual_matrices(kept, flipped, before, after)
    a, b = int(np.flatnonzero(truth)[0]) - windows.SCAN_START, int(np.flatnonzero(truth)[-1]) - windows.SCAN_START
    assert length[a, b] == b - a + 1
    assert rss_c[a, b] < rss_u[a, b] / 100                           # the flip explains the hump
    assert np.isnan(rss_u[5, 2])                                     # end before start: undefined


def test_step6_evidence_sign_and_floor():
    rss_u = np.array([[4.0, 1.0]])
    rss_c = np.array([[1.0, 4.0]])
    length = np.array([[2.0, 2.0]])
    r = evidence.evidence_matrix(rss_u, rss_c, length, phi=0.0, admissible=np.ones((1, 2), dtype=bool))
    assert r[0, 0] == pytest.approx(np.log(4.0))                     # (L/2) log 4 with L = 2
    assert r[0, 1] == pytest.approx(-np.log(4.0))                    # symmetric
    ones = np.ones((1, 1))
    both_perfect = evidence.evidence_matrix(np.zeros((1, 1)), np.zeros((1, 1)), ones, 1e-3, ones.astype(bool))
    assert both_perfect[0, 0] == 0.0                                 # the floor makes two perfect fits tie
    masked = evidence.evidence_matrix(rss_u, rss_c, length, 0.0, np.zeros((1, 2), dtype=bool))
    assert np.isneginf(masked).all()


def test_step7_tie_order_and_runner_up():
    r = np.full((48, 48), -np.inf)
    r[10, 20] = 5.0
    r[10, 22] = 5.0                                                  # same evidence, longer: loses
    r[12, 20] = 5.0                                                  # same evidence, shorter: wins
    r[30, 34] = 3.0                                                  # non-overlapping runner-up
    best, runner = winner.rank(r)
    assert (best.start, best.end) == (windows.SCAN_START + 12, windows.SCAN_START + 20)
    assert (runner.start, runner.end) == (windows.SCAN_START + 30, windows.SCAN_START + 34)
    negative = np.full((48, 48), -np.inf)
    negative[3, 5] = -0.5
    best, runner = winner.rank(negative)
    assert best == NO_CORRECTION and runner.evidence == -0.5
    assert winner.best_evidence(best, runner) == -0.5


def test_candidate_mask_and_length():
    c = Candidate(10, 12, 1.0)
    assert c.length == 3 and c.mask()[10:13].all() and c.mask().sum() == 3
    assert NO_CORRECTION.length == 0 and not NO_CORRECTION.mask().any()


def test_planted_flip_is_found_end_to_end():
    from pynrpf.m9 import score_siteday
    y, s, truth = clean_flip()
    score = score_siteday(y, s)
    span = np.flatnonzero(truth)
    assert score.input_ok and score.evidence > 0
    assert abs(score.winner.start - span[0]) <= 1 and abs(score.winner.end - span[-1]) <= 1
    y0, s0, _ = clean_day()
    assert score_siteday(y0, s0).winner.is_null
