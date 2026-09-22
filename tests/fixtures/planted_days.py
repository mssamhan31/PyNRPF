"""Synthetic site-days with a known answer, for the step tests.

A smooth demand curve and a solar bell are combined into a true net load; a wrong sign is
planted by taking the absolute value inside the export span, exactly how the Alpha dataset
was built. Every function returns ``(y, s, truth)`` with ``truth`` the planted slots.
"""

from __future__ import annotations

import numpy as np

SLOTS = 96


def base_day(demand_level: float = 8.0, solar_peak: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """A smooth demand day (MW) and a solar bell (MW) peaking at noon."""
    t = np.arange(SLOTS) / 4.0
    demand = demand_level + 1.5 * np.sin((t - 6) / 24 * 2 * np.pi)
    solar = solar_peak * np.clip(np.sin((t - 6) / 12 * np.pi), 0.0, None) ** 2
    return demand, solar


def clean_flip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A day whose midday export was stored as an import: recorded = |true|."""
    demand, solar = base_day()
    true = demand - solar
    truth = true < 0
    return np.abs(true), solar, truth


def clean_day() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A day with no export at all: nothing to flip."""
    demand, solar = base_day(demand_level=15.0, solar_peak=6.0)
    y = demand - solar
    return y, solar, np.zeros(SLOTS, dtype=bool)


def flip_with_gap_before() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The clean flip with the reading just before the span missing."""
    y, s, truth = clean_flip()
    start = int(np.flatnonzero(truth)[0])
    y = y.copy()
    y[start - 1] = np.nan
    return y, s, truth
