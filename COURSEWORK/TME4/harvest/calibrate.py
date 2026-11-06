"""Choosing a generator's free parameter so that the two classes come out balanced (docs/HARVEST_V2.md §1.3).

Example: for random graphs from the Erdős–Rényi model, the label 'connected' is almost always
false when the edge probability p is small and almost always true when p is large. Somewhere in
between, half the graphs are connected. That p is what we want, and it depends on n.

Rule (stated, automatic, never tuned by hand):
  1. for each value θ of a fixed grid, draw `n_draws` inputs with a *calibration* random stream;
  2. measure the fraction of +1 labels at each θ;
  3. keep the θ whose fraction is closest to 1/2;
  4. if even that fraction is outside [low, high], the setting cannot be balanced: return None.

The calibration stream is seeded with ("calibration", seed, ...) and the data stream with
("data", seed, ...), so calibration draws are never reused as data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    """WHAT: the result of one calibration.
    FIELDS: theta = the chosen parameter value; fraction = the fraction of +1 labels measured at that
    value; table = [(θ, fraction), ...] for every grid value tried (stored in the provenance, so anyone
    can see why that θ was chosen).
    WHY: the choice of θ is part of how the data were made, so it must be recorded, not just used."""
    theta: object
    fraction: float
    table: list


def calibration_rng(seed: int, *key) -> np.random.Generator:
    """WHAT: the random stream used only for calibration. `key` identifies the setting
    (e.g. model, label, n) as integers."""
    return np.random.default_rng([seed, 7_777_777, *key])


def data_rng(seed: int, *key) -> np.random.Generator:
    """WHAT: the random stream used for the data of a setting (different from calibration_rng)."""
    return np.random.default_rng([seed, 1_111_111, *key])


def calibrate(draw, label, grid, rng, n_draws: int = 100, low: float = 0.2,
              high: float = 0.8):
    """WHAT: pick the grid value θ that makes label(draw(θ)) closest to balanced.
    INPUTS:
      draw(theta, rng) -> one input, or None if θ is impossible here (e.g. n·d odd for a
                          d-regular graph): θ is then skipped
      label(x) -> bool
      grid: the candidate θ values, in a fixed order
    OUTPUT: Calibration, or None when no θ gives a fraction in [low, high]."""
    table = []
    for theta in grid:
        first = draw(theta, rng)
        if first is None:
            continue
        values = [label(first)] + [label(draw(theta, rng)) for _ in range(n_draws - 1)]
        table.append((theta, float(np.mean(values))))
    if not table:
        return None
    theta, fraction = min(table, key=lambda row: abs(row[1] - 0.5))
    if not low <= fraction <= high:
        return None
    return Calibration(theta, fraction, table)
