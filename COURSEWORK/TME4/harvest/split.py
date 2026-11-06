"""The fixed TRAIN / TEST split.

Two requirements:
  1. stratified: each class keeps (about) the same proportion on both sides;
  2. by unit: all samples of one unit (e.g. crops of one EEG trial, the two members of a
     SAT/UNSAT pair) land on the same side, so nothing derived from a TEST original is seen in
     TRAIN (principle P4 of docs/HARVEST_SCALE.md).

The split is made once, with a recorded seed, and stored. Every later stage (task operator,
structure extraction, training) uses this same split.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def split_by_unit(instances: list, test_fraction: float, seed: int) -> list:
    """WHAT: assign "TRAIN" or "TEST" to every instance.
    HOW:
      1. group the instances by unit;
      2. give each unit the label most of its samples have;
      3. for each label separately, shuffle that label's units (with the seed) and put the
         first round(test_fraction * number of units) of them in TEST, the rest in TRAIN;
      4. every sample inherits its unit's side.
    OUTPUT: a list of "TRAIN"/"TEST", in the order of `instances`."""
    rng = np.random.default_rng(seed)
    samples_of_unit = defaultdict(list)
    for index, inst in enumerate(instances):
        samples_of_unit[inst.unit].append(index)

    units_of_label = defaultdict(list)
    for unit, indices in samples_of_unit.items():
        majority_label = Counter(instances[i].y for i in indices).most_common(1)[0][0]
        units_of_label[majority_label].append(unit)

    side_of_unit = {}
    for label in sorted(units_of_label):
        units = sorted(units_of_label[label])
        rng.shuffle(units)
        n_test = int(round(test_fraction * len(units)))
        for k, unit in enumerate(units):
            side_of_unit[unit] = "TEST" if k < n_test else "TRAIN"

    return [side_of_unit[inst.unit] for inst in instances]
