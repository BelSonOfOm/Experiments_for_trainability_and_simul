"""Labels set at the median of the TRAIN scores (docs/HARVEST_V2.md §1.1; C1 in NEXT_STEPS_JULIA_KAUE.md).

Some sources produce a *number* per input rather than a yes/no answer: a graph's algebraic
connectivity, a formula's number of satisfying assignments, a molecule's correlation energy.
Such a number is called the instance's *score*. It becomes a ±1 label by a cut:
"+1 when the score is above the median".

Why the median of TRAIN only: if TEST scores helped choose the cut, information from the test
set would be inside the labels (a leak). So the order is: split first, then compute the cut on
TRAIN, then label everyone.

A median cut gives balanced classes by construction (half above, half below), except when many
scores are equal to the median (*ties*: integer scores pile up). Ties are not fixed by hand:
if the TRAIN balance comes out outside the config range, the problem is rejected with the
reason 'ties at median'.
"""
from __future__ import annotations

import numpy as np

from .records import Instance


def train_median(instances: list, split: list) -> float:
    """WHAT: the median of the scores (instance.info["score"]) of the TRAIN instances.
    WHY: the cut must be blind to TEST."""
    scores = [inst.info["score"] for inst, side in zip(instances, split) if side == "TRAIN"]
    if not scores:
        raise ValueError("no TRAIN instances")
    return float(np.median(scores))


def label_by_train_median(instances: list, split: list) -> tuple:
    """WHAT: relabel every instance: +1 when its score is strictly above the TRAIN median, else -1.
    INPUTS: instances carrying info["score"]; split = "TRAIN"/"TEST" per instance.
    OUTPUT: (new list of Instance, rule sentence, fraction of +1 among TRAIN).
    NOTE: new Instance objects are made; the source's objects are not modified."""
    cut = train_median(instances, split)
    relabelled = [Instance(x=inst.x, y=1 if inst.info["score"] > cut else -1, unit=inst.unit,
                           info={**inst.info, "cut": cut}) for inst in instances]
    train_labels = [inst.y for inst, side in zip(relabelled, split) if side == "TRAIN"]
    train_balance = sum(1 for y in train_labels if y == 1) / len(train_labels)
    return relabelled, f"score > TRAIN median ({cut:.6g})", train_balance
