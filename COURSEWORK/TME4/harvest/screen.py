"""The screening rules S1 to S7: keep or reject, always with a reason.

S1 to S4 are checked on a dataset's *metadata*, before anything is downloaded:
    S1  input type is one we can turn into an operator (graph, multichannel signal / SPD matrix,
        Hamiltonian, CNF formula)
    S2  labels are binary, or a stated rule makes them binary (one-vs-rest, threshold)
    S3  size: after the source's reduction (vertex count, channel selection, ...) the number
        of sites can be in [n_min, n_max]
    S4  the licence allows research use

S5 to S7 are checked on the converted *instances* of one candidate problem:
    S5  at least m_min samples (after balancing)
    S6  at least min_distinct different inputs
    S7  conflict rate at most max_conflict

A rejection is not a failure of the code: the funnel of rejections is reported as a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import HarvestConfig

SUPPORTED_INPUTS = {"GRAPH", "COVARIANCE", "DIAGONAL", "QUANTUM"}


@dataclass
class DatasetMeta:
    """What a source says about one dataset before downloading it.

    name            the dataset's name in its catalogue
    problem_type    which of our four types it would become (None if it fits none)
    n_samples       how many samples the dataset has (approximately is fine)
    label_kind      "binary", "multiclass", "regression" or "none"
    sites_possible  the numbers of sites the dataset can produce after reduction
    licence         licence string ("" if unknown)
    url             where it is downloaded from
    notes           anything else (free text)
    """
    name: str
    problem_type: str | None
    n_samples: int
    label_kind: str
    sites_possible: list
    licence: str = ""
    url: str = ""
    notes: dict = field(default_factory=dict)


@dataclass
class ScreenResult:
    """accepted: True or False. reason: which rule decided, in words."""
    accepted: bool
    reason: str


RESEARCH_OK_LICENCES = ("cc-by", "cc0", "mit", "bsd", "apache", "odc", "public", "generated", "research")


def screen_metadata(meta: DatasetMeta, cfg: HarvestConfig) -> ScreenResult:
    """WHAT: apply S1 to S4 to one dataset's metadata, in order; the first failing rule rejects.
    WHY: never download what cannot fit."""
    if meta.problem_type not in SUPPORTED_INPUTS:
        return ScreenResult(False, f"S1: input type {meta.problem_type!r} not supported")
    if meta.label_kind not in ("binary", "multiclass", "regression"):
        return ScreenResult(False, f"S2: labels are {meta.label_kind!r}")
    if not any(cfg.n_min <= n <= cfg.n_max for n in meta.sites_possible):
        return ScreenResult(False, f"S3: no site count in [{cfg.n_min}, {cfg.n_max}] "
                                   f"(possible: {meta.sites_possible})")
    licence = meta.licence.lower()
    if licence and not any(word in licence for word in RESEARCH_OK_LICENCES):
        return ScreenResult(False, f"S4: licence {meta.licence!r} not recognised as research-use")
    return ScreenResult(True, "S1-S4 passed" + ("" if licence else " (licence unknown: check by hand)"))


def screen_instances(n_samples: int, n_distinct: int, conflict_rate: float,
                     cfg: HarvestConfig) -> ScreenResult:
    """WHAT: apply S5 to S7 to one candidate problem's converted instances.
    WHY: catches problems that look fine on paper but have too few, too repetitive or
    contradictory samples (the IMDB case)."""
    if n_distinct < cfg.min_distinct:
        return ScreenResult(False, f"S6: only {n_distinct} distinct inputs (< {cfg.min_distinct})")
    if conflict_rate > cfg.max_conflict:
        return ScreenResult(False, f"S7: conflict rate {conflict_rate:.2f} (> {cfg.max_conflict})")
    if n_samples < cfg.m_min:
        return ScreenResult(False, f"S5: {n_samples} samples after balancing (< {cfg.m_min})")
    return ScreenResult(True, "S5-S7 passed")
