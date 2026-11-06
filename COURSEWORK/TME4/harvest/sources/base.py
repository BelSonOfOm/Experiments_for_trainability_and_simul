"""What every source of problems must provide.

A *source* is anything that produces candidate problems: a generator that runs offline
(SAT formulas, spin chains) or an adapter to a public catalogue (TUDataset, MOABB, ...).
Adding a new catalogue = writing one new Source; nothing else in the pipeline changes.

The pipeline talks to a source in three steps:

    1. list_datasets()      -> the datasets it offers, as DatasetMeta (no download)
    2. fetch(meta)          -> the raw data of one dataset (download + hash; generators: nothing)
    3. drafts(meta, raw)    -> one ProblemDraft per candidate problem in that dataset
                               (e.g. one per vertex count n, one per subject, one per class)

A ProblemDraft is a problem *before* the pipeline's checks: instances with labels, plus
everything the source knows about how they were made.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..records import Equation, Provenance


@dataclass
class ProblemDraft:
    """A candidate problem, as produced by a source, before deduplication, balancing,
    splitting and checks.

    task_name            short name of the task inside its dataset, e.g. "satisfiable" or
                         "class 2 vs rest"; becomes part of the task id
    problem_type         GRAPH / COVARIANCE / DIAGONAL / QUANTUM
    n                    number of sites
    instances            list of Instance
    label_rule           how the labels were obtained, as a sentence
    equation             ℰₙ without the check fields (the pipeline fills them)
    provenance           where it comes from
    sites_have_identity  True if site k means the same thing in every instance (EEG channel,
                         chain site); False if the numbering is arbitrary (graph vertices,
                         SAT variables). Decides how M̄ is built (task_operator.py).
    """
    task_name: str
    problem_type: str
    n: int
    instances: list
    label_rule: str
    equation: Equation
    provenance: Provenance
    sites_have_identity: bool
    notes: dict = field(default_factory=dict)


class Source(ABC):
    """The interface every source implements (see the module docstring)."""

    name: str = "unnamed source"

    @abstractmethod
    def list_datasets(self) -> list:
        """Return the datasets this source offers, as a list of DatasetMeta."""

    @abstractmethod
    def fetch(self, meta):
        """Return the raw data of one dataset (whatever drafts() needs), or None for generators.
        Sources that download must record a sha256 hash of what they downloaded."""

    @abstractmethod
    def drafts(self, meta, raw):
        """Yield ProblemDraft objects: the candidate problems inside one dataset."""
