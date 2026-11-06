"""The numbers that decide what the harvest keeps.

Every threshold lives here, in one place, so the group can discuss them without reading code.
Each value is a proposal from docs/HARVEST_SCALE.md and docs/SUBFAMILIES.md, not a decision.
"""
from dataclasses import dataclass


@dataclass
class HarvestConfig:
    # --- size of a problem -------------------------------------------------------------
    n_min: int = 4
    """Smallest number of sites (= qubits). Below 4, almost every structure is trivial."""

    n_max: int = 12
    """Largest number of sites. 12 qubits = state vectors of 4096 numbers: exact simulation
    on a laptop is still easy."""

    # --- how many samples a problem needs -------------------------------------------------
    m_min: int = 100
    """Minimum number of samples (after deduplication and balancing). Below this, the
    TEST set is too small to measure an accuracy."""

    # --- class balance ---------------------------------------------------------------------
    balance_low: float = 0.2
    balance_high: float = 0.8
    """The fraction of +1 labels must lie in [balance_low, balance_high]. Outside this range
    the majority class is downsampled (randomly thrown away) until both classes are equal."""

    # --- screening rules S6, S7 (new on 5 Oct, from the IMDB finding) ------------------------
    min_distinct: int = 50
    """S6: a problem needs at least this many *different* inputs. IMDB-MULTI at n = 7 had
    332 graphs but only 10 different ones."""

    max_conflict: float = 0.2
    """S7: the fraction of samples whose exact input also appears with the other label must
    be at most this. Above it, the label is not a function of the input we keep."""

    # --- the TRAIN / TEST split ------------------------------------------------------------
    test_fraction: float = 0.3
    """Fraction of samples put in TEST (stratified: same class proportions on both sides)."""

    # --- the task operator M̄ --------------------------------------------------------------
    support_threshold: float = 0.1
    """An entry of M̄ counts as 'present' (part of the support = locality) only if it is non-zero
    in at least this fraction of TRAIN instances. Without a threshold, noise switches every
    entry on."""

    partial_correlation_cutoff: float = 0.05
    """COVARIANCE only: a precision matrix has (almost) no exact zeros, so 'non-zero' says
    nothing. A channel pair (i, j) is in the support when the TRAIN-mean precision gives a
    partial correlation |-Q_ij / sqrt(Q_ii Q_jj)| of at least this value. Placeholder: the
    planned method is the graphical lasso (docs/HARVEST_SCALE.md §3.3)."""

    # --- merging problems that are the same problem ---------------------------------------
    near_duplicate_agreement: float = 0.95
    """Two problems on the same inputs whose labels agree on more than this fraction of the
    shared inputs (or disagree on more than this fraction: a flipped label is the same
    problem) are merged."""

    # --- reproducibility -------------------------------------------------------------------
    seed: int = 0
    """Every random choice in the harvest is derived from this seed."""
