"""M̄: the single operator of the whole task, built from the TRAIN instances only.

Why TRAIN only: M̄ decides the circuit's generators. If TEST instances shaped it, the circuit
would have seen the test set before being tested.

Two cases (docs/HARVEST_SCALE.md §4.2):

1. Sites have an identity (EEG channel 'C3' is the same channel in every trial; site 4 of a
   spin chain is the same site for every parameter value). Then averaging is meaningful:
       M̄ = mean of M(x) over TRAIN,
   and the support (locality) is the set of entries present in at least a fraction
   `support_threshold` of TRAIN instances.

2. Sites have no identity (graph vertices and SAT variables are numbered arbitrarily: vertex 3
   of one molecule has nothing to do with vertex 3 of another). Then a plain average depends on
   arbitrary numberings and means nothing. The only numbering-independent average is the
   average over *all* renumberings. For a matrix this keeps two numbers: the mean diagonal
   entry a and the mean off-diagonal entry b, giving M̄ = a I + b (J - I) (J = all-ones matrix).
   For a graph Laplacian without potentials this is (mean edge count / C(n,2)) times the
   Laplacian of the complete graph. FORCED RESULT: for graphs the task operator only knows the
   edge density; all structure enters through the encoding of each instance.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from math import comb

import numpy as np

from .config import HarvestConfig
from .pauli import support_qubits
from .records import TaskOperator


def _train_only(items: list, split: list) -> list:
    """The items whose split entry is TRAIN."""
    return [item for item, side in zip(items, split) if side == "TRAIN"]


# ----------------------------------------------------------------------------------------
# Case 1: sites have an identity
# ----------------------------------------------------------------------------------------

def mean_matrix_operator(operators: list, kind: str, cfg: HarvestConfig) -> TaskOperator:
    """WHAT: M̄ = mean of the TRAIN matrices; support = site pairs present often enough.
    HOW (support):
      - LAPLACIAN: pair (i, j) is present in an instance when M_ij != 0; kept when present in
        at least cfg.support_threshold of TRAIN instances.
      - PRECISION: every entry is non-zero, so instead a pair is kept when the partial
        correlation of the mean, |-M̄_ij / sqrt(M̄_ii M̄_jj)|, is at least
        cfg.partial_correlation_cutoff (placeholder for the graphical lasso)."""
    matrices = np.array([op.matrix for op in operators])
    mean = matrices.mean(axis=0)
    n = mean.shape[0]
    support = []
    for i, j in itertools.combinations(range(n), 2):
        if kind == "PRECISION":
            partial = -mean[i, j] / np.sqrt(mean[i, i] * mean[j, j])
            if abs(partial) >= cfg.partial_correlation_cutoff:
                support.append((i, j))
        else:
            frequency = np.mean(np.abs(matrices[:, i, j]) > 1e-12)
            if frequency >= cfg.support_threshold:
                support.append((i, j))
    rule = (f"mean of {len(operators)} TRAIN matrices; support: "
            + ("|partial correlation| >= %g" % cfg.partial_correlation_cutoff if kind == "PRECISION"
               else "non-zero in >= %g of TRAIN" % cfg.support_threshold))
    return TaskOperator(kind=kind, matrix=mean, support=support, rule=rule)


def mean_pauli_operator(operators: list, kind: str, cfg: HarvestConfig) -> TaskOperator:
    """WHAT: M̄ = mean of the TRAIN Pauli sums (a term missing from an instance counts as 0);
    support = Pauli strings present in at least cfg.support_threshold of TRAIN instances."""
    total = defaultdict(float)
    count = defaultdict(int)
    for op in operators:
        for string, coefficient in op.pauli.items():
            total[string] += float(np.real(coefficient))
            count[string] += 1
    m = len(operators)
    mean = {s: total[s] / m for s in total if abs(total[s]) > 1e-14}
    support = sorted(s for s in count if count[s] / m >= cfg.support_threshold)
    return TaskOperator(kind=kind, pauli=mean, support=support,
                        rule=f"mean of {m} TRAIN Pauli sums; support: present in >= "
                             f"{cfg.support_threshold:g} of TRAIN")


# ----------------------------------------------------------------------------------------
# Case 2: sites have no identity
# ----------------------------------------------------------------------------------------

def symmetrised_matrix_operator(operators: list, kind: str) -> TaskOperator:
    """WHAT: the average of the TRAIN matrices over all renumberings of the sites:
    M̄ = a I + b (J - I), a = mean diagonal entry, b = mean off-diagonal entry.
    WHY a and b: renumbering moves every diagonal entry to every diagonal position and every
    off-diagonal entry to every off-diagonal position with equal frequency, so the average
    over renumberings is constant on the diagonal and constant off it."""
    matrices = np.array([op.matrix for op in operators])
    n = matrices.shape[1]
    off_diagonal = ~np.eye(n, dtype=bool)
    a = float(np.mean([M[np.eye(n, dtype=bool)].mean() for M in matrices]))
    b = float(np.mean([M[off_diagonal].mean() for M in matrices]))
    mean = a * np.eye(n) + b * (np.ones((n, n)) - np.eye(n))
    support = list(itertools.combinations(range(n), 2)) if abs(b) > 1e-14 else []
    return TaskOperator(kind=kind, matrix=mean, support=support,
                        rule=f"sites have no identity: average over all renumberings "
                             f"= {a:.4g} I + {b:.4g} (J - I)")


def all_arrangements(n: int, letters: tuple) -> list:
    """WHAT: every Pauli string on n qubits whose non-identity letters are exactly `letters`
    (as a multiset), in every possible arrangement.
    EXAMPLE: all_arrangements(3, ("Z", "Z")) == ["ZZI", "ZIZ", "IZZ"] (in some order)."""
    strings = set()
    for positions in itertools.combinations(range(n), len(letters)):
        for order in set(itertools.permutations(letters)):
            word = ["I"] * n
            for position, letter in zip(positions, order):
                word[position] = letter
            strings.add("".join(word))
    return sorted(strings)


def symmetrised_pauli_operator(operators: list, kind: str, cfg: HarvestConfig) -> TaskOperator:
    """WHAT: the average of the TRAIN Pauli sums over all renumberings of the qubits.
    HOW: renumbering only rearranges the letters of a string. So strings are grouped by their
    multiset of non-identity letters (e.g. {Z, Z, Z}); within a group, the averaged coefficient
    is the same for every arrangement: (sum of the group's coefficients, averaged over TRAIN)
    divided by the number of arrangements.
    SUPPORT: a group is kept when at least cfg.support_threshold of TRAIN instances contain a
    string of that group; its support is then all its arrangements."""
    n = operators[0].n
    m = len(operators)
    group_total = defaultdict(float)
    group_presence = defaultdict(int)
    for op in operators:
        seen = set()
        for string, coefficient in op.pauli.items():
            group = tuple(sorted(string[q] for q in support_qubits(string)))
            group_total[group] += float(np.real(coefficient))
            seen.add(group)
        for group in seen:
            group_presence[group] += 1
    mean, support = {}, []
    for group, total in group_total.items():
        arrangements = all_arrangements(n, group)
        value = total / m / len(arrangements)
        if abs(value) > 1e-14:
            for s in arrangements:
                mean[s] = value
        if group_presence[group] / m >= cfg.support_threshold:
            support.extend(arrangements)
    return TaskOperator(kind=kind, pauli=mean, support=sorted(support),
                        rule="sites have no identity: average over all renumberings "
                             "(one coefficient per letter multiset)")


# ----------------------------------------------------------------------------------------
# One entry point
# ----------------------------------------------------------------------------------------

def task_operator(operators: list, split: list, sites_have_identity: bool,
                  cfg: HarvestConfig) -> TaskOperator:
    """WHAT: M̄ for a problem, from the TRAIN operators, choosing the right case.
    INPUTS: operators = M(x) of every instance; split = "TRAIN"/"TEST" per instance;
            sites_have_identity = declared by the source (it knows whether its sites are named)."""
    train = _train_only(operators, split)
    kind = train[0].kind
    uses_matrix = train[0].matrix is not None
    if sites_have_identity:
        return mean_matrix_operator(train, kind, cfg) if uses_matrix else mean_pauli_operator(train, kind, cfg)
    return symmetrised_matrix_operator(train, kind) if uses_matrix else symmetrised_pauli_operator(train, kind, cfg)


def complete_graph_laplacian(n: int) -> np.ndarray:
    """The Laplacian of the complete graph Kₙ: n I - J. Used by the tests: for unweighted graphs
    without potentials, the symmetrised M̄ must equal (mean edge count / C(n, 2)) * L(Kₙ)."""
    return n * np.eye(n) - np.ones((n, n))


def expected_symmetrised_graph_operator(mean_edge_count: float, n: int) -> np.ndarray:
    """(mean edge count / C(n,2)) * L(Kₙ): the forced result for graphs (see module docstring)."""
    return mean_edge_count / comb(n, 2) * complete_graph_laplacian(n)
