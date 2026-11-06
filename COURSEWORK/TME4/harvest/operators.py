"""operator_of: the operator M(x) of one input x.

Each problem type has one natural operator (docs/DE_BRIDGE.md §2, docs/HARVEST_SCALE.md §4.1):

    GRAPH       M = Bᵀ W B + V        the graph Laplacian (+ an optional diagonal 'potential')
    COVARIANCE  M = R⁻¹               the precision matrix of the trial's correlation matrix R
    QUANTUM     M = H                 the Hamiltonian, as a Pauli sum
    DIAGONAL    M = Σ_clauses P_c     P_c = projector onto 'clause c is violated' (SAT)

For GRAPH and COVARIANCE the matrix is also stored in factorised form (B, w, V): this is the
discrete differential form of docs/HARVEST_SCALE.md §3.2.

    B  the oriented incidence matrix: one row per edge e = (i, j), with +1 in column j
       (the edge's head) and -1 in column i (its tail). Applied to a vector x on the sites,
       (B x)_e = x_j - x_i is the difference along the edge: B is the discrete gradient,
       also called the coboundary d₀.
    W  diagonal matrix of edge weights w_e (the Hodge star ⋆₁: an inner product on edges).
    V  diagonal matrix of site values (a 'potential').

Then Bᵀ W B is the weighted graph Laplacian: (Bᵀ W B)_ii = Σ_e∋i w_e, (Bᵀ W B)_ij = -w_ij.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.covariance import ledoit_wolf

from .pauli import add_term, pauli_string
from .records import Operator


# ----------------------------------------------------------------------------------------
# GRAPH
# ----------------------------------------------------------------------------------------

def incidence_matrix(n: int, edges: list) -> np.ndarray:
    """WHAT: the oriented incidence matrix B (one row per edge, one column per site).
    HOW: for edge number e = (i, j): B[e, i] = -1, B[e, j] = +1. The orientation (which end is
    the head) is arbitrary: the Laplacian Bᵀ W B does not depend on it.
    WHY: B is the coboundary d₀; storing it makes the Laplacian a derived object."""
    B = np.zeros((len(edges), n))
    for e, (i, j) in enumerate(edges):
        B[e, i] = -1.0
        B[e, j] = +1.0
    return B


def graph_operator(n: int, edges: list, weights=None, potentials=None) -> Operator:
    """WHAT: the operator of a graph, M = Bᵀ W B + V.
    INPUTS: n sites; edges as pairs (i, j) with 0 <= i, j < n (each edge listed once);
            weights: one number per edge (default 1); potentials: one number per site
            (default 0; later, e.g., one value per atom type for molecules).
    OUTPUT: Operator(kind="LAPLACIAN") with the matrix and the factors B, w, V.
    WHY: the graph's heat equation dx/dt = -M x is the graph problem's equation ℰₙ."""
    edges = [tuple(e) for e in edges]
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, dtype=float)
    V = np.zeros(n) if potentials is None else np.asarray(potentials, dtype=float)
    B = incidence_matrix(n, edges)
    M = B.T @ np.diag(w) @ B + np.diag(V)
    return Operator(kind="LAPLACIAN", n=n, matrix=M,
                    factors={"B": B, "w": w, "V": V, "edges": edges})


# ----------------------------------------------------------------------------------------
# COVARIANCE
# ----------------------------------------------------------------------------------------

def unusable_channels(trial) -> list:
    """WHAT: the channels of a recording (channels x time) that cannot enter a correlation matrix:
    a channel containing a missing or infinite value (NaN, inf), or a channel whose value never
    changes (zero variance: its correlation with anything is 0/0).
    WHY: such a recording has no correlation matrix and no precision operator. Sources drop it by
    this stated rule and record how many they dropped; recordings are never patched by hand."""
    trial = np.asarray(trial, dtype=float)
    bad = []
    for k in range(trial.shape[0]):
        if not np.all(np.isfinite(trial[k])) or np.std(trial[k]) == 0.0:
            bad.append(k)
    return bad


def correlation_from_trial(trial: np.ndarray) -> tuple:
    """WHAT: the correlation matrix R of one recorded trial (channels x time).
    HOW:
      1. Ledoit-Wolf covariance C: the ordinary sample covariance pulled towards a multiple
         of the identity by an amount chosen from the data. Why: with few time points the
         ordinary estimate is noisy and can be nearly singular (not invertible); the pull
         keeps it invertible and stable.
      2. R = D^(-1/2) C D^(-1/2), D = diagonal of C: divide each channel by its own spread,
         so that a channel's amplitude (how loud it is) does not matter, only how channels
         vary together.
    OUTPUT: (R, C, shrinkage amount)."""
    samples = np.asarray(trial, dtype=float).T          # ledoit_wolf wants (time, channels)
    C, shrinkage = ledoit_wolf(samples)
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    return R, C, float(shrinkage)


def covariance_operator(trial: np.ndarray) -> Operator:
    """WHAT: the operator of one trial: the precision matrix Q = R⁻¹.
    WHY: Q is the operator of the Gaussian density exp(-½ xᵀ Q x), and the drift of the
    Ornstein-Uhlenbeck equation whose stationary covariance is R (docs/HARVEST_SCALE.md §3.3).
    Its zero pattern is the conditional-independence graph: Q_ij = 0 means channels i and j
    are independent once all other channels are known.
    FACTORS: the discrete form Q = Bᵀ W B + V with w_ij = -Q_ij and V_i = Σ_j Q_ij (row sums).
    A weight w_ij is >= 0 exactly when the partial correlation of i and j,
    -Q_ij / sqrt(Q_ii Q_jj), is >= 0. The fraction of negative weights is recorded: when it is
    0 the form is a genuine metric (an 'attractive' Gaussian model)."""
    R, C, shrinkage = correlation_from_trial(trial)
    Q = np.linalg.inv(R)
    n = Q.shape[0]
    edges = [(i, j) for i, j in itertools.combinations(range(n), 2)]
    w = np.array([-Q[i, j] for i, j in edges])
    V = Q.sum(axis=1)
    B = incidence_matrix(n, edges)
    return Operator(kind="PRECISION", n=n, matrix=Q,
                    factors={"B": B, "w": w, "V": V, "edges": edges, "R": R,
                             "shrinkage": shrinkage,
                             "fraction_negative_weights": float(np.mean(w < 0))})


# ----------------------------------------------------------------------------------------
# QUANTUM
# ----------------------------------------------------------------------------------------

def hamiltonian_operator(n: int, pauli: dict, interaction_strings=None) -> Operator:
    """WHAT: the operator of a quantum input: its Hamiltonian H, as a Pauli sum.
    INPUTS: interaction_strings = the terms that are the 'interaction' (the non-free-fermion
    part, e.g. the h_z terms of the Ising chain); recorded so the dial can be read later.
    WHY: the quantum problem's equation is Schrödinger's, dψ/dt = -i H ψ."""
    return Operator(kind="HAMILTONIAN", n=n, pauli=dict(pauli),
                    factors={"interaction_strings": list(interaction_strings or [])})


# ----------------------------------------------------------------------------------------
# DIAGONAL (SAT)
# ----------------------------------------------------------------------------------------

def clean_clause(clause) -> tuple | None:
    """WHAT: tidy one clause: remove repeated literals; return None if the clause is a
    tautology (contains x and not-x, so it is always satisfied).
    LITERALS: DIMACS convention: +j means 'variable j is true', -j means 'variable j is
    false', with j = 1, 2, ..., n."""
    literals = sorted(set(clause), key=lambda lit: (abs(lit), lit))
    variables = [abs(lit) for lit in literals]
    if len(set(variables)) < len(variables):
        return None
    return tuple(literals)


def clause_operator(n: int, clauses: list) -> Operator:
    """WHAT: the clause-penalty Hamiltonian H = Σ_c P_c, as a Pauli sum. For every assignment
    of the variables (= every computational basis state), H's value is the number of clauses
    that assignment violates. So H's ground energy is 0 exactly when the formula is satisfiable.

    CONVENTION: qubit j-1 holds variable j; the basis state |0> means 'false', |1> means 'true'.
    Z has value +1 on |0> and -1 on |1>.

    HOW: clause c = (l_1 or ... or l_k) is violated when every literal is false.
      - literal +j is false when the bit is 0: projector (I + Z)/2 on that qubit
      - literal -j is false when the bit is 1: projector (I - Z)/2
    So P_c = Π_literals (I + s Z)/2 with s = +1 for +j, -1 for -j. Expanding the product gives
    2^k Pauli strings: for every subset S of the literals, coefficient (Π_{l in S} s_l) / 2^k
    on the string with Z on the qubits of S.

    WHY: this is how a logic problem becomes a spin problem (docs/SUBFAMILIES.md §4). With
    3-literal clauses it has 3-qubit terms: type DIAGONAL, not GRAPH."""
    pauli = {}
    kept_clauses = []
    for clause in clauses:
        clause = clean_clause(clause)
        if clause is None:
            continue                       # a tautology is never violated: contributes 0
        kept_clauses.append(clause)
        k = len(clause)
        for size in range(k + 1):
            for subset in itertools.combinations(clause, size):
                sign = 1.0
                letters = {}
                for literal in subset:
                    sign *= 1.0 if literal > 0 else -1.0
                    letters[abs(literal) - 1] = "Z"
                add_term(pauli, pauli_string(n, letters), sign / 2 ** k)
    return Operator(kind="CLAUSE_PENALTY", n=n, pauli=pauli,
                    factors={"clauses": kept_clauses,
                             "max_locality": max((len(c) for c in kept_clauses), default=0)})


# ----------------------------------------------------------------------------------------
# One entry point
# ----------------------------------------------------------------------------------------

def operator_of(problem_type: str, x) -> Operator:
    """WHAT: the operator M(x) of one input, whatever its problem type.
    WHY: the pipeline calls this once per instance without knowing the type's details."""
    if problem_type == "GRAPH":
        return graph_operator(x["n"], x["edges"], x.get("weights"), x.get("potentials"))
    if problem_type == "COVARIANCE":
        return covariance_operator(x)
    if problem_type == "QUANTUM":
        return hamiltonian_operator(x["n"], x["pauli"], x.get("interaction_strings"))
    if problem_type == "DIAGONAL":
        return clause_operator(x["n"], x["clauses"])
    raise ValueError(f"unknown problem type {problem_type}")
