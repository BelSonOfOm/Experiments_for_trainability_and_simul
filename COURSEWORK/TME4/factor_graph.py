"""factor_graph.py — the common object on which structure is measured (Q2, spec 2.1; decision D2.1).

WHAT THIS FILE PROVIDES
A *factor graph* of one operator M(x): a bipartite graph with
  - site nodes, one per site (vertex, channel, variable, qubit), numbered 0..n-1;
  - factor nodes, one per TERM of the operator, joined to every site the term acts on;
  - on each factor a weight = |coefficient| / (largest |coefficient| among the factors of this M(x)),
    so the object does not depend on units, and a label (the kind of term);
  - on each site a 1-site weight (its own field / diagonal value), used as the starting colour of the
    symmetry computation (features.colour_refinement).
How each type becomes a factor graph:
  GRAPH       one factor per edge (2 sites), weight = edge weight; label "edge". Read from the stored
              factorisation M = Bᵀ W B + V (operator.factors: B incidence, w weights, V diagonal) when
              present, else from the off-diagonal entries of the Laplacian matrix. Site weight = V_ii (0 if
              no potential).
  COVARIANCE  one factor per pair i < j, weight = |partial correlation| = |Q_ij| / sqrt(Q_ii Q_jj),
              label "edge". The precision matrix Q is dense, so (almost) every pair is a factor: the
              structure is in the weights (status table 2.4: support forced ~ 1, weighted spectra found).
              Site weight: constant 1 for every site (after normalisation every diagonal is 1). PROPOSED,
              TO CONFIRM WITH CHARBEL.
  DIAGONAL    operator.pauli is the clause penalty written as a Pauli sum {"ZZIZ": c, ...}. One factor
  QUANTUM     per Pauli string with c != 0, EXCEPT the identity string "II...I" (a constant energy shift,
              acts on no site). The factor joins the qubits where the letter is not I
              (harvest.pauli.support_qubits; qubit 0 = leftmost letter); label = the non-I letters in site
              order (e.g. "ZZ", "XZX"); the factor's label is its letters SORTED and each site keeps the letter
              it receives (`roles`), so nothing depends on how sites are numbered. 1-site strings are BOTH factors of size 1 AND the site weight of
              their qubit (sum of |c| of the 1-site strings on that qubit, normalised like the factors).
Why: a Laplacian and a precision matrix are n x n and indexed by sites; a Hamiltonian is 2^n x 2^n and
indexed by bitstrings, so site structure cannot be read from its matrix. The factor graph is the standard
site-level object for a sum of terms (Kschischang, Frey, Loeliger 2001; Mézard & Montanari 2009) and,
unlike a pairwise coupling matrix, keeps k-body terms: XOR-3 (one factor on 3 sites) and a triangle of
pairwise terms (three factors on 2 sites) stay different.
Known loss (say it in the report): how terms overlap (the topology) is not kept; that is the Hodge
complex built on top of the factor graph (next steps).
Known-answer tests: tests_analysis/test_factor_graph_features.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class FactorGraph:
    """FIELDS:
    n_sites       number of sites n
    factors       list of tuples of site indices, one tuple per factor (sorted, no repeats)
    weights       list of floats in (0, 1], one per factor (|coefficient| / largest |coefficient|)
    labels        list of strings, one per factor: "edge", or the factor's Pauli letters SORTED (e.g. "XXZ"
                  for X Z X), so the label does not depend on how the sites are numbered
    roles         list of tuples, one per factor, aligned with `factors`: the letter each of its sites
                  receives (("X", "Z", "X") for X_{i-1} Z_i X_{i+1}; ("e", "e") for an edge). Used as edge
                  colours by colour refinement. FIXED 11 Oct: labels were read in site order, so on a ring
                  the term wrapping around (sites n-2, n-1, 0) got a different label from the others and
                  periodic cluster-Ising chains lost their symmetry (found by the pipeline check)
    site_weights  list of n floats, the 1-site weight of each site (0 when there is none)
    kind          the operator kind it came from (LAPLACIAN, PRECISION, HAMILTONIAN, CLAUSE_PENALTY)
    """
    n_sites: int
    factors: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    site_weights: list = field(default_factory=list)
    kind: str = ""

    def n_factors(self) -> int:
        """WHAT: number of factors. HOW: len(self.factors)."""
        return len(self.factors)

    def factor_sizes(self) -> list:
        """WHAT: number of sites of each factor (1, 2, 3, ...). HOW: [len(f) for f in factors]."""
        return [len(f) for f in self.factors]

    def site_degrees(self, weighted: bool = False) -> list:
        """WHAT: for each site, the number of factors containing it (or, if weighted, the sum of their
        weights). HOW: one pass over factors. WHY: feature site_degree_cv (2.2)."""
        degrees = [0.0] * self.n_sites
        for f, w in zip(self.factors, self.weights):
            for i in f:
                degrees[i] += w if weighted else 1.0
        return degrees

    def covered_pairs(self) -> set:
        """WHAT: the set of site pairs (i, j), i < j, that share at least one factor.
        HOW: for each factor, add every pair of its sites. WHY: feature support_density (2.2)."""
        pairs = set()
        for f in self.factors:
            for a in range(len(f)):
                for b in range(a + 1, len(f)):
                    pairs.add((f[a], f[b]))
        return pairs

    def incidence_matrix(self, weighted: bool = True) -> sp.csr_matrix:
        """WHAT: the n_sites x n_factors matrix A with A[i, f] = weight of f (or 1 if not weighted) when
        site i belongs to factor f, else 0.
        HOW: build COO arrays (rows = sites, cols = factor index) and convert to CSR."""
        rows, cols, vals = [], [], []
        for k, (f, w) in enumerate(zip(self.factors, self.weights)):
            for i in f:
                rows.append(i)
                cols.append(k)
                vals.append(w if weighted else 1.0)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.n_sites, len(self.factors)))

    def bipartite_laplacian(self, normalized: bool = True) -> np.ndarray:
        """WHAT: the Laplacian of the bipartite graph whose nodes are the n sites followed by the
        factors, edge (i, f) with weight A[i, f].
        HOW: adjacency W = [[0, A], [Aᵀ, 0]] ((n + F) x (n + F)); degrees d = row sums of W;
        L = D - W; normalized: L_sym = I - D^{-1/2} W D^{-1/2}, with isolated nodes (d = 0) given a 0 row
        and column. Eigenvalues of L_sym lie in [0, 2] whatever the size, which makes the spectral
        features (2.2) comparable across types.
        WHY: spec_entropy, participation_ratio, alg_connectivity are computed from it."""
        n, F = self.n_sites, len(self.factors)
        A = self.incidence_matrix(weighted=True).toarray()
        W = np.zeros((n + F, n + F))
        W[:n, n:] = A
        W[n:, :n] = A.T
        d = W.sum(axis=1)
        if not normalized:
            return np.diag(d) - W
        inv_sqrt = np.zeros_like(d)
        nonzero = d > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
        L = -inv_sqrt[:, None] * W * inv_sqrt[None, :]
        L[np.diag_indices_from(L)] = nonzero.astype(float)   # 1 on connected nodes, 0 on isolated ones
        return L


def _normalise(fg: FactorGraph) -> FactorGraph:
    """WHAT: divide factor weights and site weights by the largest factor weight (no change if there is
    no factor). WHY: features must not depend on the operator's units."""
    if fg.weights:
        top = max(fg.weights)
        if top > 0:
            fg.weights = [w / top for w in fg.weights]
            fg.site_weights = [w / top for w in fg.site_weights]
    return fg


def from_laplacian(operator) -> FactorGraph:
    """WHAT: the factor graph of a GRAPH operator (kind LAPLACIAN).
    HOW: if operator.factors holds "B" and "w": each row e of B has -1 at the tail i and +1 at the head j;
    factor (i, j) with weight w_e; site weights from "V" if present. Otherwise use the matrix: factor
    (i, j) for every i < j with M_ij != 0, weight -M_ij. Normalise weights by their maximum.
    TEST: path 0-1-2-3, unit weights -> factors (0,1), (1,2), (2,3), weights all 1."""
    n = operator.n
    fg = FactorGraph(n_sites=n, kind=operator.kind, site_weights=[0.0] * n)
    facts = operator.factors or {}
    if "B" in facts and "w" in facts:
        B, w = np.asarray(facts["B"]), np.asarray(facts["w"], dtype=float)
        for e in range(B.shape[0]):
            ends = tuple(sorted(int(i) for i in np.flatnonzero(B[e])))
            if len(ends) == 2 and w[e] != 0:
                fg.factors.append(ends)
                fg.weights.append(abs(float(w[e])))
                fg.labels.append("edge")
                fg.roles.append(("e", "e"))
        if "V" in facts:
            fg.site_weights = [abs(float(v)) for v in np.asarray(facts["V"])]
    else:
        M = np.asarray(operator.matrix, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] != 0:
                    fg.factors.append((i, j))
                    fg.weights.append(abs(float(M[i, j])))
                    fg.labels.append("edge")
                    fg.roles.append(("e", "e"))
    return _normalise(fg)


def from_precision(operator) -> FactorGraph:
    """WHAT: the factor graph of a COVARIANCE operator (kind PRECISION).
    HOW: Q = operator.matrix; d = sqrt(diag(Q)); rho_ij = -Q_ij / (d_i d_j) (partial correlation of
    channels i and j given all others); one factor (i, j) for every i < j with rho_ij != 0 exactly,
    weight |rho_ij| / max |rho|. Site weights: constant 1 (PROPOSED, see the module docstring).
    If Q is diagonal (e.g. the identity of an R6 'uninformative_operator' sample) there is NO factor: the
    factor graph has zero factors and the features return NaN for it (they must not crash).
    TEST: Q = [[2, -1], [-1, 2]] -> one factor (0, 1), rho = 0.5, weight 1."""
    Q = np.asarray(operator.matrix, dtype=float)
    n = Q.shape[0]
    d = np.sqrt(np.diag(Q))
    fg = FactorGraph(n_sites=n, kind=operator.kind, site_weights=[1.0] * n)
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                fg.factors.append((i, j))
                fg.weights.append(abs(Q[i, j]) / (d[i] * d[j]))
                fg.labels.append("edge")
                fg.roles.append(("e", "e"))
    if fg.weights:                                   # site weights stay 1: do not rescale them
        top = max(fg.weights)
        fg.weights = [w / top for w in fg.weights]
    return fg


def from_pauli_sum(operator) -> FactorGraph:
    """WHAT: the factor graph of a DIAGONAL (CLAUSE_PENALTY) or QUANTUM (HAMILTONIAN) operator.
    HOW: for every (string, c) in operator.pauli with c != 0 and string not all 'I':
    sites = harvest.pauli.support_qubits(string); roles = the letter on each of those sites;
    label = the same letters sorted; one factor (sites, |c|, label, roles). Site weight of qubit q = sum of |c| over 1-site strings on q.
    Normalise factor weights and site weights by the largest factor |c|.
    TEST: XOR-3 term ZZZ on 3 qubits -> one factor of size 3; TFIM n=3 open, J = h = 1 ->
    factors (0,1) "ZZ", (1,2) "ZZ", (0,) "X", (1,) "X", (2,) "X"."""
    from harvest.pauli import support_qubits
    n = operator.n
    fg = FactorGraph(n_sites=n, kind=operator.kind, site_weights=[0.0] * n)
    for string, c in operator.pauli.items():
        size = abs(complex(c))
        if size == 0 or set(string) == {"I"}:
            continue
        sites = support_qubits(string)
        fg.factors.append(tuple(sites))
        fg.weights.append(size)
        fg.roles.append(tuple(string[q] for q in sites))
        fg.labels.append("".join(sorted(string[q] for q in sites)))
        if len(sites) == 1:
            fg.site_weights[sites[0]] += size
    return _normalise(fg)


def factor_graph(operator) -> FactorGraph:
    """WHAT: dispatch on operator.kind: LAPLACIAN -> from_laplacian; PRECISION -> from_precision;
    HAMILTONIAN and CLAUSE_PENALTY -> from_pauli_sum. Raise ValueError on any other kind.
    WHY: the single entry point used by features.py."""
    if operator.kind == "LAPLACIAN":
        return from_laplacian(operator)
    if operator.kind == "PRECISION":
        return from_precision(operator)
    if operator.kind in ("HAMILTONIAN", "CLAUSE_PENALTY"):
        return from_pauli_sum(operator)
    raise ValueError(f"unknown operator kind {operator.kind!r}")
