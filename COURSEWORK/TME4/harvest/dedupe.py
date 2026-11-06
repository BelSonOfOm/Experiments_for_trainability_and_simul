"""How many *different* inputs does a problem have, and do equal inputs carry different labels?

Why this file exists: on 5 Oct we found that IMDB-MULTI at n = 7 has 332 graphs but only 10
different ones (307 are the same complete graph K₇), and that equal graphs carry different
labels. Such a 'problem' has nothing to learn. Screening rules S6 and S7 (config.py) reject it.

Two inputs count as 'the same' when they have the same *canonical key*:

    GRAPH       the Weisfeiler-Lehman hash (a fingerprint of the graph that does not depend on
                how the vertices are numbered) joined with the rounded Laplacian spectrum.
                Two isomorphic graphs (same graph, renumbered) always get the same key.
                WL alone fails on regular graphs (all d-regular graphs on n vertices get one
                hash); the spectrum separates them except for rare 'cospectral' pairs.
                Node types and edge weights, if present, are part of the key.
    COVARIANCE  the trial's raw numbers, rounded to 8 digits (recordings are practically never
                equal, so this mostly confirms they are distinct)
    QUANTUM     the Hamiltonian's terms and coefficients, rounded to 8 digits
    DIAGONAL    the set of clauses (order of clauses and of literals inside a clause ignored).
                Renaming the variables is NOT ignored: two formulas equal up to renaming count
                as different (a choice; the group may want to change it).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import networkx as nx
import numpy as np

from .operators import clean_clause


def canonical_key(problem_type: str, x) -> str:
    """WHAT: a string that is equal for two inputs exactly when we consider them the same input.
    WHY: counting distinct inputs (S6) and conflicting labels (S7), and matching inputs across
    problems (near_duplicates.py)."""
    if problem_type == "GRAPH":
        # WL hash + Laplacian spectrum (8 Oct). WL alone gives every d-regular graph on n
        # vertices the same hash, so random regular graphs all looked identical. The spectrum
        # (eigenvalues of the weighted Laplacian) does not depend on the numbering either, and
        # separates almost all of them. Edge weights now enter both parts.
        graph = nx.Graph()
        types = x.get("node_types")
        weights = x.get("weights")
        for v in range(x["n"]):
            graph.add_node(v, t=str(types[v]) if types is not None else "")
        for e, (i, j) in enumerate(x["edges"]):
            graph.add_edge(i, j, w="" if weights is None else f"{float(weights[e]):.8f}")
        wl = nx.weisfeiler_lehman_graph_hash(graph, node_attr="t", edge_attr="w", iterations=3)
        L = np.zeros((x["n"], x["n"]))
        for e, (i, j) in enumerate(x["edges"]):
            w = 1.0 if weights is None else float(weights[e])
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
        spectrum = np.round(np.linalg.eigvalsh(L), 6) + 0.0      # + 0.0 turns -0.0 into 0.0
        return "G" + wl + hashlib.sha1(spectrum.tobytes()).hexdigest()[:16]
    if problem_type == "COVARIANCE":
        rounded = np.round(np.asarray(x, dtype=float), 8)
        return "C" + hashlib.sha1(rounded.tobytes()).hexdigest()
    if problem_type == "QUANTUM":
        terms = sorted((s, round(float(np.real(c)), 8)) for s, c in x["pauli"].items())
        return "Q" + hashlib.sha1(repr(terms).encode()).hexdigest()
    if problem_type == "DIAGONAL":
        clauses = sorted(c for c in (clean_clause(c) for c in x["clauses"]) if c is not None)
        return "D" + hashlib.sha1(repr(clauses).encode()).hexdigest()
    raise ValueError(f"unknown problem type {problem_type}")


def distinct_and_conflicts(problem_type: str, instances: list) -> tuple:
    """WHAT: (number of distinct inputs, conflict rate).
    HOW: group the instances by canonical key. The conflict rate is the fraction of instances
    whose key appears with both labels.
    EXAMPLE: keys/labels  A:+1, A:+1, A:-1, B:+1  ->  2 distinct, conflict rate 3/4 (the three A's).
    WHY: rules S6 and S7."""
    labels_by_key = defaultdict(list)
    for inst in instances:
        labels_by_key[canonical_key(problem_type, inst.x)].append(inst.y)
    in_conflict = sum(len(ys) for ys in labels_by_key.values() if len(set(ys)) > 1)
    return len(labels_by_key), in_conflict / max(1, len(instances))
