"""forced_table.py — forced / predictable / found status of every (feature, type) cell (Q2 spec 2.4;
decision D2.2). Written BEFORE any result was looked at (11 Oct 2026: only the feature table's median
values of TFIM-open were printed, to check the code of the forced predictions below).

WHAT THIS FILE PROVIDES
1. The approved status table as data (`status`), one status per (feature, cell), where a cell is a type or
   a sub-case of a type (spin chains vs molecules; uniform vs disordered chains; periodic vs open):
     forced       the value is fixed exactly by the definition of the type
     predictable  the value varies, but its expected value follows from the generator's parameters
     found        the value depends on real data and cannot be known in advance
   Approved table (QnA_what_to_do.md 2.4):
   | feature           | GRAPH | COVARIANCE         | DIAGONAL           | spin chains                         | molecules |
   | term sizes        | forced: all 2 | forced: all 2 | predictable (k)  | forced: 1 and 2 (3 cluster-Ising)   | found |
   | support density   | found | forced ~ 1         | predictable        | forced 2/n open NN, ~2x with NNN     | found |
   | terms per site    | found | forced (n-1)/2     | predictable        | forced ~ 2                          | found |
   | site degree spread| found | forced 0           | predictable        | forced 0 periodic, boundary open    | found |
   | symmetry          | found | forced ~ 0         | predictable        | forced 1 periodic, ~n/2 classes open, ~0 disordered | found |
   | spectral (weighted)| found | found             | predictable        | forced uniform; found disordered    | found |
   Type-specific features: entanglement and in-sector gap (and their per-side versions) are PREDICTABLE for
   spin chains (Q4 spec 4.5) and FOUND for molecules (entanglement); interaction range is FORCED for spin
   chains (1 with nearest-neighbour terms, 2 with next-nearest or three-site terms: fixed by the
   Hamiltonian) and for COVARIANCE (n - 1, dense precision matrix); the operator spectra and effective rank
   are FOUND within their type.
2. The PREDICTED value of forced cells and the EXPECTED value of predictable cells, where a derivation is
   written below (`predicted_value`); NaN where no derivation has been written yet (listed by
   `prediction_coverage`).
3. `pipeline_check`: for forced cells with a prediction, computed vs predicted; a mismatch is a BUG.
4. `finding_mask`: which rows count as findings for a feature (predictable or found cells).
"""
from __future__ import annotations

import math
from enum import Enum

import numpy as np


class Status(Enum):
    """The three statuses of a (feature, cell) pair (decision D2.2)."""
    FORCED = "forced"
    PREDICTABLE = "predictable"
    FOUND = "found"


F, P, D = Status.FORCED, Status.PREDICTABLE, Status.FOUND

CELLS = ("GRAPH", "COVARIANCE", "DIAGONAL", "SPIN_UNIFORM_OPEN", "SPIN_UNIFORM_PERIODIC", "SPIN_DISORDERED",
         "SPIN_CLUSTER", "MOLECULE")
"""Every cell of the status table."""

FEATURE_GROUP = {
    "term_size_mean": "term sizes", "frac_size_1": "term sizes", "frac_size_2": "term sizes",
    "frac_size_3plus": "term sizes", "support_density": "support density", "terms_per_site": "terms per site",
    "site_degree_cv": "site degree spread", "sym_classes": "symmetry", "sym_fraction": "symmetry",
    "spec_entropy": "spectral", "participation_ratio": "spectral", "alg_connectivity": "spectral",
    "op_spec_entropy": "operator spectrum", "op_gap_ratio": "operator spectrum", "effective_rank": "effective rank",
    "interaction_range": "interaction range", "entanglement": "entanglement", "in_sector_gap": "gap",
    "entanglement_below": "entanglement", "entanglement_above": "entanglement",
    "gap_below": "gap", "gap_above": "gap"}
"""Each feature of features.py -> its row of the status table."""

#                  GRAPH COV  DIAG UNI_O UNI_P DISORD CLUST MOLEC
_TABLE = {
    "term sizes":         (F, F, P, F, F, F, F, D),
    "support density":    (D, F, P, F, F, F, F, D),
    "terms per site":     (D, F, P, F, F, F, F, D),
    "site degree spread": (D, F, P, F, F, F, F, D),
    "symmetry":           (D, F, P, F, F, F, F, D),
    "spectral":           (D, D, P, F, F, D, F, D),
    "operator spectrum":  (D, D, None, None, None, None, None, None),
    "effective rank":     (None, D, None, None, None, None, None, None),
    "interaction range":  (None, F, None, F, F, F, F, None),
    "entanglement":       (None, None, None, P, P, P, P, D),
    "gap":                (None, None, None, P, P, P, P, None),
}
"""None = the feature does not exist in that cell (NaN in the feature table)."""

NN_FAMILIES = ("TFIM", "XXZ", "XY", "DISORDERED", "BOND_ALT")
"""Spin families with nearest-neighbour terms only (harvest/sources/spin_generator.py, spin_more.py)."""
NNN_FAMILIES = ("J1J2", "ANNNI", "CLUSTER")
"""Spin families whose terms also reach distance 2 (J1-J2 and ANNNI next-nearest couplings; the
cluster-Ising three-site X Z X terms span sites i-1, i+1)."""
K_OF = {"2-SAT": 2, "3-SAT": 3, "4-SAT": 4, "5-SAT": 5, "XOR-3": 3, "1-IN-3": 3, "NAE-3": 3, "PLANTED-3": 3}
"""Number of distinct variables per constraint for the SAT generators with a single constraint size
(harvest/sources/sat_more.py). CA-3 (community attachment) and SR have no single-k prediction here;
2+P-SAT is handled separately (half 2-, half 3-variable constraints)."""


def family_of(subfamily: str) -> str:
    """WHAT: the spin family of a sub-family name ("TFIM-open" -> "TFIM")."""
    return subfamily.split("-")[0]


def cell_of(row) -> str:
    """WHAT: the cell name of one feature-table row: "GRAPH", "COVARIANCE", "DIAGONAL",
    "SPIN_UNIFORM_OPEN", "SPIN_UNIFORM_PERIODIC", "SPIN_DISORDERED", "SPIN_CLUSTER", "MOLECULE".
    HOW: from problem_type, group_id ("CHEM:..." -> MOLECULE) and subfamily (e.g. "TFIM-open",
    "DISORDERED-periodic", "CLUSTER-open"). Raise on an unknown spin family rather than guessing.
    WHY: the status depends on the sub-case, not only on the type."""
    t = row["problem_type"]
    if t in ("GRAPH", "COVARIANCE", "DIAGONAL"):
        return t
    if str(row["group_id"]).startswith("CHEM:"):
        return "MOLECULE"
    family = family_of(row["subfamily"])
    if family == "DISORDERED":
        return "SPIN_DISORDERED"
    if family == "CLUSTER":
        return "SPIN_CLUSTER"
    if family in NN_FAMILIES + NNN_FAMILIES:
        return "SPIN_UNIFORM_PERIODIC" if row["subfamily"].endswith("periodic") else "SPIN_UNIFORM_OPEN"
    raise ValueError(f"unknown spin sub-family {row['subfamily']!r}")


def status(feature: str, cell: str):
    """WHAT: the Status of (feature, cell), or None if the feature does not exist in that cell.
    HOW: FEATURE_GROUP gives the table row, CELLS the column. A feature missing from FEATURE_GROUP raises
    KeyError: every pair must be decided explicitly."""
    return _TABLE[FEATURE_GROUP[feature]][CELLS.index(cell)]


def _chain_pairs(n: int, periodic: bool, distances: tuple) -> set:
    """WHAT: the site pairs at the given distances along a chain, with the harvest's own bonds() rule
    (harvest/sources/spin_generator.py: a pair reached twice on a ring is listed once)."""
    from harvest.sources.spin_generator import bonds
    pairs = set()
    for d in distances:
        pairs.update(tuple(sorted(p)) for p in bonds(n, "periodic" if periodic else "open", d))
    return pairs


def predicted_value(feature: str, row) -> float:
    """WHAT: the predicted value of a FORCED cell (exact) or the expected value of a PREDICTABLE cell, for
    one row of the feature table; NaN for FOUND cells and where no derivation is written.
    DERIVATIONS (each from the generator's definition):
      GRAPH, COVARIANCE term sizes: every factor joins 2 sites -> term_size_mean 2, frac_size_2 1,
        frac_size_1 = frac_size_3plus = 0.
      COVARIANCE (dense precision matrix, every partial correlation non-zero): support_density 1;
        terms_per_site (n(n-1)/2)/n = (n-1)/2; site_degree_cv 0 (every channel in n-1 factors);
        interaction_range n-1; sym_classes n (site weights all 1, factor weights distinct real numbers, so
        colour refinement separates every site), sym_fraction 0.
      spin chains (Hamiltonians in harvest/sources/spin_generator.py and spin_more.py):
        support_density = |pairs at distance 1 (and 2 for J1-J2, ANNNI, cluster-Ising)| / (n(n-1)/2),
          pairs counted with the harvest's bonds() rule (open: 2/n for nearest-neighbour chains);
        interaction_range = 1 (nearest-neighbour families) or 2 (J1-J2, ANNNI, cluster-Ising);
        frac_size_3plus = 0 except cluster-Ising (three-site X Z X terms); frac_size_1 = 0 for cluster-Ising
          (no field term);
        periodic chains with identical terms on every site: sym_classes 1, sym_fraction 1, site_degree_cv 0
          (also the disordered chain for site_degree_cv: random weights, same term pattern);
        open chains with identical terms (reflection the only symmetry): sym_classes ceil(n/2);
        disordered chain (every weight drawn at random): sym_classes n, sym_fraction 0.
        Bond-alternating chains (alternating couplings): symmetry not derived (NaN).
      DIAGONAL, expected support_density: m = max(1, round(ratio * n)) constraints (sat_more.formula), each
        on k distinct variables, covers a given pair with probability q = k(k-1)/(n(n-1)); expected
        density 1 - (1 - q)^m; 2+P-SAT: half the constraints with k = 2, half with k = 3. Exact
        coefficient cancellations between constraints are ignored (expected value, not exact).
    Everything else: NaN (listed by prediction_coverage)."""
    cell = cell_of(row)
    st = status(feature, cell) if feature in FEATURE_GROUP else None
    if st not in (Status.FORCED, Status.PREDICTABLE):
        return math.nan
    n = int(row["n"])
    if cell in ("GRAPH", "COVARIANCE"):
        sizes = {"term_size_mean": 2.0, "frac_size_2": 1.0, "frac_size_1": 0.0, "frac_size_3plus": 0.0}
        if feature in sizes:
            return sizes[feature]
    if cell == "COVARIANCE":
        return {"support_density": 1.0, "terms_per_site": (n - 1) / 2, "site_degree_cv": 0.0,
                "interaction_range": float(n - 1), "sym_classes": float(n), "sym_fraction": 0.0}.get(feature, math.nan)
    if cell == "DIAGONAL":
        if feature != "support_density":
            return math.nan
        sub, ratio = row["subfamily"], row.get("ratio", math.nan)
        if ratio != ratio:
            return math.nan
        m = max(1, int(round(ratio * n)))
        q = lambda k: k * (k - 1) / (n * (n - 1))      # noqa: E731  (probability that one constraint covers a pair)
        if sub in K_OF:
            return 1 - (1 - q(K_OF[sub])) ** m
        if sub == "2+P-SAT":
            return 1 - (1 - q(2)) ** (m / 2) * (1 - q(3)) ** (m / 2)
        return math.nan
    if cell.startswith("SPIN"):
        family = family_of(row["subfamily"])
        periodic = row["subfamily"].endswith("periodic")
        distances = (1, 2) if family in NNN_FAMILIES else (1,)
        if feature == "support_density":
            return len(_chain_pairs(n, periodic, distances)) / (n * (n - 1) / 2)
        if feature == "interaction_range":
            return float(max(distances))
        if feature == "frac_size_3plus":
            return math.nan if family == "CLUSTER" else 0.0
        if feature == "frac_size_1" and family == "CLUSTER":
            return 0.0
        if family == "BOND_ALT":
            return math.nan
        if periodic and feature in ("sym_classes", "sym_fraction", "site_degree_cv") and cell != "SPIN_DISORDERED":
            return {"sym_classes": 1.0, "sym_fraction": 1.0, "site_degree_cv": 0.0}[feature]
        if periodic and feature == "site_degree_cv":
            return 0.0
        if cell == "SPIN_DISORDERED" and feature in ("sym_classes", "sym_fraction"):
            return {"sym_classes": float(n), "sym_fraction": 0.0}[feature]
        if not periodic and cell != "SPIN_DISORDERED" and feature == "sym_classes":
            return float(math.ceil(n / 2))
    return math.nan


def add_ratio(features: "pandas.DataFrame", index: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT: the feature table with a `ratio` column for DIAGONAL rows (the constraint ratio stored in the
    modelling map of the first problem of each input set; NaN elsewhere). WHY: predicted_value needs it."""
    by_task = dict(zip(index["task_id"], index["modelling_map"]))
    ratios = []
    for task_ids, t in zip(features["task_ids"], features["problem_type"]):
        mm = by_task[task_ids.split(";")[0]] if t == "DIAGONAL" else {}
        ratios.append(float(mm.get("ratio", math.nan)))
    out = features.copy()
    out["ratio"] = ratios
    return out


def pipeline_check(features: "pandas.DataFrame", rel_tolerance: float = 1e-6) -> "pandas.DataFrame":
    """WHAT: every FORCED (feature, row) with a prediction where the computed median differs from it by more
    than rel_tolerance (relative to max(1, |predicted|)). OUTPUT columns: input_set, subfamily, n, cell,
    feature, computed, predicted, difference.
    HOW: for each row and each feature with status FORCED and a finite predicted_value: compare
    features[f + "_median"]. An empty table is the expected result.
    WHY: forced differences are reported as checks that the pipeline behaves (D2.2)."""
    import pandas as pd
    bad = []
    for _, row in features.iterrows():
        cell = cell_of(row)
        for feature in FEATURE_GROUP:
            col = f"{feature}_median"
            if col not in features.columns or status(feature, cell) is not Status.FORCED:
                continue
            predicted = predicted_value(feature, row)
            computed = row[col]
            if predicted != predicted:
                continue
            if computed != computed or abs(computed - predicted) > rel_tolerance * max(1.0, abs(predicted)):
                bad.append({"input_set": row["input_set"], "subfamily": row["subfamily"], "n": row["n"], "cell": cell,
                            "feature": feature, "computed": computed, "predicted": predicted,
                            "difference": computed - predicted})
    return pd.DataFrame(bad, columns=["input_set", "subfamily", "n", "cell", "feature", "computed", "predicted",
                                      "difference"])


def prediction_coverage(features: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT: for every (cell, feature) that is FORCED or PREDICTABLE: number of rows, and how many have a
    written prediction. WHY: says openly which forced cells are checked and which are not yet."""
    import pandas as pd
    counts = {}
    for _, row in features.iterrows():
        cell = cell_of(row)
        for feature in FEATURE_GROUP:
            st = status(feature, cell)
            if st in (Status.FORCED, Status.PREDICTABLE):
                key = (cell, feature, st.value)
                total, covered = counts.get(key, (0, 0))
                counts[key] = (total + 1, covered + (predicted_value(feature, row) == predicted_value(feature, row)))
    return pd.DataFrame([{"cell": c, "feature": f, "status": s, "rows": t, "with_prediction": k}
                         for (c, f, s), (t, k) in sorted(counts.items())])


def finding_mask(features: "pandas.DataFrame", feature: str) -> "pandas.Series":
    """WHAT: a boolean per row: True when status(feature, cell_of(row)) is PREDICTABLE or FOUND.
    WHY: only these rows enter the tests reported as findings (Q2 level/shape, Q4 tier 3) and the feature
    sets of Q3b."""
    import pandas as pd
    return pd.Series([status(feature, cell_of(row)) in (Status.PREDICTABLE, Status.FOUND)
                      for _, row in features.iterrows()], index=features.index)
