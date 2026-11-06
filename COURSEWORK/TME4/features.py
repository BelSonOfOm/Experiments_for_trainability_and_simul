"""features.py — structure features of every input set (Q2 spec 2.2 and 2.3; used by Q2, Q3, Q4, Q6).

WHAT THIS FILE PRODUCES
1. `instance_features(fg)`: for ONE input's factor graph (factor_graph.py), the numbers of spec 2.2.
   Every feature is unchanged when the operator is multiplied by a constant (weights are normalised).
     terms_per_site       number of factors / n
     term_size_mean       mean number of sites per factor
     frac_size_1, frac_size_2, frac_size_3plus   distribution of factor sizes
     site_degree_cv       (std / mean) of the number of factors per site; 0 = all sites equally involved
     support_density      fraction of the n(n-1)/2 site pairs that share at least one factor
     sym_classes          number of site classes after colour refinement (exact equality of weights,
                          decision D2.2); sym_fraction = 1 - (sym_classes - 1)/(n - 1)
     spec_entropy         entropy of the normalised eigenvalues of the bipartite normalised Laplacian
     participation_ratio  (Σλ)² / Σλ² of the same eigenvalues, divided by the number of nodes
     alg_connectivity     second-smallest eigenvalue of the same Laplacian
2. Type-specific features (spec 2.3; stored, NOT tested across types in Q2; used within type in Q3b/Q4):
     op_spec_entropy, op_gap_ratio   GRAPH: normalised Laplacian; COVARIANCE: precision scaled to unit
                                     diagonal (entropy of eigenvalues; second-smallest / largest)
     effective_rank        COVARIANCE: participation ratio of the eigenvalues of R (the trial's
                           correlation matrix, stored in operator.factors["R"])
     interaction_range     COVARIANCE and QUANTUM spin chains (sites have an identity): largest site
                           distance inside one factor (periodic chains: min(|i-j|, n-|i-j|))
     entanglement          QUANTUM: half-chain von Neumann entropy of the stored ground state
     in_sector_gap         QUANTUM spin chains: the gap stored in instance.info["gap"] (data_v3 symmetry
                           sectors, harvest/sources/spin_more.py)
     entanglement_below / _above, gap_below / _above
                           spin chains only: the same two quantities, median over the inputs on each side
                           of the cut's critical value p_c (Q4 spec 4.5: whole phases can be critical, so
                           laws are assigned per side). Side = sign of (parameter - p_c), the same rule as
                           the label (spin_more.py); p_c and the parameter name are read from
                           harvest.sources.spin_more.FAMILIES. ADDED 11 Oct (Q4 point raised in the
                           skeleton), TO CONFIRM WITH CHARBEL.
3. `input_set_features(...)`: one row per DISTINCT INPUT SET (decision D4.1): apply 1 and 2 to EVERY input
   of the input set (features are label-free, so TRAIN-only is not needed) and store, per feature, the
   median and the interquartile range (q75 - q25). Optionally excluding the R6-flagged samples
   (sensitivity checks 2.7, 3.8, 4.6).
4. `build_feature_table()`: loop over all input sets, write results/features.csv (and
   features_noflags.csv). Columns: input_set, problem_type, n, group_id, subfamily, tier, n_problems,
   task_ids, n_inputs, then <feature>_median and <feature>_iqr for every feature. Each input set's row is
   cached in results/feature_cache/ so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np

from factor_graph import factor_graph

INSTANCE_FEATURES = ("terms_per_site", "term_size_mean", "frac_size_1", "frac_size_2", "frac_size_3plus",
                     "site_degree_cv", "support_density", "sym_classes", "sym_fraction", "spec_entropy",
                     "participation_ratio", "alg_connectivity")
"""Names of the features of 2.2, in table order."""

TYPE_SPECIFIC_FEATURES = ("op_spec_entropy", "op_gap_ratio", "effective_rank", "interaction_range",
                          "entanglement", "in_sector_gap")
"""Names of the type-specific features of 2.3 (NaN where a feature does not exist for the type)."""

SIDE_FEATURES = ("entanglement_below", "entanglement_above", "gap_below", "gap_above")
"""Spin-chain features per side of p_c (input-set level only; see the module docstring)."""

NAN = float("nan")


def term_size_stats(fg) -> dict:
    """WHAT: term_size_mean, frac_size_1, frac_size_2, frac_size_3plus (fractions sum to 1).
    HOW: from fg.factor_sizes(). Zero factors (identity precision matrix): mean NaN, fractions NaN.
    TEST: factors of sizes [2, 2, 3] -> mean 7/3, fractions 0, 2/3, 1/3."""
    sizes = fg.factor_sizes()
    if not sizes:
        return {"term_size_mean": NAN, "frac_size_1": NAN, "frac_size_2": NAN, "frac_size_3plus": NAN}
    m = len(sizes)
    return {"term_size_mean": sum(sizes) / m,
            "frac_size_1": sum(s == 1 for s in sizes) / m,
            "frac_size_2": sum(s == 2 for s in sizes) / m,
            "frac_size_3plus": sum(s >= 3 for s in sizes) / m}


def site_degree_cv(fg) -> float:
    """WHAT: coefficient of variation of the unweighted site degrees: std / mean (population std).
    HOW: fg.site_degrees(); NaN if the mean is 0. TEST: periodic ring -> 0."""
    d = np.asarray(fg.site_degrees(weighted=False))
    return float(d.std() / d.mean()) if d.mean() > 0 else NAN


def support_density(fg) -> float:
    """WHAT: len(fg.covered_pairs()) / (n(n-1)/2). TEST: open nearest-neighbour chain, n = 4 -> 3/6 = 0.5
    (forced value 2/n, status table 2.4)."""
    n = fg.n_sites
    return len(fg.covered_pairs()) / (n * (n - 1) / 2) if n > 1 else NAN


def colour_refinement(fg) -> list:
    """WHAT: a class label for every SITE after colour refinement (1-dimensional Weisfeiler-Lehman) on the
    factor graph; equal labels = sites that look alike.
    HOW: initial colour of site i = its site weight (exact value, no rounding: decision D2.2); initial
    colour of factor f = (label_f, weight_f). Repeat: new colour of a node = (its colour, sorted multiset of
    (neighbour colour, Pauli letter on the connecting edge = fg.roles)); relabel colours by integers; stop when the number of classes (sites and factors)
    no longer grows (at most n + F rounds). Return the final site colours as integers 0..k-1.
    TEST: periodic uniform ring -> 1 class; open uniform chain of 4 -> 2 classes (ends, middle);
    disordered weights -> n classes.
    WHY: sym_classes / sym_fraction (2.2). Exact equality makes measured COVARIANCE ~ 0 symmetry
    (forced cell in 2.4, stated in the report)."""
    n = fg.n_sites
    roles = fg.roles if fg.roles else [tuple("e" for _ in f) for f in fg.factors]
    neighbours = [[] for _ in range(n + len(fg.factors))]          # (neighbour, letter on the edge)
    for k, (f, letters) in enumerate(zip(fg.factors, roles)):
        for i, letter in zip(f, letters):
            neighbours[i].append((n + k, letter))
            neighbours[n + k].append((i, letter))
    colours = [("s", fg.site_weights[i]) for i in range(n)] + \
              [("f", lab, w) for lab, w in zip(fg.labels, fg.weights)]
    relabel = {c: k for k, c in enumerate(sorted(set(colours), key=repr))}
    colours = [relabel[c] for c in colours]
    count = len(set(colours))
    for _ in range(len(colours) + 1):
        signatures = [(colours[v], tuple(sorted((colours[u], letter) for u, letter in neighbours[v])))
                      for v in range(len(colours))]
        relabel = {s: k for k, s in enumerate(sorted(set(signatures)))}
        new = [relabel[s] for s in signatures]
        if len(set(new)) == count:
            break
        colours, count = new, len(set(new))
    site = colours[:n]
    order = {c: k for k, c in enumerate(dict.fromkeys(site))}
    return [order[c] for c in site]


def symmetry_features(fg) -> dict:
    """WHAT: sym_classes = number of distinct colours from colour_refinement; sym_fraction =
    1 - (sym_classes - 1)/(n - 1) (n = 1: 1.0). TEST: ring -> (1, 1.0); all distinct -> (n, 0.0)."""
    k = len(set(colour_refinement(fg)))
    n = fg.n_sites
    return {"sym_classes": float(k), "sym_fraction": 1.0 if n == 1 else 1 - (k - 1) / (n - 1)}


def _entropy_and_pr(values: np.ndarray) -> tuple:
    """WHAT: (normalised entropy, participation ratio / count) of a list of non-negative values.
    HOW: p = v / Σv; H = -Σ p log p over p > 0, divided by log(count); PR = (Σv)² / Σv² / count."""
    v = np.clip(np.asarray(values, dtype=float), 0, None)
    total = v.sum()
    if total <= 0 or len(v) < 2:
        return NAN, NAN
    p = v[v > 0] / total
    entropy = float(-(p * np.log(p)).sum() / math.log(len(v)))
    pr = float(total ** 2 / (v ** 2).sum() / len(v))
    return entropy, pr


def bipartite_spectrum(fg) -> np.ndarray:
    """WHAT: the eigenvalues of fg.bipartite_laplacian(normalized=True), computed fast.
    HOW: for a bipartite graph, D^{-1/2} W D^{-1/2} = [[0, C], [Cᵀ, 0]] with C = D_s^{-1/2} A D_f^{-1/2}
    (sites x factors), whose eigenvalues are ±σ_k (σ = singular values of C) and zeros. So the normalised
    Laplacian has eigenvalues 1 - σ_k and 1 + σ_k (k = 1..min(n', F')), 1 repeated |F' - n'| times
    (n', F' = non-isolated sites and factors), and 0 once per isolated node.
    WHY: 5-SAT at n = 12 has thousands of factors; a dense eigendecomposition of the (n + F)-square
    Laplacian per input was too slow (11 Oct), while C has only n <= 12 rows. Equal to the dense result
    (test_bipartite_spectrum_matches_dense)."""
    A = fg.incidence_matrix(weighted=True).toarray()
    ds, df = A.sum(axis=1), A.sum(axis=0)
    sites, facs = ds > 0, df > 0
    C = A[np.ix_(sites, facs)] / np.sqrt(ds[sites])[:, None] / np.sqrt(df[facs])[None, :]
    sigma = np.clip(np.linalg.svd(C, compute_uv=False), 0, 1) if C.size else np.zeros(0)
    n1, f1 = int(sites.sum()), int(facs.sum())
    isolated = (len(ds) - n1) + (len(df) - f1)
    lam = np.concatenate([1 - sigma, 1 + sigma, np.ones(abs(f1 - n1)), np.zeros(isolated)])
    return np.sort(np.clip(lam, 0, None))


def spectral_features(fg) -> dict:
    """WHAT: spec_entropy, participation_ratio, alg_connectivity of fg.bipartite_laplacian(normalized=True).
    HOW: λ = eigenvalues (eigvalsh, real symmetric); clip tiny negatives to 0; p = λ / Σλ;
    spec_entropy = -Σ p log p (terms with p = 0 skipped), divided by log(number of nodes) so it lies in
    [0, 1]; participation_ratio = (Σλ)² / Σλ² / number of nodes; alg_connectivity = second-smallest λ.
    Zero factors: all NaN. TEST: one size-2 factor on 2 sites = a path of 3 nodes (site, factor, site);
    normalised Laplacian eigenvalues 0, 1, 2 -> spec_entropy = H(0, 1/3, 2/3) / log 3 = 0.5794,
    participation_ratio = (3² / 5) / 3 = 0.6, alg_connectivity = 1."""
    if fg.n_factors() == 0:
        return {"spec_entropy": NAN, "participation_ratio": NAN, "alg_connectivity": NAN}
    lam = bipartite_spectrum(fg)
    entropy, pr = _entropy_and_pr(lam)
    return {"spec_entropy": entropy, "participation_ratio": pr, "alg_connectivity": float(np.sort(lam)[1])}


def instance_features(fg) -> dict:
    """WHAT: all INSTANCE_FEATURES of one factor graph, as {name: float}.
    HOW: terms_per_site = fg.n_factors() / n; then merge term_size_stats, site_degree_cv,
    support_density, symmetry_features, spectral_features. Never raise on a degenerate input: return NaN
    (a factor graph with no factor keeps terms_per_site = support_density = 0).
    WHY: spec 2.2."""
    out = {"terms_per_site": fg.n_factors() / fg.n_sites}
    out.update(term_size_stats(fg))
    out["site_degree_cv"] = site_degree_cv(fg)
    out["support_density"] = support_density(fg)
    if fg.n_factors() == 0:
        out.update({"sym_classes": NAN, "sym_fraction": NAN})
    else:
        out.update(symmetry_features(fg))
    out.update(spectral_features(fg))
    return {k: out[k] for k in INSTANCE_FEATURES}


def operator_spectrum_features(operator) -> dict:
    """WHAT (type-specific): op_spec_entropy and op_gap_ratio of the operator's OWN matrix, for GRAPH
    (Laplacian, normalised: D^{-1/2} L D^{-1/2}) and COVARIANCE (precision, scaled to unit diagonal).
    op_gap_ratio = second-smallest eigenvalue / largest. NaN for DIAGONAL and QUANTUM (their own spectra
    are energy levels, covered by in_sector_gap for spin chains).
    WHY: comparable within a type only (Q3b, Q4)."""
    if operator.kind not in ("LAPLACIAN", "PRECISION"):
        return {"op_spec_entropy": NAN, "op_gap_ratio": NAN}
    M = np.asarray(operator.matrix, dtype=float)
    d = np.diag(M).copy()
    scale = np.zeros_like(d)
    scale[d > 0] = 1.0 / np.sqrt(d[d > 0])
    lam = np.sort(np.linalg.eigvalsh(scale[:, None] * M * scale[None, :]))
    entropy, _ = _entropy_and_pr(lam)
    return {"op_spec_entropy": entropy, "op_gap_ratio": float(lam[1] / lam[-1]) if lam[-1] > 0 else NAN}


def effective_rank(operator) -> float:
    """WHAT (COVARIANCE): participation ratio (Σμ)² / Σμ² of the eigenvalues μ of R (the trial's
    correlation matrix), i.e. the effective number of independent channel modes, between 1 and n.
    HOW: R = operator.factors["R"] when stored (it is, for every COVARIANCE operator), else Q^{-1}.
    NaN for other types."""
    if operator.kind != "PRECISION":
        return NAN
    R = operator.factors.get("R") if operator.factors else None
    mu = np.linalg.eigvalsh(np.asarray(R, dtype=float) if R is not None else np.linalg.inv(operator.matrix))
    mu = np.clip(mu, 0, None)
    return float(mu.sum() ** 2 / (mu ** 2).sum())


def interaction_range(fg, periodic: bool) -> float:
    """WHAT (COVARIANCE, QUANTUM spin chains): the largest distance between two sites of one factor, with
    distance |i - j| (open) or min(|i - j|, n - |i - j|) (periodic). NaN when there is no factor of size >= 2.
    The caller passes NaN for GRAPH, DIAGONAL and molecules (sites have no geometry there)."""
    n, best = fg.n_sites, -1
    for f in fg.factors:
        for a in f:
            for b in f:
                gap = abs(a - b)
                best = max(best, min(gap, n - gap) if periodic else gap)
    return float(best) if best > 0 else NAN


def entanglement_entropy(state, n: int) -> float:
    """WHAT (QUANTUM): half-chain von Neumann entropy of a pure state of n qubits, natural log.
    HOW: psi reshaped to (2^{n//2}, 2^{n - n//2}); qubit 0 is the leftmost letter and the first Kronecker
    factor (harvest/pauli.py), so this cut separates qubits 0..n//2-1 from the rest; s = singular values;
    p = s² (normalised to sum 1); S = -Σ p log p over p > 1e-15.
    TEST: product state |0...0> -> 0; Bell pair on 2 qubits -> log 2.
    WHY: Q4 spec 4.5 (area law vs (c/6) log n). Degenerate ground states (R6) have an arbitrary
    entanglement: the sensitivity run excludes them."""
    psi = np.asarray(state).reshape(2 ** (n // 2), 2 ** (n - n // 2))
    s = np.linalg.svd(psi, compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 1e-15]
    return float(-(p * np.log(p)).sum())


def in_sector_gap(instance) -> float:
    """WHAT (QUANTUM spin chains): the gap within the chosen symmetry sector, instance.info["gap"]
    (written by harvest/sources/symmetry_sectors.py via spin_more.py). NaN when absent.
    WHY: Q4 spec 4.5 (gap closing like 1/n at a transition, constant away from it)."""
    value = (instance.info or {}).get("gap")
    return float(value) if value is not None else NAN


def is_spin_chain(problem) -> bool:
    """WHAT: True for the QUANTUM spin-chain problems (source spin-generator-2), False for molecules."""
    return problem.problem_type == "QUANTUM" and problem.source != "chem-generator"


def type_specific_features(problem, instance, operator, fg) -> dict:
    """WHAT: every TYPE_SPECIFIC_FEATURES entry for one input, NaN where it does not apply.
    HOW: call the functions above according to problem_type (and source: molecules vs spin chains for
    QUANTUM; entanglement uses instance.x["state"], n = instance.x["n"])."""
    out = dict.fromkeys(TYPE_SPECIFIC_FEATURES, NAN)
    out.update(operator_spectrum_features(operator))
    if problem.problem_type == "COVARIANCE":
        out["effective_rank"] = effective_rank(operator)
        out["interaction_range"] = interaction_range(fg, periodic=False)
    if problem.problem_type == "QUANTUM":
        out["entanglement"] = entanglement_entropy(instance.x["state"], instance.x["n"])
        if is_spin_chain(problem):
            periodic = problem.equation.get("parameters", {}).get("boundary") == "periodic"
            out["interaction_range"] = interaction_range(fg, periodic=periodic)
            out["in_sector_gap"] = in_sector_gap(instance)
    return out


def spin_side(problem, instance) -> int:
    """WHAT: -1 or +1, the side of the critical value p_c on which a spin-chain sample lies.
    HOW: family = subfamily before "-" (e.g. "XXZ" in "XXZ-open"); cut = provenance modelling_map["cut"];
    find that Cut in harvest.sources.spin_more.FAMILIES; side = +1 if instance.info[cut.parameter] >
    cut.critical_value else -1 (the label rule of spin_more.py).
    WHY: Q4 spec 4.5 assigns laws per side."""
    from harvest.sources.spin_more import FAMILIES
    family = FAMILIES[problem.dataset.split("-")[0]]
    cut_name = problem.provenance["modelling_map"]["cut"]
    cut = next(c for c in family.cuts if c.name == cut_name)
    return 1 if instance.info[cut.parameter] > cut.critical_value else -1


def _median_iqr(values: list) -> tuple:
    """WHAT: (median, q75 - q25) of the non-NaN values; (NaN, NaN) if none."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return NAN, NAN
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return float(q50), float(q75 - q25)


def input_set_features(problems: list, exclude_flagged: bool = False) -> dict:
    """WHAT (2.3): the feature row of ONE distinct input set.
    INPUT: the problems of the input set (same inputs by construction; the first one's instances and
    operators are used, after checking the others have the same number of instances).
    HOW: for every instance i (skipping R6-flagged ones if exclude_flagged): fg = factor_graph(operator_i);
    f = instance_features(fg) | type_specific_features(...). Then for every feature: median and IQR over
    the instances (ignoring NaN). Spin chains: also the per-side medians (SIDE_FEATURES). Add n_inputs.
    OUTPUT: {"<feature>_median": ..., "<feature>_iqr": ..., "<side feature>": ..., "n_inputs": ...}."""
    first = problems[0]
    for other in problems[1:]:
        if len(other.instances) != len(first.instances):
            raise ValueError(f"input set mixes {first.task_id} and {other.task_id} with different sizes")
    skip = first.flagged() if exclude_flagged else [False] * len(first.instances)
    rows, sides = [], []
    for i, (instance, operator) in enumerate(zip(first.instances, first.operators)):
        if skip[i]:
            continue
        fg = factor_graph(operator)
        row = instance_features(fg)
        row.update(type_specific_features(first, instance, operator, fg))
        rows.append(row)
        sides.append(spin_side(first, instance) if is_spin_chain(first) else 0)
    out = {"n_inputs": len(rows)}
    for name in INSTANCE_FEATURES + TYPE_SPECIFIC_FEATURES:
        out[f"{name}_median"], out[f"{name}_iqr"] = _median_iqr([r[name] for r in rows])
    for name, key, side in (("entanglement_below", "entanglement", -1), ("entanglement_above", "entanglement", 1),
                            ("gap_below", "in_sector_gap", -1), ("gap_above", "in_sector_gap", 1)):
        out[name] = _median_iqr([r[key] for r, s in zip(rows, sides) if s == side])[0]
    return out


def _has_flags(task_id: str) -> bool:
    """WHAT: True if the problem has at least one R6-flagged sample (reads only its flags.json).
    WHY: build_feature_table(exclude_flagged=True) reuses the full row when nothing is flagged."""
    import analysis_common as ac
    with ac.closing(ac._connect()) as db:
        folder = db.execute("SELECT problem_dir FROM task WHERE task_id = ?", (task_id,)).fetchone()[0]
    path = ac.paths().data_final / folder / "flags.json"
    if not path.exists():
        return False
    flags = json.loads(path.read_text())
    return any(flags.get(kind) for kind in ac.FLAG_KINDS)


def build_feature_table(exclude_flagged: bool = False, max_seconds: float = None) -> "pandas.DataFrame":
    """WHAT: results/features.csv (or features_noflags.csv), one row per distinct input set.
    HOW: index, sets = analysis_common.attach_input_sets(load_index()); for each input set, load its
    problems (with operators) and call input_set_features; join the identity columns (input_set,
    problem_type, n, group_id, subfamily, tier, n_problems, task_ids) and save. Every row is cached in
    results/feature_cache/<input_set>[_noflags].json, so a run stopped by max_seconds (or a crash) resumes
    where it stopped; the table is written only when every input set is done (returns None otherwise).
    Progress is printed every 25 sets.
    WHY: the single feature table read by Q2, Q3 (mapped back to problems), Q4 and Q6."""
    import pandas as pd
    import analysis_common as ac
    start = time.time()
    _, sets = ac.attach_input_sets(ac.load_index())
    cache = ac.paths().results / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    suffix = "_noflags" if exclude_flagged else ""
    rows, done = [], 0
    for k, s in sets.iterrows():
        path = cache / f"{s['input_set']}{suffix}.json"
        plain = cache / f"{s['input_set']}.json"
        if exclude_flagged and not path.exists() and plain.exists() and not _has_flags(s["task_ids"].split(";")[0]):
            path.write_text(plain.read_text())             # nothing flagged: same row as the full table
        if path.exists():
            values = json.loads(path.read_text())
        else:
            if max_seconds is not None and time.time() - start > max_seconds:
                print(f"stopped after {done} new input sets ({k}/{len(sets)} done); run again to resume", flush=True)
                return None
            problems = [ac.load_problem(t) for t in s["task_ids"].split(";")[:1]]
            values = input_set_features(problems, exclude_flagged=exclude_flagged)
            path.write_text(json.dumps(values))
            done += 1
            if done % 25 == 0:
                print(f"features{suffix}: {k + 1}/{len(sets)} input sets", flush=True)
        rows.append({**s.to_dict(), **values})
    table = pd.DataFrame(rows)
    table.to_csv(ac.paths().results / f"features{suffix}.csv", index=False)
    return table
