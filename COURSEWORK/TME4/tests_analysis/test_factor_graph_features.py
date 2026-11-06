"""Known-answer tests for factor_graph.py and features.py (run: python -m pytest tests_analysis -q, from TME4/).
Each test states the hand-computed value it checks."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_common as ac  # noqa: E402

ac.import_harvest()
import factor_graph as fgm  # noqa: E402
import features as ft  # noqa: E402


def op(kind, n, matrix=None, pauli=None, factors=None):
    """A stand-in for harvest.records.Operator with the fields factor_graph reads."""
    return SimpleNamespace(kind=kind, n=n, matrix=matrix, pauli=pauli, factors=factors or {})


def path_laplacian(n):
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i] += 1; L[i + 1, i + 1] += 1; L[i, i + 1] -= 1; L[i + 1, i] -= 1
    return L


def test_path_graph_factors():
    fg = fgm.factor_graph(op("LAPLACIAN", 4, matrix=path_laplacian(4)))
    assert fg.factors == [(0, 1), (1, 2), (2, 3)] and fg.weights == [1.0, 1.0, 1.0]


def test_precision_partial_correlation():
    fg = fgm.factor_graph(op("PRECISION", 2, matrix=np.array([[2.0, -1.0], [-1.0, 2.0]])))
    assert fg.factors == [(0, 1)] and fg.weights == [1.0]


def test_identity_precision_has_no_factor_and_nan_features():
    fg = fgm.factor_graph(op("PRECISION", 3, matrix=np.eye(3)))
    f = ft.instance_features(fg)
    assert fg.n_factors() == 0 and f["support_density"] == 0 and math.isnan(f["spec_entropy"])


def test_xor3_vs_pairwise_triangle():
    xor = fgm.factor_graph(op("CLAUSE_PENALTY", 3, pauli={"III": 1.0, "ZZZ": 0.5}))
    tri = fgm.factor_graph(op("HAMILTONIAN", 3, pauli={"ZZI": 1.0, "IZZ": 1.0, "ZIZ": 1.0}))
    assert xor.factor_sizes() == [3] and tri.factor_sizes() == [2, 2, 2]
    assert ft.support_density(xor) == ft.support_density(tri) == 1.0       # same pairs, different factors


def test_tfim_open_3():
    fg = fgm.factor_graph(op("HAMILTONIAN", 3, pauli={"ZZI": -1.0, "IZZ": -1.0, "XII": -1.0, "IXI": -1.0, "IIX": -1.0}))
    assert sorted(zip(fg.factors, fg.labels)) == sorted([((0, 1), "ZZ"), ((1, 2), "ZZ"), ((0,), "X"), ((1,), "X"), ((2,), "X")])
    assert sorted(fg.roles) == sorted([("Z", "Z"), ("Z", "Z"), ("X",), ("X",), ("X",)])
    assert fg.site_weights == [1.0, 1.0, 1.0]


def test_term_sizes():
    fg = fgm.FactorGraph(n_sites=4, factors=[(0, 1), (1, 2), (0, 1, 3)], weights=[1, 1, 1], labels=["a"] * 3, site_weights=[0] * 4)
    s = ft.term_size_stats(fg)
    assert s["term_size_mean"] == pytest.approx(7 / 3) and s["frac_size_2"] == pytest.approx(2 / 3)


def test_support_density_open_chain():
    fg = fgm.factor_graph(op("LAPLACIAN", 4, matrix=path_laplacian(4)))
    assert ft.support_density(fg) == pytest.approx(2 / 4)


def ring(n):
    L = path_laplacian(n)
    L[0, 0] += 1; L[-1, -1] += 1; L[0, -1] -= 1; L[-1, 0] -= 1
    return L


def test_symmetry_ring_open_disordered():
    assert ft.symmetry_features(fgm.factor_graph(op("LAPLACIAN", 6, matrix=ring(6)))) == {"sym_classes": 1.0, "sym_fraction": 1.0}
    assert ft.symmetry_features(fgm.factor_graph(op("LAPLACIAN", 4, matrix=path_laplacian(4))))["sym_classes"] == 2.0
    pauli = {"ZZII": -1.0, "IZZI": -1.3, "IIZZ": -0.7, "XIII": -0.2, "IXII": -0.9, "IIXI": -0.4, "IIIX": -0.6}
    assert ft.symmetry_features(fgm.factor_graph(op("HAMILTONIAN", 4, pauli=pauli)))["sym_classes"] == 4.0


def test_spectral_single_edge():
    fg = fgm.FactorGraph(n_sites=2, factors=[(0, 1)], weights=[1.0], labels=["edge"], site_weights=[0, 0])
    s = ft.spectral_features(fg)
    assert s["spec_entropy"] == pytest.approx(0.5794, abs=1e-4)
    assert s["participation_ratio"] == pytest.approx(0.6) and s["alg_connectivity"] == pytest.approx(1.0)


def test_entanglement():
    product = np.zeros(8); product[0] = 1
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert ft.entanglement_entropy(product, 3) == pytest.approx(0.0, abs=1e-12)
    assert ft.entanglement_entropy(bell, 2) == pytest.approx(math.log(2))


def test_holm():
    assert ac.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_effective_rank_identity():
    assert ft.effective_rank(op("PRECISION", 4, matrix=np.eye(4), factors={"R": np.eye(4)})) == pytest.approx(4.0)


def test_bipartite_spectrum_matches_dense():
    rng = np.random.default_rng(0)
    for trial in range(5):
        factors = [tuple(sorted(rng.choice(6, size=rng.integers(1, 4), replace=False).tolist())) for _ in range(9)]
        fg = fgm.FactorGraph(n_sites=7, factors=factors, weights=list(rng.uniform(0.1, 1, 9)), labels=["a"] * 9, site_weights=[0] * 7)
        dense = np.sort(np.clip(np.linalg.eigvalsh(fg.bipartite_laplacian(normalized=True)), 0, None))
        assert np.allclose(ft.bipartite_spectrum(fg), dense, atol=1e-9)


def test_cluster_ring_is_fully_symmetric():
    """Periodic cluster-Ising, n = 6: X Z X on every (i-1, i, i+1) including the two wrapping terms, and
    Y Y on every bond: translation-invariant, so 1 site class (the bug of 11 Oct gave 6)."""
    n, pauli = 6, {}
    for i in range(n):
        word = ["I"] * n
        word[(i - 1) % n], word[i], word[(i + 1) % n] = "X", "Z", "X"
        pauli["".join(word)] = -1.0
        word = ["I"] * n
        word[i], word[(i + 1) % n] = "Y", "Y"
        pauli["".join(word)] = 0.5
    fg = fgm.factor_graph(op("HAMILTONIAN", n, pauli=pauli))
    assert ft.symmetry_features(fg)["sym_classes"] == 1.0
