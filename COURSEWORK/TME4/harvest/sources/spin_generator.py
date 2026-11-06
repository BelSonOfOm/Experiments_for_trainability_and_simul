"""Type QUANTUM: phase learning in spin chains, generated offline by exact diagonalisation.

A spin chain is n qubits in a row (open chain) or a ring (periodic chain), with a Hamiltonian H
made of terms on neighbouring qubits. Its *ground state* is the eigenvector of H with the lowest
eigenvalue (the lowest energy). As a parameter p of H changes, the ground state can change
character abruptly at a critical value p_c: a *phase transition*. The learning problem: given the
ground state at some p, answer 'which side of p_c?'.

Exact diagonalisation = computing that eigenvector numerically from the 2ⁿ x 2ⁿ matrix of H. At
n = 12 the matrix is 4096 x 4096 but very sparse, so this takes about a second.

Four models (each is one *group* for Phase B). Pauli units: S·S below means XX + YY + ZZ.

  TFIM           H = -J Σ Z_i Z_{i+1} - h Σ X_i - h_z Σ Z_i     p = h,    p_c = 1 (exact, J = 1)
                 transverse-field Ising. Ordered (h < 1) vs disordered (h > 1). With h_z = 0 it is
                 a free-fermion model (classically simulable); h_z is the interaction dial.
  XXZ            H = Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})  p = Δ, p_c = 1 (exact)
                 critical 'XY' phase (|Δ| < 1) vs antiferromagnetic Néel phase (Δ > 1). The ΔZZ term
                 is the interaction (not free-fermion).
  BOND_ALT       H = Σ (1 + δ (-1)^i) S_i·S_{i+1}, OPEN chain only    p = δ, p_c = 0 (exact)
                 bond-alternating Heisenberg chain: strong bonds on even or on odd links. On an open
                 chain the two signs of δ are different phases (one has free spins at the ends).
                 On a ring they are the same state shifted by one site, so the label would only
                 record a translation: periodic is excluded on purpose.
  J1J2           H = Σ S_i·S_{i+1} + J₂ Σ S_i·S_{i+2}      p = J₂,  p_c ≈ 0.2411 (Okamoto & Nomura
                 1992; to check). Critical (J₂ < p_c) vs dimerised (J₂ > p_c). At n <= 12 the
                 transition is blurred: a deliberately hard problem.

Labels use the large-n critical value. Small chains shift the transition (finite-size effects),
so parameter values within `window` of p_c are not generated at all (recorded in the provenance).

Sites have an identity (site 4 is the same site for every parameter value), so M̄ is a plain
TRAIN average. Each parameter value is its own unit.

Cross-check still to do: our ground energies must match PennyLane qspin's for the same models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

from ..pauli import add_term, pauli_string, sum_matrix
from ..records import Equation, Instance, Provenance
from ..screen import DatasetMeta
from .base import ProblemDraft, Source


def bonds(n: int, boundary: str, distance: int = 1) -> list:
    """WHAT: the pairs of sites (i, j) at the given distance along the chain.
    HOW: open chain: (i, i + distance) for i + distance < n. Periodic chain (ring): also the
    pairs that wrap around; a pair is listed once even if reached twice (e.g. n = 4, distance 2).
    EXAMPLE: bonds(4, "open") == [(0, 1), (1, 2), (2, 3)];
             bonds(4, "periodic") == [(0, 1), (1, 2), (2, 3), (0, 3)]."""
    pairs = []
    for i in range(n):
        j = i + distance
        if j < n:
            pairs.append((i, j))
        elif boundary == "periodic" and distance < n:
            pair = tuple(sorted((i, j % n)))
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def heisenberg_term(pauli: dict, n: int, i: int, j: int, coupling: float) -> None:
    """WHAT: add coupling * (X_i X_j + Y_i Y_j + Z_i Z_j) to a Pauli sum."""
    for letter in "XYZ":
        add_term(pauli, pauli_string(n, {i: letter, j: letter}), coupling)


def tfim(n, boundary, h, J=1.0, h_z=0.0) -> tuple:
    """Transverse-field Ising chain. Returns (Pauli sum, list of interaction strings)."""
    pauli = {}
    for i, j in bonds(n, boundary):
        add_term(pauli, pauli_string(n, {i: "Z", j: "Z"}), -J)
    for i in range(n):
        add_term(pauli, pauli_string(n, {i: "X"}), -h)
    interaction = []
    if h_z != 0:
        for i in range(n):
            add_term(pauli, pauli_string(n, {i: "Z"}), -h_z)
            interaction.append(pauli_string(n, {i: "Z"}))
    return pauli, interaction


def xxz(n, boundary, delta) -> tuple:
    """XXZ chain. The Δ Z Z terms are the interaction."""
    pauli, interaction = {}, []
    for i, j in bonds(n, boundary):
        add_term(pauli, pauli_string(n, {i: "X", j: "X"}), 1.0)
        add_term(pauli, pauli_string(n, {i: "Y", j: "Y"}), 1.0)
        add_term(pauli, pauli_string(n, {i: "Z", j: "Z"}), delta)
        interaction.append(pauli_string(n, {i: "Z", j: "Z"}))
    return pauli, interaction


def bond_alternating(n, boundary, delta) -> tuple:
    """Bond-alternating Heisenberg chain (open only). Bond (i, i+1) has coupling 1 + δ(-1)^i.
    Its ZZ parts are the interaction (Heisenberg chains are not free-fermion)."""
    if boundary != "open":
        raise ValueError("BOND_ALT is defined on open chains only (see module docstring)")
    pauli, interaction = {}, []
    for i, j in bonds(n, "open"):
        heisenberg_term(pauli, n, i, j, 1.0 + delta * (-1) ** i)
        interaction.append(pauli_string(n, {i: "Z", j: "Z"}))
    return pauli, interaction


def j1j2(n, boundary, j2) -> tuple:
    """J1-J2 Heisenberg chain: nearest-neighbour coupling 1, next-nearest coupling J₂."""
    pauli, interaction = {}, []
    for i, j in bonds(n, boundary, 1):
        heisenberg_term(pauli, n, i, j, 1.0)
        interaction.append(pauli_string(n, {i: "Z", j: "Z"}))
    for i, j in bonds(n, boundary, 2):
        heisenberg_term(pauli, n, i, j, j2)
        interaction.append(pauli_string(n, {i: "Z", j: "Z"}))
    return pauli, interaction


@dataclass
class SpinModel:
    """Everything the generator needs to know about one model.

    name, builder        the model and the function building its Pauli sum
    parameter            name of the parameter p that is varied
    critical_value       p_c (the label is +1 when p > p_c)
    low, high            the range of p that is sampled
    window               values with |p - p_c| < window are skipped (finite-size ambiguity)
    boundaries           which boundary conditions are allowed
    phases               names of the phases below / above p_c (for the label rule sentence)
    reference            where p_c comes from
    """
    name: str
    builder: object
    parameter: str
    critical_value: float
    low: float
    high: float
    window: float
    boundaries: tuple
    phases: tuple
    reference: str


MODELS = {
    "TFIM": SpinModel("TFIM", tfim, "h", 1.0, 0.2, 2.0, 0.1, ("open", "periodic"),
                      ("ordered", "disordered"), "exact (Pfeuty 1970)"),
    "XXZ": SpinModel("XXZ", xxz, "delta", 1.0, 0.0, 2.5, 0.15, ("open", "periodic"),
                     ("critical XY", "Néel"), "exact (Bethe ansatz)"),
    "BOND_ALT": SpinModel("BOND_ALT", bond_alternating, "delta", 0.0, -0.8, 0.8, 0.1, ("open",),
                          ("odd-bond dimers", "even-bond dimers"), "exact (symmetry δ -> -δ)"),
    "J1J2": SpinModel("J1J2", j1j2, "j2", 0.2411, 0.0, 0.6, 0.05, ("open", "periodic"),
                      ("critical", "dimerised"), "Okamoto & Nomura 1992 (to check)"),
}


def ground_state(pauli: dict, n: int) -> tuple:
    """WHAT: (ground energy E₀, ground state ψ₀, gap E₁ - E₀) of a Hamiltonian.
    HOW: small matrices (n <= 6): full ('dense') diagonalisation, simple and exact.
         Larger: the Lanczos method (scipy's eigsh), which finds only the 2 lowest eigenvalues
         of a sparse matrix without ever building the full 2ⁿ x 2ⁿ table.
    NOTE: a degenerate ground state (gap ≈ 0) makes ψ₀ an arbitrary vector in the ground space;
    the gap is stored so such cases can be found later."""
    H = sum_matrix(pauli, n)
    if n <= 6:
        values, vectors = np.linalg.eigh(H.toarray())
    else:
        values, vectors = eigsh(H, k=2, which="SA")
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    return float(values[0]), vectors[:, 0], float(values[1] - values[0])


class SpinChainGenerator(Source):
    """WHAT: one problem per (model, boundary, n): `samples` parameter values evenly spaced over
    the model's range (minus the window around p_c), each giving one ground state.
    PARAMETERS: models = names from MODELS; sizes = values of n; samples = parameter values per
    problem (before the window is removed)."""

    name = "spin-generator"

    def __init__(self, models=tuple(MODELS), sizes=(4, 6, 8, 10, 12), samples=200):
        self.models = [MODELS[m] for m in models]
        self.sizes = list(sizes)
        self.samples = samples

    def list_datasets(self) -> list:
        """One 'dataset' per (model, boundary)."""
        metas = []
        for model in self.models:
            for boundary in model.boundaries:
                metas.append(DatasetMeta(name=f"{model.name}-{boundary}", problem_type="QUANTUM",
                                         n_samples=self.samples, label_kind="binary",
                                         sites_possible=self.sizes, licence="generated (ours)",
                                         notes={"model": model.name, "boundary": boundary}))
        return metas

    def fetch(self, meta):
        """Generators download nothing."""
        return None

    def drafts(self, meta, raw):
        """For every n: build H(p) on a grid of p, compute ground states, label by side of p_c."""
        model = MODELS[meta.notes["model"]]
        boundary = meta.notes["boundary"]
        grid = np.linspace(model.low, model.high, self.samples)
        grid = [p for p in grid if abs(p - model.critical_value) >= model.window]
        for n in self.sizes:
            instances = []
            for p in grid:
                pauli, interaction = model.builder(n, boundary, p)
                energy, state, gap = ground_state(pauli, n)
                x = {"n": n, "pauli": pauli, "state": state, "energy": energy,
                     "interaction_strings": interaction, "params": {model.parameter: float(p)}}
                y = 1 if p > model.critical_value else -1
                instances.append(Instance(x=x, y=y, unit=f"{model.parameter}={p:.6f}",
                                          info={model.parameter: float(p), "gap": gap}))
            yield ProblemDraft(
                task_name="phase",
                problem_type="QUANTUM",
                n=n,
                instances=instances,
                label_rule=(f"+1 = {model.phases[1]} ({model.parameter} > {model.critical_value}), "
                            f"-1 = {model.phases[0]}; |{model.parameter} - p_c| < {model.window} not sampled"),
                equation=Equation(form="SCHRODINGER", operator_kind="HAMILTONIAN", storage="PAULI_SUM",
                                  parameters={"boundary": boundary, "varied": model.parameter},
                                  nonlinearity={"kind": "NONE" if model.name == "TFIM" else "INTERACTION",
                                                "term": "ZZ part of the Heisenberg/XXZ couplings"
                                                        if model.name != "TFIM" else "none (h_z = 0)",
                                                "strength": "varies with the parameter"
                                                            if model.name == "XXZ" else "fixed"}),
                provenance=Provenance(source=self.name, dataset=meta.name, group_id=f"SPIN:{model.name}",
                                      origin="SYNTHETIC", licence="generated (ours)",
                                      citation=f"critical value: {model.reference}",
                                      modelling_map={"domain": "condensed matter (spin chains)",
                                                     "map": "parameter -> ground state of H(parameter)",
                                                     "grid": [model.low, model.high, self.samples],
                                                     "excluded_window": model.window}),
                sites_have_identity=True)
