"""Type QUANTUM: eight spin-chain families, several phase cuts, near/far windows (docs/HARVEST_V2.md §5).

Same idea as spin_generator.py: an input is the ground state of a chain Hamiltonian H(p) at a
parameter value p; the label says on which side of a phase transition p_c the value lies.
This module generalises it in three ways:

  1. more families (each one group):
       TFIM       -J Σ Z_i Z_{i+1} - h Σ X_i                       (as before)
       XXZ        Σ (X X + Y Y + Δ Z Z)                              (as before, range widened)
       BOND_ALT   Σ (1 + δ(-1)^i) S_i·S_{i+1}, open, even n          (as before)
       J1J2       Σ S_i·S_{i+1} + J₂ Σ S_i·S_{i+2}                    (as before)
       ANNNI      -Σ Z_i Z_{i+1} + κ Σ Z_i Z_{i+2} - h Σ X_i          axial next-nearest-neighbour Ising:
                  the κ term competes with the nearest-neighbour term ('frustration')
       DISORDERED -Σ J_i Z_i Z_{i+1} - Σ h_i X_i, J_i ~ U[0, 1], h_i ~ U[0, H], a fresh draw per
                  instance. Transition where the average log h equals the average log J
                  (Fisher 1992): here at H = 1.
       CLUSTER    -Σ X_{i-1} Z_i X_{i+1} + λ Σ Y_i Y_{i+1}   (Smacchia et al. 2011): a
                  symmetry-protected 'cluster' phase (λ < 1) vs an antiferromagnet (λ > 1).
       XY         -Σ [(1+γ)/2 X X + (1-γ)/2 Y Y] - h Σ Z_i, γ ~ U[0.2, 1] per instance
                  (anisotropic XY chain in a field, Barouch & McCoy 1971): ordered (h < 1) vs
                  paramagnet (h > 1). γ = 1 is exactly the TFIM (up to the names of the axes).
     Flags: CLUSTER and XY are related to TFIM by known maps (duality, free fermions); each keeps
     its own group and the robustness check can leave them out.

  2. several cuts per family (a *cut* = one transition and the parameter range around it).
     Every range is symmetric around its p_c, so both windows come out balanced:
       XXZ: Δ_c = 1 (XY / Néel) on [0, 2] and Δ_c = -1 (ferromagnet / XY) on [-2, 0]
       ANNNI: κ_c = 0.5 at h = 0.1 (ferromagnetic / 'antiphase' ++--), and h_c ≈ 0.775 at κ = 0.2
       (ferromagnet / paramagnet; value from the approximation of Beccaria et al. 2006, to check)

  3. two windows per cut, so each cut gives two problems on disjoint parameter sets:
       near   window <= |p - p_c| < window + band   (close to the transition: finite-size effects
                                                     blur the phases, harder)
       far    |p - p_c| >= window + band            (deep in each phase, easier)
     band = 3 · window. Values with |p - p_c| < window are never generated (as before).

Sizes: open chains n = 4...12; periodic chains even n only (odd rings frustrate antiferromagnets);
BOND_ALT open and even n only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calibrate import data_rng
from ..pauli import add_term, pauli_string
from ..records import Equation, Instance, Provenance
from ..screen import DatasetMeta
from .base import ProblemDraft, Source
from .spin_generator import bond_alternating, bonds, j1j2, tfim, xxz
from .symmetry_sectors import sector_ground_state


def annni(n, boundary, kappa, h):
    """WHAT: the ANNNI Hamiltonian (axial next-nearest-neighbour Ising chain), as a Pauli sum.
    HOW: H = -Σ Z_i Z_{i+1} + κ Σ Z_i Z_{i+2} - h Σ X_i: nearest-neighbour ZZ terms favour aligned
    spins, the κ next-nearest-neighbour terms (with a + sign) penalise spins two sites apart being
    aligned, and the X terms are the transverse field.
    OUTPUT: (Pauli sum, list of interaction strings = the κ terms).
    WHY: frustration: the two couplings compete, giving a phase (the 'antiphase' ++--) that the plain
    Ising chain does not have."""
    pauli, interaction = {}, []
    for i, j in bonds(n, boundary, 1):
        add_term(pauli, pauli_string(n, {i: "Z", j: "Z"}), -1.0)
    for i, j in bonds(n, boundary, 2):
        add_term(pauli, pauli_string(n, {i: "Z", j: "Z"}), kappa)
        interaction.append(pauli_string(n, {i: "Z", j: "Z"}))
    for i in range(n):
        add_term(pauli, pauli_string(n, {i: "X"}), -h)
    return pauli, interaction


def disordered_tfim(n, boundary, H, rng):
    """WHAT: the random transverse-field Ising chain, as a Pauli sum, with fresh randomness.
    HOW: H = -Σ J_i Z_i Z_{i+1} - Σ h_i X_i, every coupling J_i drawn uniformly in [0, 1] and every
    field h_i uniformly in [0, H], new draws for every instance.
    OUTPUT: (Pauli sum, [] (no interaction: still free-fermion), {"J": couplings, "h": fields}).
    WHY: the only spin family whose instances differ in more than one number; the transition at
    H = 1 follows from Fisher's 1992 criterion (mean of log h = mean of log J)."""
    pauli = {}
    couplings = rng.uniform(0, 1, n)
    fields = rng.uniform(0, H, n)
    for (i, j), J in zip(bonds(n, boundary), couplings):
        add_term(pauli, pauli_string(n, {i: "Z", j: "Z"}), -float(J))
    for i, h in enumerate(fields):
        add_term(pauli, pauli_string(n, {i: "X"}), -float(h))
    return pauli, [], {"J": couplings.tolist(), "h": fields.tolist()}


def cluster_ising(n, boundary, lam):
    """WHAT: the cluster-Ising Hamiltonian (Smacchia et al. 2011), as a Pauli sum.
    HOW: H = -Σ X_{i-1} Z_i X_{i+1} + λ Σ Y_i Y_{i+1}. The three-site 'cluster' terms run over every
    interior site (and wrap around on a ring); the YY terms over every bond.
    OUTPUT: (Pauli sum, the three-site terms as interaction strings).
    WHY: its λ < 1 phase is a symmetry-protected topological phase (no local order parameter tells it
    apart), unlike the other families' ordered phases."""
    pauli, interaction = {}, []
    triples = [(i - 1, i, i + 1) for i in range(1, n - 1)]
    if boundary == "periodic":
        triples += [(n - 2, n - 1, 0), (n - 1, 0, 1)]
    for a, b, c in triples:
        add_term(pauli, pauli_string(n, {a: "X", b: "Z", c: "X"}), -1.0)
        interaction.append(pauli_string(n, {a: "X", b: "Z", c: "X"}))
    for i, j in bonds(n, boundary):
        add_term(pauli, pauli_string(n, {i: "Y", j: "Y"}), lam)
    return pauli, interaction


def xy_chain(n, boundary, h, rng):
    """WHAT: the anisotropic XY chain in a field, as a Pauli sum, with a fresh anisotropy.
    HOW: γ drawn uniformly in [0.2, 1] for each instance; H = -Σ [(1+γ)/2 X X + (1-γ)/2 Y Y] - h Σ Z.
    OUTPUT: (Pauli sum, [] (free-fermion), {"gamma": γ}).
    WHY: a two-parameter family (h, γ) whose transition at h = 1 holds for every γ (Barouch & McCoy
    1971); γ = 1 is the TFIM with the axes renamed (tested), hence the flag."""
    gamma = float(rng.uniform(0.2, 1.0))
    pauli = {}
    for i, j in bonds(n, boundary):
        add_term(pauli, pauli_string(n, {i: "X", j: "X"}), -(1 + gamma) / 2)
        add_term(pauli, pauli_string(n, {i: "Y", j: "Y"}), -(1 - gamma) / 2)
    for i in range(n):
        add_term(pauli, pauli_string(n, {i: "Z"}), -h)
    return pauli, [], {"gamma": gamma}


@dataclass
class Cut:
    """WHAT: one phase transition of a family and the parameter range sampled around it.
    FIELDS: name; parameter (the name of the varied number p); critical_value p_c; low, high = the
    sampled range (symmetric around p_c, tested); window = values with |p - p_c| < window are never
    generated; phases = names of the phases below / above p_c; reference = where p_c comes from.
    WHY: a family can have several transitions (XXZ, ANNNI); each gives its own problems."""
    name: str
    parameter: str
    critical_value: float
    low: float
    high: float
    window: float
    phases: tuple
    reference: str


@dataclass
class Family:
    """WHAT: everything the generator needs to know about one spin-chain family.
    FIELDS: name; build(n, boundary, p, rng) -> (Pauli sum, interaction strings, extra info); cuts;
    symmetry used to fix a unique ground state ("SZ", "PARITY_X" or "PARITY_Z", see
    symmetry_sectors.py; tests check that it commutes with every Hamiltonian of the family);
    boundaries allowed ("open", "periodic"); even_only (only even n); nonlinearity ("NONE" for
    free-fermion families, "INTERACTION" otherwise); note (flags); extra.
    WHY: one table (FAMILIES) describes every family, so adding a family is adding one entry."""
    name: str
    build: object
    cuts: list
    symmetry: str
    boundaries: tuple = ("open", "periodic")
    even_only: bool = False
    nonlinearity: str = "INTERACTION"
    note: str = ""
    extra: dict = field(default_factory=dict)


def plain(builder):
    """WHAT: adapt a builder of the form builder(n, boundary, p) to the common form
    build(n, boundary, p, rng) -> (Pauli sum, interaction strings, {}).
    WHY: the four older builders (from spin_generator.py) use no randomness and return no extra
    information; this lets them sit in the same table as the new random ones."""
    def build(n, boundary, p, rng):
        """WHAT: (inner function of plain) call the wrapped builder; ignore rng; no extra info."""
        pauli, interaction = builder(n, boundary, p)
        return pauli, interaction, {}
    return build


FAMILIES = {
    "TFIM": Family("TFIM", plain(tfim), [Cut("h", "h", 1.0, 0.2, 1.8, 0.1, ("ordered", "disordered"),
                                             "exact (Pfeuty 1970)")], "PARITY_X", nonlinearity="NONE",
                   note="free-fermion (classically simulable)"),
    "XXZ": Family("XXZ", plain(xxz), [
        Cut("neel", "delta", 1.0, 0.0, 2.0, 0.15, ("critical XY", "Néel"), "exact (Bethe ansatz)"),
        Cut("ferro", "delta", -1.0, -2.0, 0.0, 0.15, ("ferromagnet", "critical XY"), "exact (Bethe ansatz)")], "SZ"),
    "BOND_ALT": Family("BOND_ALT", plain(bond_alternating), [
        Cut("dimer", "delta", 0.0, -0.8, 0.8, 0.1, ("odd-bond dimers", "even-bond dimers"), "exact (symmetry)")], "SZ",
        boundaries=("open",), even_only=True),
    "J1J2": Family("J1J2", plain(j1j2), [
        Cut("dimerisation", "j2", 0.2411, 0.0, 0.4822, 0.03, ("critical", "dimerised"),
            "Okamoto & Nomura 1992 (to check)")], "SZ"),
    "ANNNI": Family("ANNNI", None, [
        Cut("kappa", "kappa", 0.5, 0.0, 1.0, 0.1, ("ferromagnetic", "antiphase"), "h = 0.1; multicritical point κ = 0.5 at h = 0"),
        Cut("field", "h", 0.775, 0.1, 1.45, 0.1, ("ferromagnetic", "paramagnetic"),
            "κ = 0.2; h_c from Beccaria et al. 2006 approximation (to check)")], "PARITY_X"),
    "DISORDERED": Family("DISORDERED", disordered_tfim, [
        Cut("H", "H", 1.0, 0.2, 1.8, 0.15, ("ordered", "disordered"), "Fisher 1992 (mean log h = mean log J)")], "PARITY_X",
        nonlinearity="NONE", note="free-fermion with disorder; fresh couplings per instance"),
    "CLUSTER": Family("CLUSTER", plain(cluster_ising), [
        Cut("lambda", "lambda", 1.0, 0.0, 2.0, 0.1, ("cluster (SPT)", "antiferromagnet"), "Smacchia et al. 2011")], "PARITY_Z",
        note="free-fermion after Jordan-Wigner; related to TFIM by duality (flag)"),
    "XY": Family("XY", xy_chain, [
        Cut("h", "h", 1.0, 0.2, 1.8, 0.1, ("ordered", "paramagnet"), "Barouch & McCoy 1971")], "PARITY_Z",
        nonlinearity="NONE", note="free-fermion; γ = 1 is the TFIM (flag)"),
}


def annni_build(cut_name):
    """WHAT: the build function for one of ANNNI's two cuts.
    HOW: cut "kappa": vary κ with h fixed at 0.1; cut "field": vary h with κ fixed at 0.2.
    WHY: the two transitions are reached by varying different parameters."""
    def build(n, boundary, p, rng):
        """WHAT: (inner function of annni_build) the ANNNI Hamiltonian at parameter p for this cut."""
        if cut_name == "kappa":
            pauli, interaction = annni(n, boundary, kappa=p, h=0.1)
        else:
            pauli, interaction = annni(n, boundary, kappa=0.2, h=p)
        return pauli, interaction, {}
    return build


def window_values(cut: Cut, window: str, samples: int) -> list:
    """WHAT: `samples` parameter values spread evenly over the near or far region of a cut.
    HOW: a fine grid over [low, high] (20 points per wanted value), filtered by the region rule
    (near: window <= |p - p_c| < 4·window; far: |p - p_c| >= 4·window), then `samples` points taken at
    evenly spaced positions of the filtered list.
    WHY: near and far give two problems on disjoint parameter values: near is harder (finite-size
    effects blur the phases), far is easier."""
    band = 3 * cut.window
    fine = np.linspace(cut.low, cut.high, 20 * samples)
    distance = np.abs(fine - cut.critical_value)
    if window == "near":
        keep = fine[(distance >= cut.window) & (distance < cut.window + band)]
    else:
        keep = fine[distance >= cut.window + band]
    if len(keep) <= samples:
        return list(keep)
    positions = np.unique(np.round(np.linspace(0, len(keep) - 1, samples)).astype(int))
    return list(keep[positions])


class SpinFamilySource(Source):
    """WHAT: one problem per (family, boundary, cut, window, n).
    PARAMETERS: families, sizes, windows, samples (parameter values per problem), seed."""

    name = "spin-generator-2"

    def __init__(self, families=tuple(FAMILIES), sizes=range(4, 13), windows=("far", "near"), samples=160, seed=0):
        """WHAT: store the choices of this generator.
        INPUTS: families = which names of FAMILIES (default all 8); sizes = values of n; windows = "far"
        and/or "near"; samples = parameter values per problem; seed.
        WHY: the quick run and the full run use the same code with different settings."""
        self.families = list(families)
        self.sizes = list(sizes)
        self.windows = list(windows)
        self.samples = samples
        self.seed = seed

    def list_datasets(self) -> list:
        """WHAT: one 'dataset' per (family, boundary): e.g. "XXZ-open", "XXZ-periodic".
        HOW: a DatasetMeta with type QUANTUM, binary labels and the sizes offered.
        WHY: the pipeline screens datasets (rules S1-S4) before asking for their problems."""
        metas = []
        for name in self.families:
            for boundary in FAMILIES[name].boundaries:
                metas.append(DatasetMeta(name=f"{name}-{boundary}", problem_type="QUANTUM", n_samples=self.samples,
                                         label_kind="binary", sites_possible=self.sizes, licence="generated (ours)",
                                         notes={"family": name, "boundary": boundary}))
        return metas

    def fetch(self, meta):
        """WHAT: nothing (returns None).
        WHY: required by the Source interface; a generator has nothing to download."""
        return None

    def sizes_for(self, family: Family, boundary: str) -> list:
        """WHAT: the values of n used for one family and boundary.
        HOW: periodic chains and even_only families keep even n only; open chains keep every n.
        WHY: an odd ring frustrates antiferromagnetic order (the alternation cannot close), which would
        change the phases being learned."""
        if family.even_only or boundary == "periodic":
            return [n for n in self.sizes if n % 2 == 0]
        return list(self.sizes)

    def drafts(self, meta, raw):
        """WHAT: every candidate problem of one (family, boundary): one per (cut, window, n).
        HOW: for each cut and window, the parameter values from window_values; for each n, a separate
        random stream; for each value p: build H(p), compute its ground state inside the family's
        symmetry sector (symmetry_sectors.sector_ground_state: a unique, reproducible vector even when
        the lowest energy is degenerate) and label +1 when p > p_c.
        WHY: the core of this file: families x cuts x windows x sizes -> problems."""
        family = FAMILIES[meta.notes["family"]]
        boundary = meta.notes["boundary"]
        f_index = list(FAMILIES).index(family.name)
        for c_index, cut in enumerate(family.cuts):
            build = annni_build(cut.name) if family.name == "ANNNI" else family.build
            for w_index, window in enumerate(self.windows):
                values = window_values(cut, window, self.samples)
                for n in self.sizes_for(family, boundary):
                    rng = data_rng(self.seed, 500 + f_index, c_index, w_index, n, boundary == "periodic")
                    instances = []
                    for k, p in enumerate(values):
                        pauli, interaction, extra = build(n, boundary, float(p), rng)
                        ground = sector_ground_state(pauli, n, family.symmetry)
                        energy, state = ground["energy"], ground["state"]
                        x = {"n": n, "pauli": pauli, "state": state, "energy": energy,
                             "interaction_strings": interaction, "params": {cut.parameter: float(p), **extra}}
                        instances.append(Instance(x=x, y=1 if p > cut.critical_value else -1,
                                                  unit=f"{cut.parameter}={p:.6f}#{k}",
                                                  info={cut.parameter: float(p), "gap": ground["gap"],
                                                        "global_gap": ground["global_gap"],
                                                        "symmetry": family.symmetry, "sector": ground["sector"],
                                                        "tied_sectors": ground["tied_sectors"],
                                                        "residual_degeneracy": ground["residual_degeneracy"]}))
                    yield ProblemDraft(
                        task_name=f"{cut.name}:{window}", problem_type="QUANTUM", n=n, instances=instances,
                        label_rule=(f"+1 = {cut.phases[1]} ({cut.parameter} > {cut.critical_value}), "
                                    f"-1 = {cut.phases[0]}; window '{window}' (excluded |p - p_c| < {cut.window}, "
                                    f"band {3 * cut.window})"),
                        equation=Equation(form="SCHRODINGER", operator_kind="HAMILTONIAN", storage="PAULI_SUM",
                                          parameters={"boundary": boundary, "varied": cut.parameter},
                                          nonlinearity={"kind": family.nonlinearity, "term": family.note or
                                                        "interaction terms listed per instance", "strength": "model"}),
                        provenance=Provenance(source=self.name, dataset=meta.name, group_id=f"SPIN:{family.name}",
                                              origin="SYNTHETIC", licence="generated (ours)",
                                              citation=f"critical value: {cut.reference}",
                                              modelling_map={"domain": "condensed matter (spin chains)",
                                                             "map": "parameter -> ground state of H(parameter)",
                                                             "cut": cut.name, "window": window,
                                                             "range": [cut.low, cut.high], "flag": family.note}),
                        sites_have_identity=True)
