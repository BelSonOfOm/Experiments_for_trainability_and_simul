"""Ground states computed inside a fixed symmetry sector: a unique, reproducible vector even when the
lowest energy is shared by several states (degenerate ground state).

Why this file exists (audit of 9 Oct 2026): 24 % of the spin-chain samples had a degenerate or
quasi-degenerate ground state (energy gap below 1e-6). Then 'the' ground state is not defined: any
vector of the lowest-energy space is one, and the numerical solver returned an arbitrary one. Two
consequences: the stored input is not reproducible, and a degeneracy present on one side of a phase
transition only can act as a shortcut for a learning model.

The cause is always a symmetry. A *symmetry* is an operator S that commutes with the Hamiltonian H
(SH = HS); then H does not mix the *sectors* of S (the subspaces where S has a fixed value), and each
sector can be diagonalised on its own. A degeneracy between two states of different sectors (e.g.
'net spin +1/2' and 'net spin -1/2' on an odd chain) disappears once one sector is chosen. This is the
standard practice of exact diagonalisation ('symmetry-resolved' diagonalisation).

Symmetries used (qubit basis: |0> = spin up, Z = +1; |1> = spin down, Z = -1):

  "SZ"        total S_z = (number of 0s - number of 1s) / 2: conserved by XXZ, Heisenberg, J1-J2 and
              bond-alternating chains (every term is XX + YY or ZZ). Sectors: S_z = -n/2 ... n/2.
              Diagonal in the qubit basis: a sector is a set of basis states.
  "PARITY_Z"  P = Z_1 Z_2 ... Z_n (+1 when the number of 1s is even): conserved by the XY chain in a
              field and by the cluster-Ising chain. Diagonal: a sector is a set of basis states.
  "PARITY_X"  P = X_1 X_2 ... X_n (flips every spin): conserved by the transverse-field Ising chain, the
              ANNNI chain and the disordered Ising chain. Not diagonal in the qubit basis, so the
              Hamiltonian is first rotated by a Hadamard gate on every qubit (which turns X into Z, Z into
              X and Y into -Y); in the rotated basis P becomes Z_1...Z_n and the sectors are again sets
              of basis states. The vector found is rotated back.

Rule (stated, deterministic):
  1. compute the lowest energy of every sector;
  2. keep the sector(s) whose lowest energy equals the overall minimum, within TIE = 1e-9 x (1 +
     |minimum|);
  3. if several sectors tie: SZ -> the largest S_z; parity -> the even sector (P = +1);
  4. if the ground state is still degenerate inside the chosen sector (second-lowest energy within
     TIE): take the normalised projection, onto that ground space, of a fixed reference vector (the
     basis state of smallest index in the sector: |00...0> for SZ's largest sector and for even
     parity; for PARITY_X this is read in the rotated basis, i.e. the product state |++...+>): a
     definite, reproducible vector; such samples are flagged ("residual_degeneracy").
  Lanczos starts from a fixed vector (seeded), so the whole computation is reproducible bit for bit.

Returned with the state: the energy, the gap inside the chosen sector, the global gap (between the
lowest and second-lowest energies over all sectors), the sector chosen, the number of tied sectors and
the residual-degeneracy flag, so the analysis can always see what happened.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from ..pauli import sum_matrix

TIE = 1e-9
START_SEED = 0
"""Seed of the fixed Lanczos start vector (scipy otherwise starts from a random vector)."""
DENSE_LIMIT = 256
"""Sectors up to this dimension are diagonalised exactly (dense); larger ones with Lanczos (eigsh)."""


def rotate_x_to_z(pauli: dict) -> dict:
    """WHAT: the Pauli sum of W H W, with W = a Hadamard gate on every qubit.
    HOW: the Hadamard gate maps X -> Z, Z -> X, Y -> -Y; so each string has X and Z swapped, and its
    coefficient changes sign once per Y.
    WHY: turns the spin-flip symmetry X_1...X_n into the diagonal parity Z_1...Z_n."""
    swap = str.maketrans({"X": "Z", "Z": "X"})
    return {s.translate(swap): c * (-1) ** s.count("Y") for s, c in pauli.items()}


def hadamard_all(vector: np.ndarray, n: int) -> np.ndarray:
    """WHAT: apply a Hadamard gate to every qubit of a state vector (the Walsh-Hadamard transform).
    HOW: view the vector as an n-index array of shape (2, ..., 2) and apply the 2x2 Hadamard matrix
    along each index in turn (n small matrix products instead of one 2^n x 2^n product).
    WHY: brings a vector found in the rotated basis back to the qubit basis (W is its own inverse)."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    t = np.asarray(vector).reshape((2,) * n)
    for axis in range(n):
        t = np.moveaxis(np.tensordot(h, t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


def sectors(symmetry: str, n: int) -> list:
    """WHAT: the sectors as (label, sorted array of basis-state indices), in a fixed order.
    HOW: basis state b (an integer) is read in binary, qubit 0 = leftmost bit. SZ sectors by the
    number of 1s (label = S_z = (n - 2·ones)/2); parity sectors by the parity of the number of 1s
    (label +1 = even)."""
    ones = np.array([bin(b).count("1") for b in range(2 ** n)])
    if symmetry == "SZ":
        return [((n - 2 * k) / 2, np.flatnonzero(ones == k)) for k in range(n + 1)]
    if symmetry in ("PARITY_Z", "PARITY_X"):
        return [(+1, np.flatnonzero(ones % 2 == 0)), (-1, np.flatnonzero(ones % 2 == 1))]
    raise ValueError(f"unknown symmetry {symmetry}")


def start_vector(dim: int) -> np.ndarray:
    """WHAT: the fixed Lanczos start vector of a given dimension (Gaussian entries, seed START_SEED).
    WHY: a random start would make results differ by rounding between runs; a generic fixed vector
    has a nonzero overlap with the ground state, which Lanczos needs."""
    return np.random.default_rng(START_SEED).normal(size=dim)


def lanczos(block: sp.csr_matrix, k: int) -> tuple:
    """WHAT: the k lowest eigenpairs of a symmetric sparse block, sorted by eigenvalue.
    HOW: Lanczos (eigsh) from the fixed start vector; if it does not converge, exact dense
    diagonalisation instead (np.linalg.eigh on the full block).
    WHY (10 Oct 2026 harvest): one XY chain (n = 12, periodic, h = 0.21) has its 2nd, 3rd and 4th
    levels within 1e-6 of each other; Lanczos could not separate the 2nd from the cluster and stopped
    with an error, which rejected the whole problem. Dense diagonalisation is exact and, for the
    largest block we use (2048 x 2048), takes about a second; it is only used when Lanczos fails."""
    try:
        values, vectors = eigsh(block, k=k, which="SA", tol=1e-12, v0=start_vector(block.shape[0]))
    except ArpackNoConvergence:
        values, vectors = np.linalg.eigh(block.toarray())
        values, vectors = values[:k], vectors[:, :k]
    order = np.argsort(values)
    return values[order], vectors[:, order]


def lowest_two(block: sp.csr_matrix) -> tuple:
    """WHAT: (two lowest eigenvalues, their eigenvectors) of a symmetric sparse block; for a block of
    size 1, the second eigenvalue is +infinity.
    HOW: dense diagonalisation up to DENSE_LIMIT, Lanczos above (function lanczos)."""
    dim = block.shape[0]
    if dim == 1:
        value = float(block.toarray()[0, 0].real)
        return np.array([value, np.inf]), np.ones((1, 1))
    if dim <= DENSE_LIMIT:
        values, vectors = np.linalg.eigh(block.toarray())
        return values[:2], vectors[:, :2]
    return lanczos(block, 2)


def degenerate_space(block: sp.csr_matrix, energy: float, max_k: int = 8) -> np.ndarray:
    """WHAT: an orthonormal basis (columns) of all eigenvectors of the block with eigenvalue within TIE
    of `energy`.
    HOW: compute the lowest k eigenpairs for k = 2, 4, 8 (dense when small) until one eigenvalue lies
    above energy + tolerance, then keep the ones below."""
    dim = block.shape[0]
    tol = TIE * (1 + abs(energy))
    if dim <= DENSE_LIMIT:
        values, vectors = np.linalg.eigh(block.toarray())
        return vectors[:, values <= energy + tol]
    k = 2
    while True:
        k_eff = min(k, dim - 1)
        values, vectors = lanczos(block, k_eff)
        if values[-1] > energy + tol or k_eff >= min(max_k, dim - 1):
            return vectors[:, values <= energy + tol]
        k *= 2


def sector_ground_state(pauli: dict, n: int, symmetry: str) -> dict:
    """WHAT: the ground state chosen by the rule of the module explanation.
    OUTPUT: {"energy", "state" (real vector of length 2^n, norm 1), "gap" (inside the chosen sector),
             "global_gap" (over all sectors), "sector", "tied_sectors", "residual_degeneracy"}.
    HOW: build the (possibly rotated) Hamiltonian matrix; for each sector take the block of rows and
    columns of its basis states and find its two lowest energies; choose the sector; resolve any
    degeneracy left inside it with the fixed reference vector; embed the sector vector back into the
    full space (rotating back for PARITY_X)."""
    rotated = symmetry == "PARITY_X"
    H = sum_matrix(rotate_x_to_z(pauli) if rotated else pauli, n).tocsr()
    if H.nnz and np.abs(H.data.imag).max() > 1e-12:     # all 8 families are real (checked 9 Oct 2026)
        raise ValueError("complex Hamiltonian: the sector code expects real matrix entries")
    H = H.real.tocsr()
    if H.nnz and abs(H - H.T).max() > 1e-10:
        raise ValueError("Hamiltonian is not symmetric")

    results = []
    for label, index in sectors(symmetry, n):
        block = H[index][:, index]
        values, vectors = lowest_two(block)
        results.append((label, index, block, values, vectors))
    energies = sorted(v for r in results for v in r[3] if np.isfinite(v))
    e0 = energies[0]
    global_gap = energies[1] - energies[0] if len(energies) > 1 else np.inf
    tol = TIE * (1 + abs(e0))
    tied = [r for r in results if r[3][0] <= e0 + tol]
    chosen = max(tied, key=lambda r: r[0])             # SZ: largest S_z; parity: +1 before -1
    label, index, block, values, vectors = chosen

    residual = values[1] <= values[0] + tol
    if residual:
        space = degenerate_space(block, values[0])
        reference = np.zeros(len(index))
        reference[0] = 1.0
        coeffs = space.T @ reference
        if np.linalg.norm(coeffs) < 1e-8:              # reference orthogonal to the space: use the first vector
            vec = space[:, 0]
        else:
            vec = space @ coeffs
        vec = vec / np.linalg.norm(vec)
    else:
        vec = vectors[:, 0]
    # fix the overall sign: largest-magnitude component positive (so the vector is fully reproducible)
    vec = vec * np.sign(vec[np.argmax(np.abs(vec))])

    state = np.zeros(2 ** n)
    state[index] = vec
    if rotated:
        state = hadamard_all(state, n)
        state = state * np.sign(state[np.argmax(np.abs(state))])
    return {"energy": float(values[0]), "state": state, "gap": float(values[1] - values[0]),
            "global_gap": float(global_gap), "sector": label, "tied_sectors": len(tied),
            "residual_degeneracy": bool(residual)}
