"""Pauli strings and Pauli sums: how Hamiltonians are written down.

A *Pauli string* on n qubits is a word of n letters from {I, X, Y, Z}, e.g. "XZIZ".
Letter k says which 2x2 matrix acts on qubit k (qubit 0 is the leftmost letter):

    I = [[1, 0], [0, 1]]     (do nothing)
    X = [[0, 1], [1, 0]]     (bit flip)
    Y = [[0, -i], [i, 0]]
    Z = [[1, 0], [0, -1]]    (phase flip: +1 on |0>, -1 on |1>)

The string's matrix is the Kronecker (tensor) product of its letters' matrices: a
2ⁿ x 2ⁿ matrix. A *Pauli sum* is a linear combination of strings; we store it as a plain
dictionary {"XZIZ": 0.5, "ZZII": -1.0}. Every Hamiltonian on qubits can be written this way.

Why dictionaries and not matrices: a dictionary is readable (you can print a Hamiltonian
and see its terms), it is what the structure extraction works on (docs/STRUCTURE.md), and it
is small. Matrices are built from it only when needed (ground states, checks).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string(n: int, letters: dict) -> str:
    """WHAT: build the Pauli string with the given letters on the given qubits, I elsewhere.
    EXAMPLE: pauli_string(4, {0: "Z", 2: "Z"}) == "ZIZI".
    WHY: model builders (spin chains, SAT) describe terms as 'Z on qubit i and Z on qubit j';
    this turns that description into a string."""
    word = ["I"] * n
    for qubit, letter in letters.items():
        word[qubit] = letter
    return "".join(word)


def add_term(pauli_sum: dict, string: str, coefficient: float) -> None:
    """WHAT: add coefficient * string to a Pauli sum, in place, merging equal strings.
    HOW: if the string is already there, the coefficients are added; a coefficient that
    becomes (numerically) zero is removed.
    WHY: Hamiltonians are built term by term; two builders may produce the same string."""
    new_value = pauli_sum.get(string, 0.0) + coefficient
    if abs(new_value) < 1e-14:
        pauli_sum.pop(string, None)
    else:
        pauli_sum[string] = new_value


@lru_cache(maxsize=None)
def string_matrix(string: str) -> sp.csr_matrix:
    """WHAT: the 2ⁿ x 2ⁿ matrix of one Pauli string, as a sparse matrix.
    HOW: Kronecker product of the letters' 2x2 matrices, left to right (qubit 0 first).
    WHY: needed to compute ground states and to check equations.
    NOTE: sparse = only the non-zero entries are stored; a Pauli string has exactly one
    non-zero entry per row, so the result is small even at 12 qubits.
    CACHE (@lru_cache): the result is remembered, so a string used by many instances (every
    instance of a spin chain has the same strings) is only built once. The only speed-up in this
    first version, added because without it the quick run took several minutes per problem."""
    matrix = sp.csr_matrix(PAULI_MATRICES[string[0]])
    for letter in string[1:]:
        matrix = sp.kron(matrix, sp.csr_matrix(PAULI_MATRICES[letter]), format="csr")
    return matrix


def sum_matrix(pauli_sum: dict, n: int) -> sp.csr_matrix:
    """WHAT: the 2ⁿ x 2ⁿ matrix of a Pauli sum.
    HOW: add up coefficient * string_matrix(string) over all terms.
    WHY: the matrix of a Hamiltonian, for exact diagonalisation and checks."""
    total = sp.csr_matrix((2 ** n, 2 ** n), dtype=complex)
    for string, coefficient in pauli_sum.items():
        assert len(string) == n, f"string {string} does not have {n} letters"
        total = total + coefficient * string_matrix(string)
    return total


def weight(string: str) -> int:
    """WHAT: the number of qubits a string acts on (letters that are not I).
    EXAMPLE: weight("ZIZI") == 2.
    WHY: locality. A term of weight k couples k qubits."""
    return sum(1 for letter in string if letter != "I")


def support_qubits(string: str) -> tuple:
    """WHAT: the qubits a string acts on.
    EXAMPLE: support_qubits("ZIZI") == (0, 2)."""
    return tuple(k for k, letter in enumerate(string) if letter != "I")
