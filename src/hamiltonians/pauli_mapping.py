"""
Jordan-Wigner transformation for molecular and spin Hamiltonians.

Converts second-quantized fermionic Hamiltonians to qubit (Pauli) representation
for use with quantum circuits (CUDA-Q exp_pauli).

Also provides direct Pauli construction for spin Hamiltonians (Heisenberg model)
matching the NVIDIA CUDA-Q SKQD tutorial format.

References:
    - Jordan & Wigner, Z. Phys. 47, 631 (1928)
    - NVIDIA CUDA-Q SKQD tutorial: nvidia.github.io/cuda-quantum/latest/applications/python/skqd.html
"""

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pauli algebra
# ---------------------------------------------------------------------------

# Single-qubit Pauli multiplication table: (row, col) -> (phase, result)
# Phase is a power of i: 0=1, 1=i, 2=-1, 3=-i
_PAULI_MULT = {
    ("I", "I"): (0, "I"), ("I", "X"): (0, "X"), ("I", "Y"): (0, "Y"), ("I", "Z"): (0, "Z"),
    ("X", "I"): (0, "X"), ("X", "X"): (0, "I"), ("X", "Y"): (1, "Z"), ("X", "Z"): (3, "Y"),
    ("Y", "I"): (0, "Y"), ("Y", "X"): (3, "Z"), ("Y", "Y"): (0, "I"), ("Y", "Z"): (1, "X"),
    ("Z", "I"): (0, "Z"), ("Z", "X"): (1, "Y"), ("Z", "Y"): (3, "X"), ("Z", "Z"): (0, "I"),
}

_PHASE_TO_COMPLEX = {0: 1.0, 1: 1j, 2: -1.0, 3: -1j}


def _multiply_pauli_strings(s1: str, s2: str) -> Tuple[complex, str]:
    """
    Multiply two multi-qubit Pauli strings site-by-site.

    Returns (phase, result_string) where phase is a complex number.
    """
    assert len(s1) == len(s2), f"Pauli string length mismatch: {len(s1)} vs {len(s2)}"
    total_phase = 0  # accumulated power of i
    result = []
    for p1, p2 in zip(s1, s2):
        phase_power, res = _PAULI_MULT[(p1, p2)]
        total_phase = (total_phase + phase_power) % 4
        result.append(res)
    return _PHASE_TO_COMPLEX[total_phase], "".join(result)


class PauliSum:
    """
    Sparse representation of a sum of weighted Pauli strings.

    Stores {pauli_string: complex_coefficient} and supports addition,
    scalar multiplication, and operator multiplication (Pauli product).
    """

    __slots__ = ("terms", "n_qubits")

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.terms: Dict[str, complex] = {}

    def add_term(self, coeff: complex, pauli_string: str) -> None:
        """Add coeff * pauli_string to this operator."""
        if abs(coeff) < 1e-18:
            return
        assert len(pauli_string) == self.n_qubits
        self.terms[pauli_string] = self.terms.get(pauli_string, 0.0) + coeff

    def __iadd__(self, other: "PauliSum") -> "PauliSum":
        for ps, c in other.terms.items():
            self.terms[ps] = self.terms.get(ps, 0.0) + c
        return self

    def scale(self, scalar: complex) -> "PauliSum":
        """Return a new PauliSum scaled by scalar."""
        result = PauliSum(self.n_qubits)
        for ps, c in self.terms.items():
            result.terms[ps] = c * scalar
        return result

    def multiply(self, other: "PauliSum") -> "PauliSum":
        """Operator product: self * other."""
        result = PauliSum(self.n_qubits)
        for ps1, c1 in self.terms.items():
            for ps2, c2 in other.terms.items():
                phase, ps_result = _multiply_pauli_strings(ps1, ps2)
                coeff = c1 * c2 * phase
                if abs(coeff) > 1e-18:
                    result.terms[ps_result] = result.terms.get(ps_result, 0.0) + coeff
        return result

    def simplify(self, threshold: float = 1e-15) -> None:
        """Remove near-zero terms in place."""
        self.terms = {ps: c for ps, c in self.terms.items() if abs(c) > threshold}

    def to_real_lists(self, threshold: float = 1e-8) -> Tuple[List[float], List[str]]:
        """
        Export as (coefficients, pauli_words) with real coefficients.

        Drops the all-identity term (constant energy offset returned separately).
        Raises if any coefficient has significant imaginary part.

        Returns:
            (coefficients, pauli_words, constant) where constant is the
            coefficient of the all-identity string.
        """
        self.simplify(threshold)
        identity = "I" * self.n_qubits

        coefficients: List[float] = []
        pauli_words: List[str] = []
        constant = 0.0

        for ps, c in self.terms.items():
            if abs(c.imag) > threshold:
                raise ValueError(
                    f"Pauli term '{ps}' has imaginary coefficient {c}. "
                    "Molecular Hamiltonians with real integrals should yield real Pauli coefficients."
                )
            real_c = c.real
            if abs(real_c) < threshold:
                continue
            if ps == identity:
                constant = real_c
            else:
                coefficients.append(real_c)
                pauli_words.append(ps)

        return coefficients, pauli_words, constant


# ---------------------------------------------------------------------------
# Jordan-Wigner elementary operators
# ---------------------------------------------------------------------------

def _identity_string(n_qubits: int) -> str:
    return "I" * n_qubits


def _single_pauli(n_qubits: int, qubit: int, pauli: str) -> str:
    """Create a Pauli string with `pauli` on `qubit` and I elsewhere."""
    chars = ["I"] * n_qubits
    chars[qubit] = pauli
    return "".join(chars)


def _z_chain_string(n_qubits: int, start: int, end: int) -> str:
    """Create a Pauli string with Z on qubits [start, end) and I elsewhere."""
    chars = ["I"] * n_qubits
    for q in range(start, end):
        chars[q] = "Z"
    return "".join(chars)


def one_body_op(p: int, q: int, n_qubits: int) -> PauliSum:
    """
    Jordan-Wigner transformation of a^dag_p a_q.

    For p == q: (I - Z_p) / 2  (number operator)
    For p < q:  1/4 * (XX + YY + iXY - iYX) * Z_chain(p+1..q-1)
    For p > q:  Hermitian conjugate of (q, p) case
    """
    result = PauliSum(n_qubits)

    if p == q:
        # Number operator: (I - Z_p) / 2
        result.add_term(0.5, _identity_string(n_qubits))
        result.add_term(-0.5, _single_pauli(n_qubits, p, "Z"))
        return result

    # Ensure p < q, handle conjugate
    if p > q:
        conj = one_body_op(q, p, n_qubits)
        # Hermitian conjugate: (a†_q a_p)† = a†_p a_q
        # For real Hamiltonians, h_pq = h_qp, so we just return the conjugate
        result_conj = PauliSum(n_qubits)
        for ps, c in conj.terms.items():
            result_conj.add_term(c.conjugate(), ps)
        return result_conj

    # p < q case
    # Build base strings with Z chain between p+1 and q-1
    base = ["I"] * n_qubits
    for k in range(p + 1, q):
        base[k] = "Z"

    # XX term
    xx = list(base)
    xx[p] = "X"
    xx[q] = "X"
    result.add_term(0.25, "".join(xx))

    # YY term
    yy = list(base)
    yy[p] = "Y"
    yy[q] = "Y"
    result.add_term(0.25, "".join(yy))

    # XY term (coefficient +i/4)
    xy = list(base)
    xy[p] = "X"
    xy[q] = "Y"
    result.add_term(0.25j, "".join(xy))

    # YX term (coefficient -i/4)
    yx = list(base)
    yx[p] = "Y"
    yx[q] = "X"
    result.add_term(-0.25j, "".join(yx))

    return result


def two_body_op(p: int, q: int, r: int, s: int, n_qubits: int) -> PauliSum:
    """
    Jordan-Wigner transformation of a^dag_p a^dag_r a_s a_q.

    Computed as product of one-body operators with sign correction:
        a†_p a†_r a_s a_q = (a†_p a_q)(a†_r a_s) - delta_{qr} (a†_p a_s)
    """
    # Compute (a†_p a_q) * (a†_r a_s)
    op_pq = one_body_op(p, q, n_qubits)
    op_rs = one_body_op(r, s, n_qubits)
    result = op_pq.multiply(op_rs)

    # Subtract delta_{qr} * (a†_p a_s) correction
    if q == r:
        op_ps = one_body_op(p, s, n_qubits)
        for ps_str, c in op_ps.terms.items():
            result.terms[ps_str] = result.terms.get(ps_str, 0.0) - c

    result.simplify()
    return result


# ---------------------------------------------------------------------------
# Full Hamiltonian transformations
# ---------------------------------------------------------------------------

def molecular_hamiltonian_to_pauli(
    h1e: np.ndarray,
    h2e: np.ndarray,
    nuclear_repulsion: float,
    n_orbitals: int,
) -> Tuple[List[float], List[str], float]:
    """
    Convert molecular integrals to Pauli representation via Jordan-Wigner.

    Qubit ordering matches MolecularHamiltonian convention:
        qubits 0..n_orb-1 = alpha spin-orbitals
        qubits n_orb..2*n_orb-1 = beta spin-orbitals

    Args:
        h1e: One-electron integrals (n_orbitals, n_orbitals) in MO basis
        h2e: Two-electron integrals (n_orb, n_orb, n_orb, n_orb) chemist notation (pq|rs)
        nuclear_repulsion: Nuclear repulsion energy
        n_orbitals: Number of spatial orbitals

    Returns:
        (coefficients, pauli_words, constant_energy) where constant_energy
        includes nuclear repulsion and the identity Pauli term.
    """
    n_qubits = 2 * n_orbitals
    H_pauli = PauliSum(n_qubits)

    # Nuclear repulsion as identity term
    H_pauli.add_term(nuclear_repulsion, _identity_string(n_qubits))

    # --- One-body terms ---
    # H_1 = sum_{pq,sigma} h_pq a†_{p,sigma} a_{q,sigma}
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            if abs(h1e[p, q]) < 1e-15:
                continue
            # Alpha spin: qubit indices p, q
            op_alpha = one_body_op(p, q, n_qubits)
            H_pauli += op_alpha.scale(h1e[p, q])

            # Beta spin: qubit indices p + n_orbitals, q + n_orbitals
            op_beta = one_body_op(p + n_orbitals, q + n_orbitals, n_qubits)
            H_pauli += op_beta.scale(h1e[p, q])

    # --- Two-body terms ---
    # H_2 = 1/2 sum_{pqrs,sigma,tau} (pq|rs) a†_{p,sigma} a†_{r,tau} a_{s,tau} a_{q,sigma}
    # Chemist notation: (pq|rs) = <pr|qs> (physicist)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coeff = 0.5 * h2e[p, q, r, s]
                    if abs(coeff) < 1e-15:
                        continue

                    # alpha-alpha: a†_{p,a} a†_{r,a} a_{s,a} a_{q,a}
                    op_aa = two_body_op(p, q, r, s, n_qubits)
                    H_pauli += op_aa.scale(coeff)

                    # beta-beta: a†_{p,b} a†_{r,b} a_{s,b} a_{q,b}
                    pb, qb, rb, sb = p + n_orbitals, q + n_orbitals, r + n_orbitals, s + n_orbitals
                    op_bb = two_body_op(pb, qb, rb, sb, n_qubits)
                    H_pauli += op_bb.scale(coeff)

                    # alpha-beta: a†_{p,a} a†_{r,b} a_{s,b} a_{q,a}
                    op_ab = two_body_op(p, q, r + n_orbitals, s + n_orbitals, n_qubits)
                    H_pauli += op_ab.scale(coeff)

                    # beta-alpha: a†_{p,b} a†_{r,a} a_{s,a} a_{q,b}
                    op_ba = two_body_op(p + n_orbitals, q + n_orbitals, r, s, n_qubits)
                    H_pauli += op_ba.scale(coeff)

    H_pauli.simplify()

    coefficients, pauli_words, constant = H_pauli.to_real_lists()

    # constant includes nuclear repulsion + identity Pauli terms
    return coefficients, pauli_words, constant


def heisenberg_hamiltonian_pauli(
    n_spins: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    hx: np.ndarray = None,
    hy: np.ndarray = None,
    hz: np.ndarray = None,
) -> Tuple[List[float], List[str], float]:
    """
    Heisenberg spin chain Hamiltonian in Pauli representation.

    H = sum_i [Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1}]
      + sum_i [hx_i X_i + hy_i Y_i + hz_i Z_i]

    Matches the NVIDIA CUDA-Q SKQD tutorial format exactly.

    Args:
        n_spins: Number of spins (qubits)
        Jx, Jy, Jz: Coupling constants
        hx, hy, hz: External field arrays (length n_spins), default ones

    Returns:
        (coefficients, pauli_words, constant) matching CUDA-Q format
    """
    if hx is None:
        hx = np.ones(n_spins)
    if hy is None:
        hy = np.ones(n_spins)
    if hz is None:
        hz = np.ones(n_spins)

    coefficients = []
    pauli_words = []

    # Nearest-neighbor interactions
    for i in range(n_spins - 1):
        j = i + 1
        if abs(Jx) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "X"
            ps[j] = "X"
            coefficients.append(Jx)
            pauli_words.append("".join(ps))
        if abs(Jy) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Y"
            ps[j] = "Y"
            coefficients.append(Jy)
            pauli_words.append("".join(ps))
        if abs(Jz) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Z"
            ps[j] = "Z"
            coefficients.append(Jz)
            pauli_words.append("".join(ps))

    # External fields
    for i in range(n_spins):
        if abs(hx[i]) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "X"
            coefficients.append(hx[i])
            pauli_words.append("".join(ps))
        if abs(hy[i]) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Y"
            coefficients.append(hy[i])
            pauli_words.append("".join(ps))
        if abs(hz[i]) > 1e-15:
            ps = ["I"] * n_spins
            ps[i] = "Z"
            coefficients.append(hz[i])
            pauli_words.append("".join(ps))

    return coefficients, pauli_words, 0.0
