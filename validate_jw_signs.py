"""
JW Sign Consistency Validation Script.

This script validates that Jordan-Wigner sign computations are consistent:
For each connected pair (i,j), H[i,j] computed from get_connections(j)
should equal H[j,i] computed from get_connections(i).

If they differ, the matrix cannot be Hermitian, which causes variational
principle violations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from math import comb

try:
    from pyscf import gto, scf, ao2mo
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("PySCF not available")
    sys.exit(1)

from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals


def create_test_hamiltonian(molecule="lih"):
    """Create a test Hamiltonian."""
    if molecule == "lih":
        geometry = [("Li", (0, 0, 0)), ("H", (0, 0, 1.6))]
    elif molecule == "h2":
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
    elif molecule == "h2o":
        geometry = [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.96, 0.0, 0.0)),
            ("H", (-0.24, 0.93, 0.0)),
        ]
    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = "sto-3g"
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)

    integrals = MolecularIntegrals(
        h1e=h1e, h2e=h2e, nuclear_repulsion=mol.energy_nuc(),
        n_electrons=mol.nelectron, n_orbitals=mol.nao,
        n_alpha=mol.nelectron // 2, n_beta=mol.nelectron // 2
    )

    return MolecularHamiltonian(integrals, device="cpu"), mf.e_tot


def generate_basis(H, max_configs=500):
    """Generate particle-conserving basis configurations."""
    from itertools import combinations

    n_orb = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta

    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    configs = []
    for alpha_occ in alpha_configs:
        for beta_occ in beta_configs:
            config = torch.zeros(H.num_sites, dtype=torch.long)
            for i in alpha_occ:
                config[i] = 1
            for i in beta_occ:
                config[i + n_orb] = 1
            configs.append(config)

            if len(configs) >= max_configs:
                break
        if len(configs) >= max_configs:
            break

    return torch.stack(configs)


def validate_jw_consistency(H, basis, verbose=True):
    """
    Check JW sign consistency for all connected pairs in basis.

    For each pair (i,j) where i and j are connected:
    - H[i,j] from get_connections(j) should equal H[j,i] from get_connections(i)

    Returns:
        errors: List of (config_j_idx, H_ij, H_ji, diff) for inconsistent pairs
    """
    print("\n=== JW Sign Consistency Validation ===")

    n_basis = len(basis)
    errors = []
    pairs_checked = 0
    max_diff = 0.0

    # Build hash map for quick lookup
    basis_hash = {tuple(basis[i].cpu().tolist()): i for i in range(n_basis)}

    for j in range(n_basis):
        config_j = basis[j]
        connected_j, elements_j = H.get_connections(config_j)

        if len(connected_j) == 0:
            continue

        for k in range(len(connected_j)):
            config_i = connected_j[k]
            key_i = tuple(config_i.cpu().tolist())

            if key_i not in basis_hash:
                continue  # Connected config not in our basis

            i = basis_hash[key_i]

            if i <= j:  # Only check each pair once
                continue

            # H[i,j] from get_connections(j)
            H_ij = elements_j[k].item()

            # H[j,i] from get_connections(i)
            connected_i, elements_i = H.get_connections(config_i)

            H_ji = None
            for m in range(len(connected_i)):
                if torch.all(connected_i[m] == config_j):
                    H_ji = elements_i[m].item()
                    break

            if H_ji is None:
                print(f"  ERROR: Connection ({i},{j}) exists but ({j},{i}) doesn't!")
                errors.append((j, i, H_ij, None, None))
                continue

            pairs_checked += 1
            diff = abs(H_ij - H_ji)
            max_diff = max(max_diff, diff)

            if diff > 1e-10:
                errors.append((j, i, H_ij, H_ji, diff))
                if verbose and len(errors) <= 10:
                    sign_issue = "OPPOSITE SIGNS" if H_ij * H_ji < 0 else "magnitude diff"
                    print(f"  Pair ({i},{j}): H[i,j]={H_ij:.6f}, H[j,i]={H_ji:.6f}, "
                          f"diff={diff:.2e} [{sign_issue}]")

    print(f"\n  Checked {pairs_checked} connected pairs")
    print(f"  Max difference: {max_diff:.2e}")

    if errors:
        n_sign_errors = sum(1 for e in errors if e[3] is not None and e[2] * e[3] < 0)
        print(f"  FAILED: Found {len(errors)} inconsistent pairs")
        print(f"    - {n_sign_errors} have opposite signs (critical!)")
        print(f"    - {len(errors) - n_sign_errors} have same sign but different magnitude")
        return False, errors
    else:
        print(f"  PASSED: All {pairs_checked} pairs are consistent")
        return True, []


def validate_excitation_types(H, basis, verbose=True):
    """
    Separately validate single and double excitations.

    This helps identify if the bug is in single or double excitation sign computation.
    """
    print("\n=== Excitation Type Analysis ===")

    n_basis = len(basis)
    single_errors = 0
    double_errors = 0
    single_checked = 0
    double_checked = 0

    basis_hash = {tuple(basis[i].cpu().tolist()): i for i in range(n_basis)}

    for j in range(n_basis):
        config_j = basis[j]
        connected_j, elements_j = H.get_connections(config_j)

        if len(connected_j) == 0:
            continue

        for k in range(len(connected_j)):
            config_i = connected_j[k]
            key_i = tuple(config_i.cpu().tolist())

            if key_i not in basis_hash:
                continue

            i = basis_hash[key_i]
            if i <= j:
                continue

            # Count bit differences to determine excitation type
            diff_bits = (config_i != config_j).sum().item()
            is_single = (diff_bits == 2)  # 1 create + 1 annihilate = 2 bits differ
            is_double = (diff_bits == 4)  # 2 create + 2 annihilate = 4 bits differ

            H_ij = elements_j[k].item()

            # Get reverse
            connected_i, elements_i = H.get_connections(config_i)
            H_ji = None
            for m in range(len(connected_i)):
                if torch.all(connected_i[m] == config_j):
                    H_ji = elements_i[m].item()
                    break

            if H_ji is None:
                continue

            diff = abs(H_ij - H_ji)

            if is_single:
                single_checked += 1
                if diff > 1e-10:
                    single_errors += 1
            elif is_double:
                double_checked += 1
                if diff > 1e-10:
                    double_errors += 1

    print(f"  Single excitations: {single_checked} checked, {single_errors} errors")
    print(f"  Double excitations: {double_checked} checked, {double_errors} errors")

    if single_errors > 0:
        print("  -> Single excitation JW sign computation has bugs!")
    if double_errors > 0:
        print("  -> Double excitation JW sign computation has bugs!")

    return single_errors == 0 and double_errors == 0


def validate_matrix_symmetry(H, basis):
    """Check if the full matrix is symmetric after construction."""
    print("\n=== Full Matrix Symmetry Check ===")

    H_matrix = H.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy().astype(np.float64)

    asymmetry = np.abs(H_np - H_np.T).max()
    print(f"  Max asymmetry: {asymmetry:.2e}")

    if asymmetry > 1e-10:
        diff = np.abs(H_np - H_np.T)
        i, j = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Worst pair: ({i},{j})")
        print(f"    H[{i},{j}] = {H_np[i,j]:.8f}")
        print(f"    H[{j},{i}] = {H_np[j,i]:.8f}")

        if H_np[i,j] * H_np[j,i] < 0:
            print(f"    CRITICAL: Opposite signs!")

        return False
    else:
        print(f"  PASSED: Matrix is symmetric")
        return True


def validate_variational_principle(H, basis, reference_energy):
    """Check if computed energy satisfies variational principle."""
    print("\n=== Variational Principle Check ===")

    H_matrix = H.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)  # Symmetrize

    eigenvalues, _ = np.linalg.eigh(H_np)
    E_computed = eigenvalues[0]

    print(f"  Reference energy: {reference_energy:.8f} Ha")
    print(f"  Computed energy:  {E_computed:.8f} Ha")
    print(f"  Difference: {(E_computed - reference_energy)*1000:.4f} mHa")

    if E_computed < reference_energy - 1e-6:
        print(f"  FAILED: Computed energy is BELOW reference by "
              f"{(reference_energy - E_computed)*1000:.4f} mHa!")
        return False
    else:
        print(f"  PASSED: Variational principle satisfied")
        return True


def main():
    print("=" * 60)
    print("Jordan-Wigner Sign Consistency Validation")
    print("=" * 60)

    results = {}

    for molecule in ["h2", "lih"]:
        print(f"\n{'='*60}")
        print(f"Testing {molecule.upper()}")
        print("=" * 60)

        H, hf_energy = create_test_hamiltonian(molecule)
        n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
        print(f"System: {H.num_sites} qubits, {n_valid} valid configs")

        basis = generate_basis(H, max_configs=n_valid)
        print(f"Generated {len(basis)} basis configurations")

        # Test 1: JW sign consistency
        jw_ok, _ = validate_jw_consistency(H, basis, verbose=True)

        # Test 2: Excitation type analysis
        exc_ok = validate_excitation_types(H, basis)

        # Test 3: Matrix symmetry
        sym_ok = validate_matrix_symmetry(H, basis)

        # Test 4: Variational principle
        fci_energy = H.fci_energy()
        var_ok = validate_variational_principle(H, basis, fci_energy)

        results[molecule] = {
            'jw_consistency': jw_ok,
            'excitation_types': exc_ok,
            'matrix_symmetry': sym_ok,
            'variational': var_ok,
        }

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for mol, tests in results.items():
        print(f"\n{mol.upper()}:")
        for test, passed in tests.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {test}: {status}")
            if not passed:
                all_passed = False

    if all_passed:
        print("\nAll validation tests PASSED!")
        return 0
    else:
        print("\nSome validation tests FAILED - JW sign computation needs fixing!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
