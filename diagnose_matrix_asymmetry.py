"""
Diagnostic script to identify matrix element asymmetry issues.

This script tests whether H[i,j] == H[j,i] for connected configurations,
which is required for Hermiticity. If they differ, the symmetrization
H = 0.5 * (H + H.T) will corrupt the matrix.
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


def check_h2e_symmetry(H):
    """Check if h2e tensor has proper 8-fold symmetry."""
    print("\n=== Checking h2e Symmetry ===")
    h2e = H._h2e_np
    n = h2e.shape[0]

    max_asymmetry = 0
    asymmetric_count = 0

    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    # Check (pq|rs) = (qp|sr) symmetry
                    val1 = h2e[p, q, r, s]
                    val2 = h2e[q, p, s, r]
                    diff = abs(val1 - val2)
                    if diff > 1e-10:
                        asymmetric_count += 1
                        max_asymmetry = max(max_asymmetry, diff)

    if asymmetric_count > 0:
        print(f"  WARNING: Found {asymmetric_count} asymmetric h2e elements")
        print(f"  Max asymmetry: {max_asymmetry:.2e}")
    else:
        print(f"  h2e tensor has proper symmetry (checked all {n**4} elements)")

    return asymmetric_count == 0


def check_matrix_element_symmetry(H, basis, sample_size=100):
    """
    Check if H[i,j] == H[j,i] for connected configuration pairs.

    This is the critical test for Hermiticity.
    """
    print("\n=== Checking Matrix Element Symmetry ===")

    n_basis = len(basis)
    asymmetric_pairs = []
    max_asymmetry = 0
    pairs_checked = 0

    # Build hash map for quick lookup
    basis_hash = {tuple(basis[i].cpu().tolist()): i for i in range(n_basis)}

    for j in range(n_basis):
        config_j = basis[j]
        connected, elements = H.get_connections(config_j)

        if len(connected) == 0:
            continue

        for k in range(len(connected)):
            config_i = connected[k]
            key_i = tuple(config_i.cpu().tolist())

            if key_i not in basis_hash:
                continue

            i = basis_hash[key_i]

            if i <= j:  # Only check each pair once
                continue

            # H[i,j] from get_connections(j)
            H_ij = elements[k].item()

            # H[j,i] from get_connections(i)
            connected_from_i, elements_from_i = H.get_connections(config_i)

            H_ji = None
            for m in range(len(connected_from_i)):
                if torch.all(connected_from_i[m] == config_j):
                    H_ji = elements_from_i[m].item()
                    break

            if H_ji is None:
                print(f"  ERROR: Asymmetric connection - ({i},{j}) exists but ({j},{i}) doesn't!")
                asymmetric_pairs.append((i, j, H_ij, None))
                continue

            pairs_checked += 1
            diff = abs(H_ij - H_ji)

            if diff > 1e-10:
                asymmetric_pairs.append((i, j, H_ij, H_ji))
                max_asymmetry = max(max_asymmetry, diff)

            if pairs_checked >= sample_size:
                break

        if pairs_checked >= sample_size:
            break

    print(f"  Checked {pairs_checked} connected pairs")

    if asymmetric_pairs:
        print(f"  WARNING: Found {len(asymmetric_pairs)} asymmetric pairs!")
        print(f"  Max asymmetry: {max_asymmetry:.2e}")
        print("\n  Sample asymmetric pairs:")
        for idx, (i, j, H_ij, H_ji) in enumerate(asymmetric_pairs[:5]):
            if H_ji is not None:
                print(f"    ({i},{j}): H[i,j]={H_ij:.6f}, H[j,i]={H_ji:.6f}, diff={H_ij-H_ji:.6e}")
            else:
                print(f"    ({i},{j}): H[i,j]={H_ij:.6f}, H[j,i]=NOT FOUND")
        return False
    else:
        print(f"  All {pairs_checked} pairs are symmetric (max diff < 1e-10)")
        return True


def check_matrix_symmetry_full(H, basis):
    """Check full matrix symmetry before and after symmetrization."""
    print("\n=== Checking Full Matrix Symmetry ===")

    H_matrix = H.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy().astype(np.float64)

    asymmetry_before = np.abs(H_np - H_np.T).max()
    print(f"  Max asymmetry before symmetrization: {asymmetry_before:.2e}")

    # Check where the asymmetry is
    if asymmetry_before > 1e-10:
        diff_matrix = np.abs(H_np - H_np.T)
        i, j = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
        print(f"  Worst pair: ({i},{j})")
        print(f"    H[{i},{j}] = {H_np[i,j]:.8f}")
        print(f"    H[{j},{i}] = {H_np[j,i]:.8f}")
        print(f"    Difference: {H_np[i,j] - H_np[j,i]:.2e}")

        # Check sign
        if H_np[i,j] * H_np[j,i] < 0:
            print(f"    CRITICAL: Opposite signs! Symmetrization will destroy this element!")

    # Symmetrize and compare eigenvalues
    H_sym = 0.5 * (H_np + H_np.T)

    eig_before, _ = np.linalg.eigh(H_np)
    eig_after, _ = np.linalg.eigh(H_sym)

    print(f"\n  Ground state energy before symmetrization: {eig_before[0]:.8f}")
    print(f"  Ground state energy after symmetrization:  {eig_after[0]:.8f}")
    print(f"  Energy change from symmetrization: {(eig_after[0] - eig_before[0])*1000:.4f} mHa")

    return asymmetry_before < 1e-10


def check_variational_principle(H, basis, reference_energy):
    """Check if computed energy satisfies variational principle."""
    print("\n=== Checking Variational Principle ===")

    H_matrix = H.matrix_elements(basis, basis)
    H_np = H_matrix.cpu().numpy().astype(np.float64)
    H_np = 0.5 * (H_np + H_np.T)

    eigenvalues, _ = np.linalg.eigh(H_np)
    E_computed = eigenvalues[0]

    print(f"  Reference energy: {reference_energy:.8f} Ha")
    print(f"  Computed energy:  {E_computed:.8f} Ha")
    print(f"  Difference: {(E_computed - reference_energy)*1000:.4f} mHa")

    if E_computed < reference_energy - 1e-6:
        print(f"  WARNING: VARIATIONAL PRINCIPLE VIOLATED!")
        print(f"  Computed energy is {(reference_energy - E_computed)*1000:.4f} mHa BELOW reference!")
        return False
    else:
        print(f"  Variational principle satisfied (E_computed >= E_reference)")
        return True


def generate_particle_conserving_basis(H, max_configs=1000):
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


def main():
    print("="*60)
    print("Matrix Element Asymmetry Diagnostic")
    print("="*60)

    # Test with LiH
    print("\n### Testing LiH ###")
    H, hf_energy = create_test_hamiltonian("lih")

    n_valid = comb(H.n_orbitals, H.n_alpha) * comb(H.n_orbitals, H.n_beta)
    print(f"System: LiH, {H.num_sites} qubits, {n_valid} valid configs")

    # Check h2e symmetry
    h2e_ok = check_h2e_symmetry(H)

    # Generate full basis
    basis = generate_particle_conserving_basis(H, max_configs=n_valid)
    print(f"\nGenerated {len(basis)} basis configurations")

    # Check matrix element symmetry
    elem_ok = check_matrix_element_symmetry(H, basis, sample_size=500)

    # Check full matrix symmetry
    matrix_ok = check_matrix_symmetry_full(H, basis)

    # Check variational principle against HF
    # (For LiH with full basis, we should get FCI energy which is below HF)
    fci_energy = H.fci_energy()
    var_ok = check_variational_principle(H, basis, fci_energy)

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"  h2e symmetry:           {'PASS' if h2e_ok else 'FAIL'}")
    print(f"  Matrix element symmetry: {'PASS' if elem_ok else 'FAIL'}")
    print(f"  Full matrix symmetry:    {'PASS' if matrix_ok else 'FAIL'}")
    print(f"  Variational principle:   {'PASS' if var_ok else 'FAIL'}")

    if all([h2e_ok, elem_ok, matrix_ok, var_ok]):
        print("\nAll tests PASSED for LiH")
    else:
        print("\nSome tests FAILED - investigate the issues above")

    return all([h2e_ok, elem_ok, matrix_ok, var_ok])


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
