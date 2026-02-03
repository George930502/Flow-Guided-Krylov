"""
Benchmark script to test GPU optimization effectiveness.

Compares performance of:
1. Vectorized batch connections vs sequential connections
2. GPU-based matrix construction
3. Integer encoding vs hash-based deduplication
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import time
from math import comb
from itertools import combinations

try:
    from pyscf import gto, scf, ao2mo
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("PySCF not available")
    sys.exit(1)

from hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals


def create_hamiltonian(molecule="lih", device="cuda"):
    """Create a test Hamiltonian."""
    if molecule == "h2":
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, 0.74))]
    elif molecule == "lih":
        geometry = [("Li", (0, 0, 0)), ("H", (0, 0, 1.6))]
    elif molecule == "h2o":
        geometry = [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.96, 0.0, 0.0)),
            ("H", (-0.24, 0.93, 0.0)),
        ]
    elif molecule == "beh2":
        geometry = [
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.3)),
            ("H", (0.0, 0.0, -1.3)),
        ]
    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)

    integrals = MolecularIntegrals(
        h1e=h1e, h2e=h2e, nuclear_repulsion=mol.energy_nuc(),
        n_electrons=mol.nelectron, n_orbitals=mol.nao,
        n_alpha=mol.nelectron // 2, n_beta=mol.nelectron // 2
    )

    return MolecularHamiltonian(integrals, device=device), mol.nao, mol.nelectron


def generate_basis(H, max_configs=None):
    """Generate particle-conserving basis configurations."""
    n_orb = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta

    alpha_configs = list(combinations(range(n_orb), n_alpha))
    beta_configs = list(combinations(range(n_orb), n_beta))

    n_valid = len(alpha_configs) * len(beta_configs)
    if max_configs is None:
        max_configs = n_valid

    configs = []
    for alpha_occ in alpha_configs:
        for beta_occ in beta_configs:
            config = torch.zeros(H.num_sites, dtype=torch.long, device=H.device)
            for i in alpha_occ:
                config[i] = 1
            for i in beta_occ:
                config[i + n_orb] = 1
            configs.append(config)

            if len(configs) >= max_configs:
                break
        if len(configs) >= max_configs:
            break

    return torch.stack(configs), n_valid


def benchmark_get_connections_sequential(H, basis, n_samples=None):
    """Benchmark sequential get_connections calls."""
    if n_samples is None:
        n_samples = min(100, len(basis))

    # Warm up
    for i in range(min(5, n_samples)):
        H.get_connections(basis[i])

    if H.device == "cuda" or (hasattr(H.device, 'type') and H.device.type == 'cuda'):
        torch.cuda.synchronize()

    start = time.perf_counter()
    total_connections = 0
    for i in range(n_samples):
        connected, elements = H.get_connections(basis[i])
        total_connections += len(connected)

    if H.device == "cuda" or (hasattr(H.device, 'type') and H.device.type == 'cuda'):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed, total_connections, n_samples


def benchmark_get_connections_vectorized(H, basis, n_samples=None):
    """Benchmark vectorized batch get_connections."""
    if not hasattr(H, 'get_connections_vectorized_batch'):
        return None, 0, 0

    if n_samples is None:
        n_samples = min(100, len(basis))

    sample_basis = basis[:n_samples]

    # Warm up
    H.get_connections_vectorized_batch(sample_basis[:min(5, n_samples)])

    if H.device == "cuda" or (hasattr(H.device, 'type') and H.device.type == 'cuda'):
        torch.cuda.synchronize()

    start = time.perf_counter()
    all_connected, all_elements, batch_indices = H.get_connections_vectorized_batch(sample_basis)

    if H.device == "cuda" or (hasattr(H.device, 'type') and H.device.type == 'cuda'):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed, len(all_connected), n_samples


def benchmark_matrix_construction(H, basis):
    """Benchmark full matrix construction."""
    # Warm up
    if len(basis) > 10:
        H.matrix_elements_fast(basis[:10])

    if H.device == "cuda" or (hasattr(H.device, 'type') and H.device.type == 'cuda'):
        torch.cuda.synchronize()

    start = time.perf_counter()
    H_matrix = H.matrix_elements_fast(basis)

    if H.device == "cuda" or (hasattr(H.device, 'type') and H.device.type == 'cuda'):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Verify correctness
    H_np = H_matrix.cpu().numpy()
    eigenvalues, _ = np.linalg.eigh(H_np)
    ground_energy = eigenvalues[0]

    return elapsed, ground_energy, H_matrix.shape[0]


def benchmark_integer_encoding(H, basis):
    """Benchmark integer encoding vs hash-based encoding."""
    n_configs = len(basis)
    num_sites = basis.shape[1]
    device = basis.device

    # Method 1: GPU integer encoding (new)
    if hasattr(H, '_powers_gpu'):
        powers = H._powers_gpu
    else:
        powers = (2 ** torch.arange(num_sites, device=device, dtype=torch.long)).flip(0)

    if device == "cuda" or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.synchronize()

    start = time.perf_counter()
    config_ints = (basis.long() * powers).sum(dim=1)
    int_set = set(config_ints.cpu().tolist())

    if device == "cuda" or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.synchronize()

    time_int = time.perf_counter() - start

    # Method 2: Hash-based encoding (old)
    start = time.perf_counter()
    basis_np = basis.cpu().numpy()
    hash_set = {hash(c.tobytes()) for c in basis_np}
    time_hash = time.perf_counter() - start

    return time_int, time_hash, n_configs


def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    print("=" * 70)
    print("GPU Optimization Benchmark")
    print("=" * 70)

    molecules = ["h2", "lih", "h2o", "beh2"]

    for mol_name in molecules:
        print(f"\n{'='*70}")
        print(f"Molecule: {mol_name.upper()}")
        print("=" * 70)

        try:
            H, n_orb, n_elec = create_hamiltonian(mol_name, device=device)
        except Exception as e:
            print(f"  Error creating Hamiltonian: {e}")
            continue

        n_valid = comb(n_orb, n_elec // 2) ** 2
        print(f"  Qubits: {H.num_sites}, Valid configs: {n_valid:,}")

        # Generate basis
        max_basis = min(500, n_valid)
        basis, _ = generate_basis(H, max_configs=max_basis)
        print(f"  Test basis size: {len(basis)}")

        # Benchmark 1: Sequential vs Vectorized get_connections
        print(f"\n  --- Connection Computation ---")
        n_samples = min(50, len(basis))

        time_seq, conn_seq, _ = benchmark_get_connections_sequential(H, basis, n_samples)
        print(f"  Sequential ({n_samples} configs): {time_seq*1000:.2f} ms, {conn_seq:,} connections")

        time_vec, conn_vec, _ = benchmark_get_connections_vectorized(H, basis, n_samples)
        if time_vec is not None:
            speedup = time_seq / time_vec if time_vec > 0 else float('inf')
            print(f"  Vectorized ({n_samples} configs): {time_vec*1000:.2f} ms, {conn_vec:,} connections")
            print(f"  Speedup: {speedup:.1f}x")
        else:
            print(f"  Vectorized: Not available")

        # Benchmark 2: Matrix construction
        print(f"\n  --- Matrix Construction ---")
        time_matrix, energy, matrix_size = benchmark_matrix_construction(H, basis)
        print(f"  Time ({matrix_size}x{matrix_size}): {time_matrix*1000:.2f} ms")
        print(f"  Ground energy: {energy:.8f} Ha")

        # Verify against FCI if small enough
        if n_valid <= 1000:
            fci_energy = H.fci_energy()
            error = abs(energy - fci_energy) * 1000
            print(f"  FCI energy: {fci_energy:.8f} Ha")
            print(f"  Error: {error:.4f} mHa")

        # Benchmark 3: Integer encoding vs hash
        print(f"\n  --- Config Encoding ---")
        time_int, time_hash, n_enc = benchmark_integer_encoding(H, basis)
        speedup_enc = time_hash / time_int if time_int > 0 else float('inf')
        print(f"  Integer encoding ({n_enc} configs): {time_int*1000:.2f} ms")
        print(f"  Hash encoding ({n_enc} configs): {time_hash*1000:.2f} ms")
        print(f"  Encoding speedup: {speedup_enc:.1f}x")

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
