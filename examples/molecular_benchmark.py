#!/usr/bin/env python3
"""
Molecular Systems Benchmark: H2, LiH, H2O

Benchmarks the GPU-optimized implementation for:
1. Hamiltonian construction time
2. Training epoch time
3. Total pipeline time
4. Energy accuracy

Run with:
    python examples/molecular_benchmark.py [--molecule h2|lih|h2o|all] [--epochs N]
"""

import sys
from pathlib import Path
import argparse
import time
import gc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

# Check imports
try:
    from hamiltonians.molecular import (
        create_h2_hamiltonian,
        create_lih_hamiltonian,
        create_h2o_hamiltonian,
        MolecularHamiltonian,
    )
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from flows.discrete_flow import DiscreteFlowSampler
from nqs.dense import DenseNQS
from flows.training import FlowNQSTrainer, TrainingConfig


def print_header(title: str, width: int = 70):
    """Print formatted header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_section(title: str, width: int = 70):
    """Print section divider."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def benchmark_hamiltonian_construction(
    hamiltonian,
    n_configs: int = 500,
    n_trials: int = 3,
) -> dict:
    """
    Benchmark Hamiltonian matrix construction.

    Returns timing statistics.
    """
    device = hamiltonian.device

    # Generate random valid configurations
    configs = torch.randint(0, 2, (n_configs, hamiltonian.num_sites),
                           dtype=torch.long, device=device)

    # Warm-up
    _ = hamiltonian.matrix_elements(configs[:10], configs[:10])
    if str(device) == "cuda":
        torch.cuda.synchronize()

    # Benchmark diagonal computation (batch)
    diag_times = []
    for _ in range(n_trials):
        if str(device) == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        if hasattr(hamiltonian, 'diagonal_elements_batch'):
            _ = hamiltonian.diagonal_elements_batch(configs)
        else:
            for i in range(len(configs)):
                _ = hamiltonian.diagonal_element(configs[i])

        if str(device) == "cuda":
            torch.cuda.synchronize()
        diag_times.append(time.perf_counter() - start)

    # Benchmark full matrix construction
    matrix_times = []
    for _ in range(n_trials):
        if str(device) == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        if hasattr(hamiltonian, 'matrix_elements_fast'):
            _ = hamiltonian.matrix_elements_fast(configs)
        else:
            _ = hamiltonian.matrix_elements(configs, configs)

        if str(device) == "cuda":
            torch.cuda.synchronize()
        matrix_times.append(time.perf_counter() - start)

    return {
        "diag_mean": np.mean(diag_times),
        "diag_std": np.std(diag_times),
        "matrix_mean": np.mean(matrix_times),
        "matrix_std": np.std(matrix_times),
        "n_configs": n_configs,
    }


def benchmark_training(
    hamiltonian,
    num_epochs: int = 10,
) -> dict:
    """
    Benchmark training epoch time.

    Returns timing and energy statistics.
    """
    device = hamiltonian.device
    n_sites = hamiltonian.num_sites

    # Create models
    flow = DiscreteFlowSampler(
        num_sites=n_sites,
        num_coupling_layers=3,
        hidden_dims=[128, 128],
    )

    nqs = DenseNQS(
        num_sites=n_sites,
        hidden_dims=[128, 128, 128],
    )

    # Create config with GPU optimizations enabled
    config = TrainingConfig(
        samples_per_batch=1000,
        num_epochs=num_epochs,
        min_epochs=1,
        convergence_threshold=0.01,
        cache_hamiltonian=True,  # Enable GPU caching
    )

    # Create trainer
    trainer = FlowNQSTrainer(
        flow=flow,
        nqs=nqs,
        hamiltonian=hamiltonian,
        config=config,
        device=str(device),
    )

    # Train and time
    start = time.perf_counter()
    history = trainer.train(num_epochs=num_epochs)
    total_time = time.perf_counter() - start

    # Get epoch times if available
    if "epoch_times" in history:
        epoch_times = history["epoch_times"]
    else:
        epoch_times = [total_time / num_epochs] * num_epochs

    return {
        "total_time": total_time,
        "epoch_mean": np.mean(epoch_times),
        "epoch_std": np.std(epoch_times),
        "final_energy": history["energies"][-1],
        "final_ema": history["ema_energies"][-1],
        "num_epochs": num_epochs,
    }


def run_molecule_benchmark(
    molecule: str,
    training_epochs: int = 50,
    ham_configs: int = 300,
) -> dict:
    """
    Run complete benchmark for a molecule.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"molecule": molecule, "device": device}

    # Create Hamiltonian (now with GPU support built-in)
    print(f"Creating {molecule.upper()} Hamiltonian on {device}...")
    if molecule == "h2":
        hamiltonian = create_h2_hamiltonian(device=device)
        bond_info = "bond=0.74 Ang"
    elif molecule == "lih":
        hamiltonian = create_lih_hamiltonian(device=device)
        bond_info = "bond=1.6 Ang"
    elif molecule == "h2o":
        hamiltonian = create_h2o_hamiltonian(device=device)
        bond_info = "O-H=0.96 Ang, angle=104.5 deg"
    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    results["n_qubits"] = hamiltonian.num_sites
    results["hilbert_dim"] = hamiltonian.hilbert_dim
    results["bond_info"] = bond_info

    # Get exact energy
    print(f"Computing exact ground state energy...")
    E_exact, _ = hamiltonian.exact_ground_state()
    results["exact_energy"] = E_exact
    print(f"Exact energy: {E_exact:.8f} Ha")

    # Benchmark Hamiltonian construction
    print_section("Hamiltonian Construction Benchmark")
    print(f"Running Hamiltonian benchmark ({ham_configs} configurations)...")

    ham_bench = benchmark_hamiltonian_construction(
        hamiltonian, n_configs=ham_configs
    )
    results["ham_bench"] = ham_bench
    print(f"  Diagonal ({ham_configs} configs): {ham_bench['diag_mean']*1000:.2f} +/- {ham_bench['diag_std']*1000:.2f} ms")
    print(f"  Full matrix: {ham_bench['matrix_mean']:.3f} +/- {ham_bench['matrix_std']:.3f} s")

    # Benchmark training
    print_section("Training Benchmark")
    print(f"Running training ({training_epochs} epochs)...")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    train_bench = benchmark_training(
        hamiltonian,
        num_epochs=training_epochs,
    )
    results["train_bench"] = train_bench
    print(f"  Total time: {train_bench['total_time']:.2f} s")
    print(f"  Epoch time: {train_bench['epoch_mean']:.3f} +/- {train_bench['epoch_std']:.3f} s")
    print(f"  Final energy: {train_bench['final_energy']:.6f} Ha")
    print(f"  Final EMA energy: {train_bench['final_ema']:.6f} Ha")

    # Energy error
    energy_error = abs(train_bench['final_ema'] - E_exact)
    results["energy_error"] = energy_error
    print(f"  Energy error: {energy_error:.6f} Ha ({energy_error*1000:.3f} mHa)")

    return results


def print_summary_table(all_results: list):
    """Print summary comparison table."""
    print_header("BENCHMARK SUMMARY")

    # Header
    print(f"{'Molecule':<10} {'Qubits':<8} {'Diag (ms)':<12} {'Matrix (s)':<12} {'Energy (Ha)':<14} {'Error (mHa)':<12}")
    print("-" * 70)

    for r in all_results:
        ham = r.get('ham_bench', {})
        diag_ms = ham.get('diag_mean', 0) * 1000
        matrix_s = ham.get('matrix_mean', 0)
        energy_error_mha = r.get('energy_error', 0) * 1000

        print(f"{r['molecule']:<10} {r['n_qubits']:<8} {diag_ms:<12.2f} {matrix_s:<12.3f} {r['exact_energy']:<14.6f} {energy_error_mha:<12.3f}")

    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Molecular Systems Benchmark")
    parser.add_argument(
        "--molecule",
        type=str,
        default="all",
        choices=["h2", "lih", "h2o", "all"],
        help="Which molecule to benchmark"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for benchmark"
    )
    parser.add_argument(
        "--ham-configs",
        type=int,
        default=200,
        help="Number of configs for Hamiltonian benchmark"
    )

    args = parser.parse_args()

    # Check dependencies
    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF required. Install with: pip install pyscf")
        sys.exit(1)

    # Print system info
    print_header("MOLECULAR SYSTEMS BENCHMARK")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Run benchmarks
    molecules = ["h2", "lih", "h2o"] if args.molecule == "all" else [args.molecule]
    all_results = []

    for mol in molecules:
        print_header(f"BENCHMARKING: {mol.upper()}")
        try:
            result = run_molecule_benchmark(
                mol,
                training_epochs=args.epochs,
                ham_configs=args.ham_configs,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR benchmarking {mol}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if len(all_results) > 0:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
