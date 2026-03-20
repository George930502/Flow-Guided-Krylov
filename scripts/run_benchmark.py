#!/usr/bin/env python3
"""
Unified benchmark script for all 16 molecules across 9 methods.

Automatically determines method feasibility per molecule and adjusts
hyperparameters based on system size (tier).

Usage:
    python scripts/run_benchmark.py                    # all molecules
    python scripts/run_benchmark.py --tier 1           # only Tier 1 (≤14Q)
    python scripts/run_benchmark.py --tier 3           # Tier 1-3
    python scripts/run_benchmark.py --molecules H2,LiH # specific molecules
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver, MAX_FCI_CONFIGS
from src.solvers.ccsd import CCSDSolver, CCSDTSolver
from src.solvers.sci import CIPSISolver
from src.solvers.sqd import SQDSolver, SQDConfig
from src.solvers.skqd import SKQDSolverB, SKQDSolverC, SKQDConfig
from src.solvers.dci_skqd import DCISKQDSolverB, DCISKQDSolverC, DCISKQDConfig
from src.solvers.iterative_nf_sqd import IterativeNFSQDSolver, IterativeNFSKQDSolver, IterativeNFSQDConfig
from src.samplers.nf_sampler import NFSampler, NFSamplerConfig
from src.samplers.transformer_nf_sampler import TransformerNFSampler, TransformerSamplerConfig

# =============================================================================
# Experiment configuration per molecule
# =============================================================================

EXPERIMENT_CONFIG = {
    # --- Tier 1: ≤14Q --- full methods, fast
    "H2":   {"tier": 1, "nf_epochs": 100,  "nf_samples_per_epoch": 512,  "nf_n_samples": 1000,  "skqd_max_basis": 5000,  "sci_max_basis": 5000},
    "LiH":  {"tier": 1, "nf_epochs": 200,  "nf_samples_per_epoch": 512,  "nf_n_samples": 2000,  "skqd_max_basis": 5000,  "sci_max_basis": 5000},
    "H2O":  {"tier": 1, "nf_epochs": 200,  "nf_samples_per_epoch": 512,  "nf_n_samples": 2000,  "skqd_max_basis": 5000,  "sci_max_basis": 5000},
    "BeH2": {"tier": 1, "nf_epochs": 200,  "nf_samples_per_epoch": 512,  "nf_n_samples": 3000,  "skqd_max_basis": 5000,  "sci_max_basis": 5000},

    # --- Tier 2: 16-18Q --- quantum methods slow but feasible
    "NH3":  {"tier": 2, "nf_epochs": 300,  "nf_samples_per_epoch": 1024, "nf_n_samples": 5000,  "skqd_max_basis": 8000,  "sci_max_basis": 8000},
    "CH4":  {"tier": 2, "nf_epochs": 400,  "nf_samples_per_epoch": 1024, "nf_n_samples": 5000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},

    # --- Tier 3: 20Q --- classical + NF only
    "N2":   {"tier": 3, "nf_epochs": 400,  "nf_samples_per_epoch": 1024, "nf_n_samples": 5000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "CO":   {"tier": 3, "nf_epochs": 400,  "nf_samples_per_epoch": 1024, "nf_n_samples": 5000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},

    # --- Tier 4: 22-24Q ---
    "HCN":           {"tier": 4, "nf_epochs": 500,  "nf_samples_per_epoch": 2048, "nf_n_samples": 8000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "C2H2":          {"tier": 4, "nf_epochs": 500,  "nf_samples_per_epoch": 2048, "nf_n_samples": 8000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "N2-CAS(10,12)": {"tier": 4, "nf_epochs": 500,  "nf_samples_per_epoch": 2048, "nf_n_samples": 8000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "Cr2":           {"tier": 4, "nf_epochs": 500,  "nf_samples_per_epoch": 2048, "nf_n_samples": 8000,  "skqd_max_basis": 10000, "sci_max_basis": 10000},

    # --- Tier 5: 26-30Q --- large hyperparameters
    "H2S":           {"tier": 5, "nf_epochs": 600,  "nf_samples_per_epoch": 2048, "nf_n_samples": 10000, "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "C2H4":          {"tier": 5, "nf_epochs": 600,  "nf_samples_per_epoch": 2048, "nf_n_samples": 10000, "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "Benzene":       {"tier": 5, "nf_epochs": 600,  "nf_samples_per_epoch": 2048, "nf_n_samples": 10000, "skqd_max_basis": 10000, "sci_max_basis": 10000},
    "N2-CAS(10,15)": {"tier": 5, "nf_epochs": 600,  "nf_samples_per_epoch": 2048, "nf_n_samples": 10000, "skqd_max_basis": 10000, "sci_max_basis": 10000},

    # --- Tier 6: 34-40Q --- large CAS
    "N2-CAS(10,17)":  {"tier": 6, "nf_epochs": 700,  "nf_samples_per_epoch": 2048, "nf_n_samples": 15000, "skqd_max_basis": 12000, "sci_max_basis": 12000},
    "Cr2-CAS(12,18)": {"tier": 6, "nf_epochs": 700,  "nf_samples_per_epoch": 2048, "nf_n_samples": 15000, "skqd_max_basis": 12000, "sci_max_basis": 12000},
    "N2-CAS(10,20)":  {"tier": 6, "nf_epochs": 700,  "nf_samples_per_epoch": 4096, "nf_n_samples": 15000, "skqd_max_basis": 12000, "sci_max_basis": 12000},
    "Cr2-CAS(12,20)": {"tier": 6, "nf_epochs": 700,  "nf_samples_per_epoch": 4096, "nf_n_samples": 15000, "skqd_max_basis": 12000, "sci_max_basis": 12000},

    # --- Tier 7: 52-58Q --- ultra-large CAS, only SCI + DCI-SKQD feasible
    "N2-CAS(10,26)":  {"tier": 7, "nf_epochs": 800,  "nf_samples_per_epoch": 4096, "nf_n_samples": 20000, "skqd_max_basis": 15000, "sci_max_basis": 15000},
    "Cr2-CAS(12,26)": {"tier": 7, "nf_epochs": 800,  "nf_samples_per_epoch": 4096, "nf_n_samples": 20000, "skqd_max_basis": 15000, "sci_max_basis": 15000},
    "Cr2-CAS(12,28)": {"tier": 7, "nf_epochs": 800,  "nf_samples_per_epoch": 4096, "nf_n_samples": 20000, "skqd_max_basis": 15000, "sci_max_basis": 15000},
    "Cr2-CAS(12,29)": {"tier": 7, "nf_epochs": 800,  "nf_samples_per_epoch": 4096, "nf_n_samples": 20000, "skqd_max_basis": 15000, "sci_max_basis": 15000},
}

# Ordered by n_qubits (small to large)
ALL_MOLECULES = [
    "H2", "LiH", "H2O", "BeH2",
    "NH3", "CH4",
    "N2", "CO",
    "HCN", "C2H2", "N2-CAS(10,12)", "Cr2",
    "H2S", "C2H4", "Benzene", "N2-CAS(10,15)",
    "N2-CAS(10,17)", "Cr2-CAS(12,18)", "N2-CAS(10,20)", "Cr2-CAS(12,20)",
    "N2-CAS(10,26)", "Cr2-CAS(12,26)", "Cr2-CAS(12,28)", "Cr2-CAS(12,29)",
]


def get_methods_for_molecule(mol_name, mol_info):
    """Determine which methods are feasible for a given molecule.

    Returns a list of method name strings.
    """
    cfg = EXPERIMENT_CONFIG[mol_name]
    tier = cfg["tier"]
    is_cas = mol_info["is_cas"]
    n_orb = mol_info["n_qubits"] // 2

    methods = []

    # FCI: check determinant space size
    # We need n_alpha, n_beta — estimate from n_qubits or compute
    # For now, always attempt FCI and let the solver decide via MAX_FCI_CONFIGS
    methods.append("FCI")

    # CCSD / CCSD(T): not valid for CAS systems
    if not is_cas:
        methods.append("CCSD")
        methods.append("CCSD(T)")

    # SCI: always feasible
    methods.append("SCI")

    # NF methods: always feasible
    methods.append("NF-SQD")
    methods.append("NF-SKQD-B")
    methods.append("NF-SKQD-C")

    # DCI-SKQD methods: always feasible (no sampler needed)
    methods.append("DCI-SKQD-B")
    methods.append("DCI-SKQD-C")

    # Transformer NF methods: always feasible
    methods.append("TF-SQD")
    methods.append("TF-SKQD-B")
    methods.append("TF-SKQD-C")

    # Iterative NF-SQD methods: always feasible
    methods.append("Iter-NF-SQD")
    methods.append("Iter-NF-SKQD")

    # Quantum methods: only for small systems (tier <= 2) and if deps available
    if tier <= 2:
        try:
            import ffsim  # noqa: F401
            methods.append("Q-SQD")
            methods.append("Q-SKQD")
        except ImportError:
            pass

    return methods


def _run_baselines(mol_name, H, mol_info, methods, fci_energies):
    """Run baseline methods (FCI, CCSD, CCSD(T), SCI) and return results."""
    cfg = EXPERIMENT_CONFIG[mol_name]
    results = []

    baseline_solvers = {}
    if "FCI" in methods:
        baseline_solvers["FCI"] = FCISolver()
    if "CCSD" in methods:
        baseline_solvers["CCSD"] = CCSDSolver()
    if "CCSD(T)" in methods:
        baseline_solvers["CCSD(T)"] = CCSDTSolver()
    if "SCI" in methods:
        baseline_solvers["SCI"] = CIPSISolver(max_basis_size=cfg["sci_max_basis"])

    for solver_name, solver in baseline_solvers.items():
        try:
            result = solver.solve(H, mol_info)

            if solver_name == "FCI" and result.energy is not None:
                fci_energies[mol_name] = result.energy

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else ("SKIP" if result.energy is None else "FAIL")

            print(f"  {solver_name:<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name,
                "method": solver_name,
                "energy": result.energy,
                "error_mha": error_mha,
                "diag_dim": result.diag_dim,
                "wall_time": result.wall_time,
                "converged": result.converged,
                "metadata": result.metadata,
            })

        except Exception as e:
            print(f"  {solver_name:<14} ERROR: {e}")
            results.append({
                "molecule": mol_name,
                "method": solver_name,
                "energy": None,
                "error_mha": None,
                "diag_dim": 0,
                "wall_time": 0,
                "converged": False,
                "metadata": {"error": str(e)},
            })

    return results


def _run_nf_methods(mol_name, H, mol_info, methods, fci_energies):
    """Train NF sampler and run NF-SQD, NF-SKQD-B, NF-SKQD-C."""
    nf_methods = [m for m in methods if m.startswith("NF-")]
    if not nf_methods:
        return []

    cfg = EXPERIMENT_CONFIG[mol_name]
    results = []

    # Train NF sampler (time it separately)
    print(f"  Training NF ({cfg['nf_epochs']} epochs, {cfg['nf_samples_per_epoch']} samples/epoch)...")
    import time as _time
    train_t0 = _time.time()
    nf_config = NFSamplerConfig(
        n_epochs=cfg["nf_epochs"],
        samples_per_epoch=cfg["nf_samples_per_epoch"],
    )
    nf_sampler = NFSampler(H, config=nf_config, device="cpu")
    nf_sampler.train(verbose=True)
    nf_train_time = _time.time() - train_t0
    print(f"  NF training completed in {nf_train_time:.2f}s")

    # Build solvers
    n_samples = cfg["nf_n_samples"]
    skqd_max_basis = cfg["skqd_max_basis"]

    solvers = {}
    if "NF-SQD" in nf_methods:
        solvers["NF-SQD"] = SQDSolver(nf_sampler, SQDConfig(n_samples=n_samples))
    if "NF-SKQD-B" in nf_methods:
        solvers["NF-SKQD-B"] = SKQDSolverB(nf_sampler, SKQDConfig(
            n_samples=n_samples, max_basis_size=skqd_max_basis,
        ))
    if "NF-SKQD-C" in nf_methods:
        solvers["NF-SKQD-C"] = SKQDSolverC(nf_sampler, SKQDConfig(
            n_samples=n_samples, max_basis_size=skqd_max_basis,
        ))

    for method_name, solver in solvers.items():
        try:
            result = solver.solve(H, mol_info)

            # Total time = NF training + solve time
            total_time = nf_train_time + result.wall_time

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"  {method_name:<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {total_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": result.energy,
                "error_mha": error_mha,
                "diag_dim": result.diag_dim,
                "wall_time": total_time,
                "converged": result.converged,
                "metadata": {
                    **result.metadata,
                    "nf_train_time": nf_train_time,
                    "solve_time": result.wall_time,
                },
            })

        except Exception as e:
            print(f"  {method_name:<14} ERROR: {e}")
            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": None,
                "error_mha": None,
                "diag_dim": 0,
                "wall_time": 0,
                "converged": False,
                "metadata": {"error": str(e)},
            })

    return results


def _run_tf_methods(mol_name, H, mol_info, methods, fci_energies):
    """Train Transformer NF sampler and run TF-SQD, TF-SKQD-B, TF-SKQD-C."""
    tf_methods = [m for m in methods if m.startswith("TF-")]
    if not tf_methods:
        return []

    cfg = EXPERIMENT_CONFIG[mol_name]
    results = []

    # Train Transformer NF sampler
    print(f"  Training Transformer NF ({cfg['nf_epochs']} epochs, {cfg['nf_samples_per_epoch']} samples/epoch)...")
    import time as _time
    train_t0 = _time.time()
    tf_config = TransformerSamplerConfig(
        n_epochs=cfg["nf_epochs"],
        samples_per_epoch=cfg["nf_samples_per_epoch"],
    )
    tf_sampler = TransformerNFSampler(H, config=tf_config, device="cpu")
    tf_sampler.train(verbose=True)
    tf_train_time = _time.time() - train_t0
    print(f"  Transformer NF training completed in {tf_train_time:.2f}s")

    # Build solvers
    n_samples = cfg["nf_n_samples"]
    skqd_max_basis = cfg["skqd_max_basis"]

    solvers = {}
    if "TF-SQD" in tf_methods:
        solvers["TF-SQD"] = SQDSolver(tf_sampler, SQDConfig(n_samples=n_samples))
    if "TF-SKQD-B" in tf_methods:
        solvers["TF-SKQD-B"] = SKQDSolverB(tf_sampler, SKQDConfig(
            n_samples=n_samples, max_basis_size=skqd_max_basis,
        ))
    if "TF-SKQD-C" in tf_methods:
        solvers["TF-SKQD-C"] = SKQDSolverC(tf_sampler, SKQDConfig(
            n_samples=n_samples, max_basis_size=skqd_max_basis,
        ))

    for method_name, solver in solvers.items():
        try:
            result = solver.solve(H, mol_info)

            total_time = tf_train_time + result.wall_time

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"  {method_name:<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {total_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": result.energy,
                "error_mha": error_mha,
                "diag_dim": result.diag_dim,
                "wall_time": total_time,
                "converged": result.converged,
                "metadata": {
                    **result.metadata,
                    "tf_train_time": tf_train_time,
                    "solve_time": result.wall_time,
                },
            })

        except Exception as e:
            print(f"  {method_name:<14} ERROR: {e}")
            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": None,
                "error_mha": None,
                "diag_dim": 0,
                "wall_time": 0,
                "converged": False,
                "metadata": {"error": str(e)},
            })

    return results


def _run_iterative_methods(mol_name, H, mol_info, methods, fci_energies):
    """Run Iterative NF-SQD and Iterative NF-SKQD."""
    iter_methods = [m for m in methods if m.startswith("Iter-")]
    if not iter_methods:
        return []

    cfg = EXPERIMENT_CONFIG[mol_name]
    results = []

    # Build a fresh flow for iterative training
    # Use ParticleConservingFlow (same as NF methods)
    from src.flows.particle_conserving_flow import ParticleConservingFlowSampler
    n_sites = H.num_sites
    n_alpha = H.n_alpha
    n_beta = H.n_beta

    for method_name in iter_methods:
        try:
            # Create a fresh flow for each method
            flow = ParticleConservingFlowSampler(
                num_sites=n_sites,
                n_alpha=n_alpha,
                n_beta=n_beta,
                temperature=1.0,
            )

            if method_name == "Iter-NF-SQD":
                iter_config = IterativeNFSQDConfig(
                    n_samples=cfg["nf_n_samples"],
                    max_basis_size=cfg["skqd_max_basis"],
                    do_expansion=False,
                )
                solver = IterativeNFSQDSolver(flow, iter_config)
            else:  # Iter-NF-SKQD
                iter_config = IterativeNFSQDConfig(
                    n_samples=cfg["nf_n_samples"],
                    max_basis_size=cfg["skqd_max_basis"],
                    do_expansion=True,
                    expansion_size=500,
                    n_expansion_configs=200,
                )
                solver = IterativeNFSKQDSolver(flow, iter_config)

            result = solver.solve(H, mol_info)

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"  {method_name:<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": result.energy,
                "error_mha": error_mha,
                "diag_dim": result.diag_dim,
                "wall_time": result.wall_time,
                "converged": result.converged,
                "metadata": result.metadata,
            })

        except Exception as e:
            print(f"  {method_name:<14} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": None,
                "error_mha": None,
                "diag_dim": 0,
                "wall_time": 0,
                "converged": False,
                "metadata": {"error": str(e)},
            })

    return results


def _run_dci_methods(mol_name, H, mol_info, methods, fci_energies):
    """Run DCI-SKQD-B and DCI-SKQD-C (CIPSI-seeded SKQD)."""
    dci_methods = [m for m in methods if m.startswith("DCI-")]
    if not dci_methods:
        return []

    cfg = EXPERIMENT_CONFIG[mol_name]
    results = []

    dci_config = DCISKQDConfig(max_basis_size=cfg["skqd_max_basis"])

    solvers = {}
    if "DCI-SKQD-B" in dci_methods:
        solvers["DCI-SKQD-B"] = DCISKQDSolverB(dci_config)
    if "DCI-SKQD-C" in dci_methods:
        solvers["DCI-SKQD-C"] = DCISKQDSolverC(dci_config)

    for method_name, solver in solvers.items():
        try:
            result = solver.solve(H, mol_info)

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"  {method_name:<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": result.energy,
                "error_mha": error_mha,
                "diag_dim": result.diag_dim,
                "wall_time": result.wall_time,
                "converged": result.converged,
                "metadata": result.metadata,
            })

        except Exception as e:
            print(f"  {method_name:<14} ERROR: {e}")
            results.append({
                "molecule": mol_name,
                "method": method_name,
                "energy": None,
                "error_mha": None,
                "diag_dim": 0,
                "wall_time": 0,
                "converged": False,
                "metadata": {"error": str(e)},
            })

    return results


def _run_quantum_methods(mol_name, H, mol_info, methods, fci_energies):
    """Run Q-SQD and Q-SKQD using quantum circuit samplers."""
    q_methods = [m for m in methods if m.startswith("Q-")]
    if not q_methods:
        return []

    results = []

    if "Q-SQD" in q_methods:
        try:
            from src.samplers.lucj_sampler import LUCJSampler
            lucj_sampler = LUCJSampler(H, n_reps=2)
            q_sqd = SQDSolver(lucj_sampler, SQDConfig(n_samples=10000))
            result = q_sqd.solve(H, mol_info)

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"  {'Q-SQD':<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name, "method": "Q-SQD",
                "energy": result.energy, "error_mha": error_mha,
                "diag_dim": result.diag_dim, "wall_time": result.wall_time,
                "converged": result.converged, "metadata": result.metadata,
            })
        except ImportError as e:
            print(f"  {'Q-SQD':<14} SKIP (missing dep): {e}")
        except Exception as e:
            print(f"  {'Q-SQD':<14} ERROR: {e}")
            results.append({
                "molecule": mol_name, "method": "Q-SQD",
                "energy": None, "error_mha": None, "diag_dim": 0,
                "wall_time": 0, "converged": False,
                "metadata": {"error": str(e)},
            })

    if "Q-SKQD" in q_methods:
        try:
            from src.samplers.trotter_sampler import TrotterSampler
            trotter_sampler = TrotterSampler(H, n_krylov_steps=6, dt=0.1)
            q_skqd = SKQDSolverC(trotter_sampler, SKQDConfig(n_samples=5000))
            result = q_skqd.solve(H, mol_info)

            error_mha = None
            if result.energy is not None and mol_name in fci_energies:
                error_mha = (result.energy - fci_energies[mol_name]) * 1000

            e_str = f"{result.energy:.10f}" if result.energy is not None else "N/A"
            err_str = f"{error_mha:.3f}" if error_mha is not None else "N/A"
            status = "OK" if result.converged else "FAIL"

            print(f"  {'Q-SKQD':<14} {e_str:>18} {err_str:>12} "
                  f"{result.diag_dim:>10} {result.wall_time:>10.2f}  {status}")

            results.append({
                "molecule": mol_name, "method": "Q-SKQD",
                "energy": result.energy, "error_mha": error_mha,
                "diag_dim": result.diag_dim, "wall_time": result.wall_time,
                "converged": result.converged, "metadata": result.metadata,
            })
        except ImportError as e:
            print(f"  {'Q-SKQD':<14} SKIP (missing dep): {e}")
        except Exception as e:
            print(f"  {'Q-SKQD':<14} ERROR: {e}")
            results.append({
                "molecule": mol_name, "method": "Q-SKQD",
                "energy": None, "error_mha": None, "diag_dim": 0,
                "wall_time": 0, "converged": False,
                "metadata": {"error": str(e)},
            })

    return results


def print_summary(all_results):
    """Print summary table grouped by molecule."""
    by_mol = defaultdict(list)
    for r in all_results:
        by_mol[r["molecule"]].append(r)

    fci_energies = {}
    for r in all_results:
        if r["method"] == "FCI" and r.get("energy") is not None:
            fci_energies[r["molecule"]] = r["energy"]

    print(f"\n{'Molecule':<16} {'Method':<14} {'Energy (Ha)':>18} {'Error (mHa)':>12} "
          f"{'Diag Dim':>10} {'Time (s)':>10}")
    print("=" * 86)

    # Print in molecule order
    for mol_name in ALL_MOLECULES:
        if mol_name not in by_mol:
            continue
        fci_e = fci_energies.get(mol_name)
        for r in by_mol[mol_name]:
            e = r.get("energy")
            e_str = f"{e:.10f}" if e is not None else "N/A"

            err = r.get("error_mha")
            if err is None and e is not None and fci_e is not None:
                err = (e - fci_e) * 1000
            err_str = f"{err:.3f}" if err is not None else "N/A"

            print(f"{mol_name:<16} {r['method']:<14} {e_str:>18} {err_str:>12} "
                  f"{r.get('diag_dim', 0):>10} {r.get('wall_time', 0):>10.2f}")
        print()


def run_benchmark(molecule_names=None, max_tier=None):
    """Run the full benchmark.

    Args:
        molecule_names: List of molecule names, or None for all.
        max_tier: If set, only run molecules with tier <= max_tier.
    """
    # Determine which molecules to run
    if molecule_names is None:
        molecule_names = list(ALL_MOLECULES)

    if max_tier is not None:
        molecule_names = [
            m for m in molecule_names
            if EXPERIMENT_CONFIG.get(m, {}).get("tier", 99) <= max_tier
        ]

    if not molecule_names:
        print("No molecules to run.")
        return []

    print(f"\nBenchmark: {len(molecule_names)} molecules")
    print(f"Molecules: {', '.join(molecule_names)}")
    print()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    fci_energies = {}
    total_t0 = time.time()

    for mol_name in molecule_names:
        mol_t0 = time.time()

        # Get molecule
        try:
            H, mol_info = get_molecule(mol_name)
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"SKIP {mol_name}: {e}")
            continue

        # Determine feasible methods
        methods = get_methods_for_molecule(mol_name, mol_info)
        tier = EXPERIMENT_CONFIG[mol_name]["tier"]

        print(f"\n{'='*80}")
        print(f"{mol_name} (Tier {tier}, {mol_info['n_qubits']}Q, "
              f"{'CAS' if mol_info['is_cas'] else 'Full'}) — "
              f"Methods: {', '.join(methods)}")
        print(f"  {'Method':<14} {'Energy (Ha)':>18} {'Error (mHa)':>12} "
              f"{'Diag Dim':>10} {'Time (s)':>10}  {'Status'}")
        print(f"  {'-'*76}")

        # Phase 1: Baselines
        baseline_results = _run_baselines(mol_name, H, mol_info, methods, fci_energies)
        all_results.extend(baseline_results)

        # Phase 2: NF methods
        nf_results = _run_nf_methods(mol_name, H, mol_info, methods, fci_energies)
        all_results.extend(nf_results)

        # Phase 3: Transformer NF methods
        tf_results = _run_tf_methods(mol_name, H, mol_info, methods, fci_energies)
        all_results.extend(tf_results)

        # Phase 4: DCI-SKQD methods
        dci_results = _run_dci_methods(mol_name, H, mol_info, methods, fci_energies)
        all_results.extend(dci_results)

        # Phase 5: Iterative NF-SQD methods
        iter_results = _run_iterative_methods(mol_name, H, mol_info, methods, fci_energies)
        all_results.extend(iter_results)

        # Phase 4: Quantum methods
        q_results = _run_quantum_methods(mol_name, H, mol_info, methods, fci_energies)
        all_results.extend(q_results)

        mol_time = time.time() - mol_t0
        print(f"  Total time for {mol_name}: {mol_time:.1f}s")

    total_time = time.time() - total_t0

    # Save JSON
    json_path = results_dir / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save CSV
    csv_path = results_dir / "benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "molecule", "method", "energy", "error_mha",
            "diag_dim", "wall_time", "converged",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "molecule": r["molecule"],
                "method": r["method"],
                "energy": r.get("energy"),
                "error_mha": r.get("error_mha"),
                "diag_dim": r.get("diag_dim", 0),
                "wall_time": f"{r.get('wall_time', 0):.2f}",
                "converged": r.get("converged", False),
            })

    # Summary
    print(f"\n\n{'='*40} SUMMARY {'='*40}")
    print_summary(all_results)

    print(f"Total benchmark time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run NQS-SQD benchmark")
    parser.add_argument("--tier", type=int, default=None,
                        help="Max tier to run (1-5). Default: all tiers.")
    parser.add_argument("--molecules", type=str, default=None,
                        help="Comma-separated molecule names (e.g. H2,LiH,N2)")
    args = parser.parse_args()

    molecule_names = None
    if args.molecules:
        # Smart split: don't split commas inside parentheses
        # e.g. "N2-CAS(10,26),Cr2-CAS(12,28)" -> ["N2-CAS(10,26)", "Cr2-CAS(12,28)"]
        parts = []
        depth = 0
        current = []
        for ch in args.molecules:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())
        molecule_names = [p for p in parts if p]

    run_benchmark(molecule_names=molecule_names, max_tier=args.tier)


if __name__ == "__main__":
    main()
