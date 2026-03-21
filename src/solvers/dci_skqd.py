"""
DCI-SKQD: Deterministic CI seeded Krylov Quantum Diagonalization.

Uses CIPSI (deterministic perturbative selection) to build the initial
basis, then continues expansion with SKQD strategies (Method B or C).

This avoids the cost of NF training while providing a high-quality
initial basis — every configuration is selected by PT2 importance
rather than stochastic sampling.

Two variants:
  - DCI-SKQD-B: CIPSI seed → SKQD-B expansion (PT2 importance scoring)
  - DCI-SKQD-C: CIPSI seed → SKQD-C expansion (coupling strength scoring)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .base import Solver, SolverResult
from ..utils.config_hash import config_integer_hash

logger = logging.getLogger(__name__)


@dataclass
class DCISKQDConfig:
    """Configuration for DCI-SKQD solvers."""
    # CIPSI seeding phase
    cipsi_iterations: int = 5          # number of CIPSI iterations for seeding
    cipsi_expansion_size: int = 500    # configs added per CIPSI iteration

    # SKQD expansion phase
    max_iterations: int = 20           # total iterations (CIPSI + SKQD)
    expansion_size: int = 500          # configs added per SKQD iteration
    max_basis_size: int = 10000
    convergence_threshold: float = 1e-5  # Ha

    # How many high-amplitude configs to process per SKQD iteration
    n_process: int = 200


def _build_cipsi_basis(hamiltonian, config):
    """Run CIPSI seeding phase: build initial basis from HF via PT2 selection.

    Returns (basis, basis_hashes, e0, psi0, iteration_energies).
    """
    hf_config = hamiltonian.get_hf_state()
    if hf_config.dim() == 1:
        hf_config = hf_config.unsqueeze(0)

    basis = hf_config.clone()
    basis_hashes = set(config_integer_hash(basis))
    iteration_energies = []
    e0 = None
    psi0 = None

    for it in range(config.cipsi_iterations):
        n_basis = basis.shape[0]

        # Diagonalize
        h_matrix = hamiltonian.matrix_elements_fast(basis)
        h_np = np.asarray(h_matrix, dtype=np.float64)
        h_np = 0.5 * (h_np + h_np.T)
        eigvals, eigvecs = np.linalg.eigh(h_np)
        e0 = float(eigvals[0])
        psi0 = eigvecs[:, 0]
        iteration_energies.append(e0)

        if n_basis >= config.max_basis_size:
            break

        # Collect candidates via full CIPSI scan (all basis configs)
        numerator_accum = defaultdict(float)
        candidate_configs = {}

        for idx in range(n_basis):
            c_i = float(psi0[idx])
            if abs(c_i) < 1e-14:
                continue

            connections, h_elements = hamiltonian.get_connections(basis[idx])
            if connections is None or len(connections) == 0:
                continue

            conn_hashes = config_integer_hash(connections)
            for j, h_conn in enumerate(conn_hashes):
                if h_conn in basis_hashes:
                    continue
                numerator_accum[h_conn] += c_i * float(h_elements[j])
                if h_conn not in candidate_configs:
                    candidate_configs[h_conn] = connections[j]

        if not candidate_configs:
            break

        # PT2 importance scoring
        cand_hash_list = list(candidate_configs.keys())
        cand_tensor = torch.stack([candidate_configs[h] for h in cand_hash_list])
        h_diag = np.asarray(
            hamiltonian.diagonal_elements_batch(cand_tensor), dtype=np.float64
        )

        importances = np.empty(len(cand_hash_list), dtype=np.float64)
        for k, h_key in enumerate(cand_hash_list):
            numer_sq = numerator_accum[h_key] ** 2
            denom = abs(e0 - h_diag[k])
            importances[k] = numer_sq / max(denom, 1e-14)

        # Select top-k
        room = config.max_basis_size - n_basis
        n_add = min(config.cipsi_expansion_size, len(cand_hash_list), room)
        if n_add >= len(cand_hash_list):
            top_indices = np.arange(len(cand_hash_list))
        else:
            top_indices = np.argpartition(-importances, n_add)[:n_add]

        new_configs = cand_tensor[top_indices]
        new_hashes = [cand_hash_list[i] for i in top_indices]
        basis = torch.cat([basis, new_configs], dim=0)
        basis_hashes.update(new_hashes)

    return basis, basis_hashes, e0, psi0, iteration_energies


class DCISKQDSolverB(Solver):
    """DCI-SKQD Method B: CIPSI seed → PT2 importance expansion.

    Phase 1 (CIPSI): Full deterministic scan of all basis configs' connections.
    Phase 2 (SKQD-B): Process top-amplitude configs only, PT2 scoring.
    """

    def __init__(self, config=None):
        self.config = config or DCISKQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config

        # Phase 1: CIPSI seeding
        basis, basis_hashes, e0, psi0, iter_energies = _build_cipsi_basis(
            hamiltonian, cfg
        )
        cipsi_basis_size = len(basis)
        cipsi_time = time.time() - t0

        prev_energy = e0 if e0 is not None else float("inf")
        converged = False

        # Phase 2: SKQD-B expansion (PT2 importance, top-amplitude only)
        remaining_iters = cfg.max_iterations - cfg.cipsi_iterations
        for iteration in range(remaining_iters):
            if len(basis) > cfg.max_basis_size:
                break

            # Diagonalize
            H_proj = hamiltonian.matrix_elements_fast(basis)
            H_np = H_proj.cpu().numpy().astype(np.float64)
            H_np = 0.5 * (H_np + H_np.T)

            if len(H_np) <= 2000:
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            else:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                eigenvalues, eigenvectors = eigsh(
                    csr_matrix(H_np), k=1, which="SA"
                )

            e0 = float(eigenvalues[0])
            psi0 = eigenvectors[:, 0]
            iter_energies.append(e0)

            # Check convergence
            delta_e = abs(e0 - prev_energy)
            if delta_e < cfg.convergence_threshold:
                converged = True
                break
            prev_energy = e0

            # Find new configs via H-connections (top-amplitude only)
            new_configs = []
            new_importance = []

            sorted_idx = np.argsort(np.abs(psi0))[::-1]
            n_process = min(len(sorted_idx), cfg.n_process)

            for idx in sorted_idx[:n_process]:
                c_i = psi0[idx]
                if abs(c_i) < 1e-8:
                    continue

                connected, elements = hamiltonian.get_connections(basis[idx])
                if len(connected) == 0:
                    continue

                conn_hashes = config_integer_hash(connected)
                for k, h in enumerate(conn_hashes):
                    if h not in basis_hashes:
                        h_elem = float(elements[k])
                        coupling = c_i * h_elem
                        h_xx = float(hamiltonian.diagonal_element(connected[k]))
                        denom = abs(e0 - h_xx) + 1e-12
                        importance = coupling ** 2 / denom

                        new_configs.append(connected[k])
                        new_importance.append(importance)
                        basis_hashes.add(h)

            if not new_configs:
                converged = True
                break

            importance_arr = np.array(new_importance)
            top_k = min(cfg.expansion_size, len(new_configs))
            top_indices = np.argsort(importance_arr)[-top_k:]

            new_batch = torch.stack([new_configs[i] for i in top_indices])
            basis = torch.cat([basis, new_batch], dim=0)

        wall_time = time.time() - t0

        return SolverResult(
            energy=e0,
            diag_dim=len(basis),
            wall_time=wall_time,
            method="DCI-SKQD-B",
            converged=converged,
            metadata={
                "total_iterations": len(iter_energies),
                "cipsi_iterations": cfg.cipsi_iterations,
                "cipsi_basis_size": cipsi_basis_size,
                "cipsi_time": cipsi_time,
                "final_basis_size": len(basis),
                "iteration_energies": iter_energies,
                "delta_e": delta_e if "delta_e" in dir() else None,
            },
        )


class DCISKQDSolverC(Solver):
    """DCI-SKQD Method C: CIPSI seed → coupling strength expansion.

    Phase 1 (CIPSI): Full deterministic scan of all basis configs' connections.
    Phase 2 (SKQD-C): Process boundary configs, coupling score |c_i|*|H_{x',x_i}|.
    """

    def __init__(self, config=None):
        self.config = config or DCISKQDConfig()

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        t0 = time.time()
        cfg = self.config

        # Phase 1: CIPSI seeding
        basis, basis_hashes, e0, psi0, iter_energies = _build_cipsi_basis(
            hamiltonian, cfg
        )
        cipsi_basis_size = len(basis)
        cipsi_time = time.time() - t0

        prev_energy = e0 if e0 is not None else float("inf")
        converged = False

        # Phase 2: SKQD-C expansion (coupling strength scoring)
        remaining_iters = cfg.max_iterations - cfg.cipsi_iterations
        for iteration in range(remaining_iters):
            if len(basis) > cfg.max_basis_size:
                break

            # Diagonalize
            H_proj = hamiltonian.matrix_elements_fast(basis)
            H_np = H_proj.cpu().numpy().astype(np.float64)
            H_np = 0.5 * (H_np + H_np.T)

            if len(H_np) <= 2000:
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            else:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                eigenvalues, eigenvectors = eigsh(
                    csr_matrix(H_np), k=1, which="SA"
                )

            e0 = float(eigenvalues[0])
            psi0 = eigenvectors[:, 0]
            iter_energies.append(e0)

            # Check convergence
            delta_e = abs(e0 - prev_energy)
            if delta_e < cfg.convergence_threshold:
                converged = True
                break
            prev_energy = e0

            # Identify boundary configs and collect external connections
            sorted_idx = np.argsort(np.abs(psi0))[::-1]
            n_boundary = min(len(sorted_idx), cfg.n_process)

            external_configs = []
            external_scores = []

            for idx in sorted_idx[:n_boundary]:
                c_i = abs(psi0[idx])
                if c_i < 1e-8:
                    continue

                connected, elements = hamiltonian.get_connections(basis[idx])
                if len(connected) == 0:
                    continue

                conn_hashes = config_integer_hash(connected)
                for k, h in enumerate(conn_hashes):
                    if h not in basis_hashes:
                        score = c_i * abs(float(elements[k]))
                        external_configs.append(connected[k])
                        external_scores.append(score)
                        basis_hashes.add(h)

            if not external_configs:
                converged = True
                break

            scores_arr = np.array(external_scores)
            top_k = min(cfg.expansion_size, len(external_configs))
            top_indices = np.argsort(scores_arr)[-top_k:]

            new_batch = torch.stack([external_configs[i] for i in top_indices])
            basis = torch.cat([basis, new_batch], dim=0)

        wall_time = time.time() - t0

        return SolverResult(
            energy=e0,
            diag_dim=len(basis),
            wall_time=wall_time,
            method="DCI-SKQD-C",
            converged=converged,
            metadata={
                "total_iterations": len(iter_energies),
                "cipsi_iterations": cfg.cipsi_iterations,
                "cipsi_basis_size": cipsi_basis_size,
                "cipsi_time": cipsi_time,
                "final_basis_size": len(basis),
                "iteration_energies": iter_energies,
                "delta_e": delta_e if "delta_e" in dir() else None,
            },
        )
