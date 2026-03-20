"""
CIPSI (Configuration Interaction using a Perturbative Selection made Iteratively) solver.

Iteratively builds a compact CI basis by selecting the most important
determinants via second-order perturbation theory importance estimates,
then diagonalises the projected Hamiltonian in that basis.

Algorithm
---------
1. Seed the basis with the Hartree-Fock determinant.
2. At each iteration:
   a. Build the projected Hamiltonian in the current basis and diagonalise.
   b. For every basis determinant, enumerate its single/double excitations
      (H-connections) that are *not* already in the basis.
   c. Score each candidate determinant x by PT2 importance:
          eps_x = |<x|H|Phi>|^2 / |E_0 - H_xx|
   d. Add the top-k highest-scoring candidates to the basis.
   e. Stop when |Delta E| < threshold or the basis saturates.
3. Return the variational ground-state energy and basis size.
"""

import logging
import time
from collections import defaultdict

import numpy as np
import torch

from .base import Solver, SolverResult
from ..utils.config_hash import config_integer_hash

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 30
MAX_BASIS_SIZE = 10_000
CONVERGENCE_THRESHOLD = 1e-5  # Hartree
EXPANSION_SIZE = 500  # top-k new configs added per iteration


class CIPSISolver(Solver):
    """Selected CI solver using the CIPSI perturbative selection scheme."""

    def __init__(
        self,
        max_iterations: int = MAX_ITERATIONS,
        max_basis_size: int = MAX_BASIS_SIZE,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
        expansion_size: int = EXPANSION_SIZE,
    ):
        self.max_iterations = max_iterations
        self.max_basis_size = max_basis_size
        self.convergence_threshold = convergence_threshold
        self.expansion_size = expansion_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        """Run the CIPSI selected-CI algorithm.

        Parameters
        ----------
        hamiltonian
            A ``MolecularHamiltonian`` exposing ``get_hf_state()``,
            ``matrix_elements_fast()``, ``get_connections()``,
            ``diagonal_element()`` and ``diagonal_elements_batch()``.
        mol_info : dict
            Molecule metadata (unused by this solver, kept for API compat).

        Returns
        -------
        SolverResult
        """
        t0 = time.perf_counter()

        # 1. Seed with the HF determinant -----------------------------------
        hf_config = hamiltonian.get_hf_state()  # (n_sites,) tensor
        if hf_config.dim() == 1:
            hf_config = hf_config.unsqueeze(0)  # -> (1, n_sites)

        basis = hf_config.clone()  # (n_basis, n_sites)
        basis_hashes = set(config_integer_hash(basis))

        prev_energy = None
        converged = False
        iteration_energies = []

        iterator = range(self.max_iterations)
        if tqdm is not None:
            iterator = tqdm(iterator, desc="CIPSI", leave=False)

        # 2. Main loop -------------------------------------------------------
        for it in iterator:
            n_basis = basis.shape[0]

            # (a) Build and diagonalise projected H --------------------------
            h_matrix = hamiltonian.matrix_elements_fast(basis)  # (n, n) ndarray or tensor
            h_np = np.asarray(h_matrix, dtype=np.float64)
            eigvals, eigvecs = np.linalg.eigh(h_np)
            e0 = float(eigvals[0])
            coeffs = eigvecs[:, 0]  # ground-state eigenvector

            iteration_energies.append(e0)
            logger.debug(
                "CIPSI iter %d: basis=%d  E=%.10f Ha", it, n_basis, e0
            )

            # Convergence check ----------------------------------------------
            if prev_energy is not None:
                delta_e = abs(e0 - prev_energy)
                if delta_e < self.convergence_threshold:
                    converged = True
                    logger.info(
                        "CIPSI converged at iter %d: |dE|=%.2e < %.2e",
                        it, delta_e, self.convergence_threshold,
                    )
                    break
            prev_energy = e0

            # Basis-size guard ------------------------------------------------
            if n_basis >= self.max_basis_size:
                logger.warning(
                    "CIPSI basis reached max size (%d); stopping.", self.max_basis_size
                )
                break

            # (b-c) Collect candidate configs & accumulate PT2 numerators ----
            #   <x|H|Phi> = sum_i c_i * H_{x, x_i}
            numerator_accum = defaultdict(float)
            candidate_configs = {}

            for idx in range(n_basis):
                c_i = float(coeffs[idx])
                if abs(c_i) < 1e-14:
                    continue

                config_i = basis[idx]  # (n_sites,)
                connections, h_elements = hamiltonian.get_connections(config_i)
                # connections: (n_conn, n_sites) tensor
                # h_elements: (n_conn,) tensor or array of H matrix elements

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
                logger.info("CIPSI: no new candidates found; stopping.")
                break

            # (d) Compute PT2 importance for each candidate ------------------
            cand_hash_list = list(candidate_configs.keys())
            cand_tensor = torch.stack(
                [candidate_configs[h] for h in cand_hash_list]
            )  # (n_cand, n_sites)

            # Diagonal elements for all candidates at once
            h_diag = hamiltonian.diagonal_elements_batch(cand_tensor)
            h_diag = np.asarray(h_diag, dtype=np.float64)

            importances = np.empty(len(cand_hash_list), dtype=np.float64)
            for k, h_key in enumerate(cand_hash_list):
                numer_sq = numerator_accum[h_key] ** 2
                denom = abs(e0 - h_diag[k])
                if denom < 1e-14:
                    importances[k] = numer_sq / 1e-14
                else:
                    importances[k] = numer_sq / denom

            # (e) Select top-k and expand basis ------------------------------
            room = self.max_basis_size - n_basis
            n_add = min(self.expansion_size, len(cand_hash_list), room)
            if n_add >= len(cand_hash_list):
                top_indices = np.arange(len(cand_hash_list))
            else:
                top_indices = np.argpartition(-importances, n_add)[:n_add]

            new_configs = cand_tensor[top_indices]  # (n_add, n_sites)
            new_hashes = [cand_hash_list[i] for i in top_indices]

            basis = torch.cat([basis, new_configs], dim=0)
            basis_hashes.update(new_hashes)

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    E=f"{e0:.8f}", basis=basis.shape[0], ordered=False
                )

        # 3. Final result ----------------------------------------------------
        wall_time = time.perf_counter() - t0
        diag_dim = basis.shape[0]

        logger.info(
            "CIPSI finished: E=%.10f Ha  basis=%d  converged=%s  time=%.2fs",
            e0, diag_dim, converged, wall_time,
        )

        return SolverResult(
            energy=e0,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="CIPSI",
            converged=converged,
            metadata={
                "n_iterations": len(iteration_energies),
                "iteration_energies": iteration_energies,
                "max_iterations": self.max_iterations,
                "expansion_size": self.expansion_size,
                "convergence_threshold": self.convergence_threshold,
            },
        )
