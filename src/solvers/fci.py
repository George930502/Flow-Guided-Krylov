"""Full Configuration Interaction (FCI) solver."""

import time
from math import comb

from .base import Solver, SolverResult

MAX_FCI_CONFIGS = 500_000


class FCISolver(Solver):
    """Exact diagonalisation via FCI."""

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        """Run FCI and return the ground-state energy.

        The determinant space dimension is computed as
        C(n_orb, n_alpha) * C(n_orb, n_beta).  If this exceeds
        *MAX_FCI_CONFIGS* the calculation is skipped and ``energy=None``
        is returned.
        """
        integrals = hamiltonian.integrals
        n_orb = integrals.n_orbitals
        n_alpha = integrals.n_alpha
        n_beta = integrals.n_beta

        diag_dim = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

        if diag_dim > MAX_FCI_CONFIGS:
            return SolverResult(
                energy=None,
                diag_dim=diag_dim,
                wall_time=0.0,
                method="FCI",
                converged=False,
                metadata={
                    "skipped": True,
                    "reason": (
                        f"FCI determinant space ({diag_dim:,}) exceeds "
                        f"MAX_FCI_CONFIGS ({MAX_FCI_CONFIGS:,})"
                    ),
                },
            )

        t0 = time.perf_counter()
        energy = hamiltonian.fci_energy()
        wall_time = time.perf_counter() - t0

        return SolverResult(
            energy=energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="FCI",
            converged=True,
        )
