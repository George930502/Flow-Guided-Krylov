"""Base classes for solvers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolverResult:
    """Result from a solver run."""
    energy: Optional[float]  # Hartree (None if failed/skipped)
    diag_dim: int            # 0 for non-diag methods
    wall_time: float         # seconds
    method: str
    converged: bool
    metadata: dict = field(default_factory=dict)


class Solver(ABC):
    """Abstract base class for solvers."""

    @abstractmethod
    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        """Solve for ground state energy.

        Args:
            hamiltonian: MolecularHamiltonian instance
            mol_info: dict with molecule metadata (name, n_qubits, basis, is_cas, etc.)

        Returns:
            SolverResult with energy and diagnostics
        """
        ...
