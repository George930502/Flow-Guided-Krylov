"""CCSD and CCSD(T) solvers backed by PySCF."""

import logging
import time

from .base import Solver, SolverResult

logger = logging.getLogger(__name__)


def _build_mol(mol_info: dict) -> "gto.Mole":
    """Return a PySCF Mole object from *mol_info*.

    If *mol_info* already contains a ``'pyscf_mol'`` key the object is
    returned directly.  Otherwise a new ``gto.Mole`` is built from the
    ``'geometry'``, ``'basis'``, ``'charge'``, and ``'spin'`` entries.
    """
    from pyscf import gto

    if "pyscf_mol" in mol_info:
        return mol_info["pyscf_mol"]

    mol = gto.Mole()
    mol.atom = mol_info["geometry"]
    mol.basis = mol_info["basis"]
    mol.charge = mol_info.get("charge", 0)
    mol.spin = mol_info.get("spin", 0)
    mol.build()
    return mol


class CCSDSolver(Solver):
    """Coupled-cluster singles and doubles (CCSD) via PySCF."""

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        """Run RHF -> CCSD and return the total energy."""
        if mol_info.get("is_cas", False):
            return SolverResult(
                energy=None,
                diag_dim=0,
                wall_time=0.0,
                method="CCSD",
                converged=False,
                metadata={"skipped": True, "reason": "CCSD not applicable to CAS-defined systems"},
            )

        mol = _build_mol(mol_info)

        from pyscf import scf, cc

        t0 = time.perf_counter()
        try:
            mf = scf.RHF(mol)
            mf.kernel()

            ccsd_obj = cc.CCSD(mf)
            ccsd_obj.kernel()
            converged = ccsd_obj.converged
            energy = ccsd_obj.e_tot
        except Exception as exc:  # noqa: BLE001
            logger.warning("CCSD failed: %s", exc)
            return SolverResult(
                energy=None,
                diag_dim=0,
                wall_time=time.perf_counter() - t0,
                method="CCSD",
                converged=False,
                metadata={"error": str(exc)},
            )
        wall_time = time.perf_counter() - t0

        return SolverResult(
            energy=energy,
            diag_dim=0,
            wall_time=wall_time,
            method="CCSD",
            converged=converged,
        )


class CCSDTSolver(Solver):
    """Coupled-cluster singles, doubles and perturbative triples CCSD(T)."""

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        """Run RHF -> CCSD -> (T) and return the total energy."""
        if mol_info.get("is_cas", False):
            return SolverResult(
                energy=None,
                diag_dim=0,
                wall_time=0.0,
                method="CCSD(T)",
                converged=False,
                metadata={"skipped": True, "reason": "CCSD not applicable to CAS-defined systems"},
            )

        mol = _build_mol(mol_info)

        from pyscf import scf, cc

        t0 = time.perf_counter()
        try:
            mf = scf.RHF(mol)
            mf.kernel()

            ccsd_obj = cc.CCSD(mf)
            ccsd_obj.kernel()
            converged = ccsd_obj.converged

            e_t = ccsd_obj.ccsd_t()
            energy = ccsd_obj.e_tot + e_t
        except Exception as exc:  # noqa: BLE001
            logger.warning("CCSD(T) failed: %s", exc)
            return SolverResult(
                energy=None,
                diag_dim=0,
                wall_time=time.perf_counter() - t0,
                method="CCSD(T)",
                converged=False,
                metadata={"error": str(exc)},
            )
        wall_time = time.perf_counter() - t0

        return SolverResult(
            energy=energy,
            diag_dim=0,
            wall_time=wall_time,
            method="CCSD(T)",
            converged=converged,
            metadata={"e_ccsd": ccsd_obj.e_tot, "e_triples": e_t},
        )
