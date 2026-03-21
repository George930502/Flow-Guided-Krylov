"""LUCJ circuit sampler using Qiskit + ffsim."""

import time
from typing import Optional

import numpy as np
import torch

from .base import Sampler, SamplerResult

try:
    import ffsim
    from ffsim.qiskit import (
        FfsimSampler,
        PrepareHartreeFockJW,
        UCJOpSpinBalancedJW,
    )
    from qiskit.circuit import QuantumCircuit
    QISKIT_FFSIM_AVAILABLE = True
except ImportError:
    QISKIT_FFSIM_AVAILABLE = False


class LUCJSampler(Sampler):
    """LUCJ circuit sampler using Qiskit + ffsim.

    1. PySCF RHF -> CCSD -> t1, t2 amplitudes
    2. ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1, n_reps=2)
    3. Build Qiskit circuit: PrepareHartreeFockJW + UCJOpSpinBalancedJW
    4. Simulate with ffsim.qiskit.FfsimSampler
    5. Return bitstring samples as configs
    """

    def __init__(
        self,
        hamiltonian,
        n_reps: int = 2,
        device: str = "cpu",
    ):
        if not QISKIT_FFSIM_AVAILABLE:
            raise ImportError(
                "qiskit and ffsim are required for LUCJSampler. "
                "Install with: pip install 'nqs-sqd[quantum]'"
            )

        self.hamiltonian = hamiltonian
        self.n_reps = n_reps
        self.device = device

        self.n_orbitals = hamiltonian.n_orbitals
        self.n_alpha = hamiltonian.n_alpha
        self.n_beta = hamiltonian.n_beta
        self.n_qubits = 2 * self.n_orbitals

        # Build LUCJ circuit from CCSD amplitudes
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> "QuantumCircuit":
        """Build LUCJ circuit from molecular integrals."""
        integrals = self.hamiltonian.integrals

        # Run PySCF CCSD to get amplitudes
        import pyscf
        from pyscf import gto, scf, cc

        # Reconstruct mol from integrals if possible
        if integrals._geometry is not None:
            mol = gto.Mole()
            mol.atom = integrals._geometry
            mol.basis = integrals._basis
            mol.charge = integrals._charge
            mol.spin = integrals._spin
            mol.build()

            mf = scf.RHF(mol)
            mf.kernel()

            mycc = cc.CCSD(mf)
            mycc.kernel()

            t1 = mycc.t1
            t2 = mycc.t2
        else:
            # For CAS systems, use identity amplitudes
            n_occ = self.n_alpha
            n_virt = self.n_orbitals - n_occ
            t1 = np.zeros((n_occ, n_virt))
            t2 = np.zeros((n_occ, n_occ, n_virt, n_virt))

        # Build UCJ operator
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2, t1_amplitudes=t1, n_reps=self.n_reps
        )

        # Build circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.append(
            PrepareHartreeFockJW(self.n_orbitals, (self.n_alpha, self.n_beta)),
            range(self.n_qubits),
        )
        qc.append(
            UCJOpSpinBalancedJW(ucj_op),
            range(self.n_qubits),
        )
        qc.measure_all()

        return qc

    def sample(self, n_samples: int) -> SamplerResult:
        """Sample bitstrings from the LUCJ circuit."""
        t0 = time.time()

        sampler = FfsimSampler(default_shots=n_samples, seed=42)
        job = sampler.run([self._circuit])
        result = job.result()
        counts = result[0].data.meas.get_counts()

        # Convert bitstrings to tensor configs
        configs_list = []
        for bitstring, count in counts.items():
            config = [int(b) for b in bitstring]
            for _ in range(count):
                configs_list.append(config)

        configs = torch.tensor(configs_list, dtype=torch.long, device=self.device)

        # Deduplicate
        unique_configs = torch.unique(configs, dim=0)

        wall_time = time.time() - t0

        return SamplerResult(
            configs=unique_configs,
            log_probs=None,
            wall_time=wall_time,
            metadata={
                "n_raw_samples": n_samples,
                "n_unique": len(unique_configs),
                "n_reps": self.n_reps,
                "sampler_type": "LUCJ",
            },
        )
