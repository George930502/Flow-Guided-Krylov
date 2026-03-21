#!/usr/bin/env python3
"""Test HI+NQS+SQD and HI-VQE on molecules from 12Q to 30Q."""

import sys
import numpy as np
import torch
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from src.methods.hi_vqe import run_hi_vqe, HIVQEConfig

molecules = [
    ("LiH", 12),
    ("H2O", 14),
    ("BeH2", 14),
    ("NH3", 16),
    ("CH4", 18),
    ("N2", 20),
    ("C2H2", 24),
    ("Benzene", 30),
]

n_seeds = 3

for mol_name, nq in molecules:
    t0 = time.time()
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    hilbert_size = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)

    # FCI
    fci_r = FCISolver().solve(H, info)
    fci_e = fci_r.energy

    print(f"\n{'='*80}")
    print(f"{mol_name} ({nq}Q, {n_orb} orb, Hilbert={hilbert_size:,})")
    if fci_e:
        print(f"FCI = {fci_e:.10f} Ha")
    else:
        print("FCI: SKIP (too large)")
    print(f"{'='*80}")

    # Scale n_samples: at least 8x Hilbert size, cap at 20000
    n_samp = max(5000, min(20000, int(hilbert_size * 8)))
    print(f"  n_samples = {n_samp}")

    # HI+NQS+SQD
    print(f"\n  HI+NQS+SQD:")
    nqs_energies = []
    nqs_errors = []
    nqs_times = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        cfg = HINQSSQDConfig(
            n_samples=n_samp,
            max_iterations=30,
            max_basis_size=min(hilbert_size, 15000),
        )
        r = run_hi_nqs_sqd(H, info, config=cfg)
        nqs_energies.append(r.energy)
        nqs_times.append(r.wall_time)
        if r.energy and fci_e:
            err = (r.energy - fci_e) * 1000
            nqs_errors.append(err)
            print(f"    seed {seed}: E={r.energy:.10f}, err={err:.3f} mHa, "
                  f"basis={r.diag_dim}, t={r.wall_time:.1f}s")
        else:
            print(f"    seed {seed}: E={r.energy}, basis={r.diag_dim}, "
                  f"t={r.wall_time:.1f}s")

    if nqs_errors:
        print(f"    => Mean: {np.mean(nqs_errors):.3f} +/- {np.std(nqs_errors):.3f} mHa, "
              f"time={np.mean(nqs_times):.1f}s")

    # HI-VQE (only feasible when subspace < 50000)
    if hilbert_size <= 50000:
        print(f"\n  HI-VQE:")
        vqe_energies = []
        vqe_errors = []
        vqe_times = []
        for seed in range(n_seeds):
            np.random.seed(seed * 42 + 7)
            cfg_vqe = HIVQEConfig(
                shots=n_samp,
                max_iterations=15,
                configuration_recovery=False,
            )
            r = run_hi_vqe(H, info, config=cfg_vqe)
            vqe_energies.append(r.energy)
            vqe_times.append(r.wall_time)
            if r.energy and fci_e:
                err = (r.energy - fci_e) * 1000
                vqe_errors.append(err)
                print(f"    seed {seed}: E={r.energy:.10f}, err={err:.3f} mHa, "
                      f"t={r.wall_time:.1f}s")
            else:
                print(f"    seed {seed}: E={r.energy}, t={r.wall_time:.1f}s")

        if vqe_errors:
            print(f"    => Mean: {np.mean(vqe_errors):.3f} +/- {np.std(vqe_errors):.3f} mHa, "
                  f"time={np.mean(vqe_times):.1f}s")
    else:
        print(f"\n  HI-VQE: SKIP (Hilbert={hilbert_size:,} > 50000)")

    mol_time = time.time() - t0
    print(f"\n  Total time for {mol_name}: {mol_time:.1f}s")

print("\n\nDone!")
