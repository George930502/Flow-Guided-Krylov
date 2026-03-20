#!/usr/bin/env python3
"""Test HI+NQS+SQD on 40Q and 52Q with SCI reference."""
import sys, numpy as np, torch, time
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

molecules = [
    ("N2-CAS(10,20)", 40),
    ("N2-CAS(10,26)", 52),
]

for mol_name, nq in molecules:
    print(f"\n{'='*70}", flush=True)
    print(f"{mol_name} ({nq}Q)", flush=True)

    t0 = time.time()
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f"  Hamiltonian built in {time.time()-t0:.1f}s, Hilbert={hilbert:,}", flush=True)

    # SCI reference
    t0 = time.time()
    sci_r = CIPSISolver(max_basis_size=10000).solve(H, info)
    sci_time = time.time() - t0
    if sci_r.energy:
        print(f"  SCI = {sci_r.energy:.10f} Ha, basis={sci_r.diag_dim}, time={sci_time:.1f}s", flush=True)
    else:
        print(f"  SCI: FAILED, time={sci_time:.1f}s", flush=True)

    # HI+NQS+SQD
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=20000,
        max_iterations=30,
        max_basis_size=10000,
        num_batches=3,
        nf_steps=10,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)

    if r.energy and sci_r.energy:
        err = (r.energy - sci_r.energy) * 1000
        print(f"  HI+NQS+SQD = {r.energy:.10f} Ha, err vs SCI={err:.3f} mHa, "
              f"basis={r.diag_dim}, time={r.wall_time:.1f}s", flush=True)
    elif r.energy:
        print(f"  HI+NQS+SQD = {r.energy:.10f} Ha, "
              f"basis={r.diag_dim}, time={r.wall_time:.1f}s", flush=True)

print("\nDone!", flush=True)
