#!/usr/bin/env python3
"""Test HI+NQS+SQD on 20Q-30Q molecules with GPU + IBM SQD."""
import sys, numpy as np, torch, time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")

molecules = [
    ("N2",   20, 20000),
    ("HCN",  22, 30000),
    ("C2H2", 24, 30000),
    ("H2S",  26, 30000),
    ("C2H4", 28, 30000),
    ("Benzene", 30, 30000),
]

for mol_name, nq, n_samp in molecules:
    t_build = time.time()
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    build_time = time.time() - t_build

    fci_r = FCISolver().solve(H, info)
    fci_e = fci_r.energy

    print(f"\n{'='*70}")
    print(f"{mol_name} ({nq}Q, {n_orb} orb, Hilbert={hilbert:,})")
    if fci_e:
        print(f"FCI = {fci_e:.10f} Ha (built in {build_time:.1f}s)")
    else:
        print(f"FCI: SKIP (Hilbert too large, built in {build_time:.1f}s)")

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samp,
        max_iterations=30,
        max_basis_size=0,      # no limit
        num_batches=3,
        nf_steps=10,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)

    if r.energy:
        if fci_e:
            err = (r.energy - fci_e) * 1000
            print(f"  Result: E={r.energy:.10f}, err={err:.3f} mHa, "
                  f"basis={r.diag_dim}, time={r.wall_time:.1f}s")
        else:
            print(f"  Result: E={r.energy:.10f}, "
                  f"basis={r.diag_dim}, time={r.wall_time:.1f}s")
    else:
        print(f"  FAILED, time={r.wall_time:.1f}s")

print("\nDone!")
