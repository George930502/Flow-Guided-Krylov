#!/usr/bin/env python3
"""Run remaining molecules: C2H2(24Q), C2H4(28Q), Benzene(30Q)."""
import sys, numpy as np, torch, time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.ccsd import CCSDSolver, CCSDTSolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")

molecules = [
    ("C2H2", 24, 20000),
    ("C2H4", 28, 20000),
    ("Benzene", 30, 20000),
]

for mol_name, nq, n_samp in molecules:
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    is_cas = info.get("is_cas", False)

    print(f"\n{'='*70}")
    print(f"{mol_name} ({nq}Q, {n_orb} orb, Hilbert={hilbert:,})")

    # References
    fci_r = FCISolver().solve(H, info)
    if fci_r.energy:
        print(f"  FCI     = {fci_r.energy:.10f} Ha")

    if not is_cas:
        ccsdt_r = CCSDTSolver().solve(H, info)
        if ccsdt_r.energy:
            print(f"  CCSD(T) = {ccsdt_r.energy:.10f} Ha")

    sci_r = CIPSISolver(max_basis_size=10000).solve(H, info)
    if sci_r.energy:
        print(f"  SCI     = {sci_r.energy:.10f} Ha (basis={sci_r.diag_dim})")

    # HI+NQS+SQD with max_basis=10000 to avoid OOM
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samp,
        max_iterations=30,
        max_basis_size=10000,
        num_batches=3,
        nf_steps=10,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)

    ref_e = fci_r.energy or sci_r.energy
    if r.energy:
        if ref_e:
            err = (r.energy - ref_e) * 1000
            print(f"  HI+NQS+SQD = {r.energy:.10f} Ha, err={err:.3f} mHa, "
                  f"basis={r.diag_dim}, time={r.wall_time:.1f}s")
        else:
            print(f"  HI+NQS+SQD = {r.energy:.10f} Ha, "
                  f"basis={r.diag_dim}, time={r.wall_time:.1f}s")

print("\nDone!")
