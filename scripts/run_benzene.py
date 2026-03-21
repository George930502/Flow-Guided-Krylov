#!/usr/bin/env python3
"""Benzene (30Q) only — SCI + HI+NQS+SQD."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")

H, info = get_molecule("Benzene")
print(f"Benzene (30Q, 15 orb, CAS(6,15))")

# FCI (might work, Hilbert=207,025)
r = FCISolver().solve(H, info)
if r.energy:
    print(f"  FCI     = {r.energy:.10f} Ha, time={r.wall_time:.1f}s")
    ref_e = r.energy
else:
    print(f"  FCI: SKIP (dim={r.diag_dim:,})")
    ref_e = None

# SCI
r = CIPSISolver(max_basis_size=10000).solve(H, info)
print(f"  SCI     = {r.energy:.10f} Ha, basis={r.diag_dim}, time={r.wall_time:.1f}s")
if ref_e is None:
    ref_e = r.energy

# HI+NQS+SQD
np.random.seed(42); torch.manual_seed(42)
cfg = HINQSSQDConfig(n_samples=20000, max_iterations=30, max_basis_size=10000, num_batches=3, nf_steps=10)
r = run_hi_nqs_sqd(H, info, config=cfg)
err = (r.energy - ref_e) * 1000 if r.energy and ref_e else None
err_s = f"{err:.3f}" if err else "N/A"
print(f"  HI+NQS+SQD = {r.energy:.10f} Ha, err={err_s} mHa, basis={r.diag_dim}, time={r.wall_time:.1f}s")
