#!/usr/bin/env python3
"""C2H4 (28Q) with larger basis (30000) and mini-batch NQS update."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")

H, info = get_molecule("C2H4")
print("C2H4 (28Q, 14 orb)")

# SCI reference (reuse known result)
print("  SCI     = -77.2351408123 Ha (basis=10000, from previous run)")

# HI+NQS+SQD with larger basis
np.random.seed(42); torch.manual_seed(42)
cfg = HINQSSQDConfig(
    n_samples=20000,
    max_iterations=30,
    max_basis_size=15000,
    num_batches=3,
    nf_steps=10,
)
r = run_hi_nqs_sqd(H, info, config=cfg)
sci_e = -77.2351408123
err = (r.energy - sci_e) * 1000 if r.energy else None
err_s = f"{err:.3f}" if err else "N/A"
print(f"  HI+NQS+SQD = {r.energy:.10f} Ha, err vs SCI={err_s} mHa, basis={r.diag_dim}, time={r.wall_time:.1f}s")
