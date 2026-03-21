#!/usr/bin/env python3
"""Test HI-VQE with more iterations, shots, and layers."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_vqe import run_hi_vqe, HIVQEConfig

H, info = get_molecule("LiH")
fci_e = -7.8823243789

configs_to_test = [
    {"shots": 5000,  "max_iterations": 10, "n_layers": 2, "cobyla_maxiter": 3},
    {"shots": 10000, "max_iterations": 20, "n_layers": 3, "cobyla_maxiter": 5},
    {"shots": 10000, "max_iterations": 30, "n_layers": 4, "cobyla_maxiter": 5},
]

for c in configs_to_test:
    print(f"\n{'='*60}")
    print(f"shots={c['shots']}, iters={c['max_iterations']}, layers={c['n_layers']}, cobyla={c['cobyla_maxiter']}")
    print(f"{'='*60}")

    np.random.seed(42)
    cfg = HIVQEConfig(
        shots=c["shots"],
        max_iterations=c["max_iterations"],
        n_layers=c["n_layers"],
        cobyla_maxiter=c["cobyla_maxiter"],
        configuration_recovery=False,
    )
    r = run_hi_vqe(H, info, config=cfg)
    if r.energy:
        err = (r.energy - fci_e) * 1000
        print(f"\nResult: E={r.energy:.10f}, error={err:.3f} mHa, time={r.wall_time:.1f}s")
    else:
        print("\nFAILED")
