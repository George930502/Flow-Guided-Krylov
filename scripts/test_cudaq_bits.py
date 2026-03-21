#!/usr/bin/env python3
"""Test CUDA-Q UCCSD ansatz particle number conservation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cudaq
import numpy as np
cudaq.set_target("qpp-cpu")

from src.samplers.cudaq_circuits import uccsd_ansatz, count_uccsd_params

# LiH: 12 qubits, 6 orbitals, 2 alpha, 2 beta
n_qubits = 12
n_alpha = 2
n_beta = 2
n_layers = 2
n_params = count_uccsd_params(6, n_layers)
print(f"LiH: {n_qubits}Q, {n_params} params, {n_layers} layers")

# Zero params = HF state
thetas_zero = [0.0] * n_params
result = cudaq.sample(uccsd_ansatz, n_qubits, n_alpha, n_beta, n_layers, thetas_zero, shots_count=100)
print("\nHF (zero params):")
for bs in result:
    print(f"  {bs}: {result.count(bs)}")

# Random params
thetas = (np.random.randn(n_params) * 0.5).tolist()
result = cudaq.sample(uccsd_ansatz, n_qubits, n_alpha, n_beta, n_layers, thetas, shots_count=5000)
print(f"\nRandom params ({n_params} params, 5000 shots):")
print(f"Unique bitstrings: {len(list(result))}")

# Check ALL samples preserve particle number
all_valid = True
total = 0
for bs in result:
    count = result.count(bs)
    total += count
    alpha = sum(int(b) for b in bs[:6])
    beta = sum(int(b) for b in bs[6:])
    if alpha != n_alpha or beta != n_beta:
        print(f"  INVALID: {bs} alpha={alpha} beta={beta} (count={count})")
        all_valid = False

print(f"\nParticle number conservation: {'ALL VALID' if all_valid else 'BROKEN'}")
print(f"Total samples checked: {total}")

# Show top bitstrings
print("\nTop bitstrings:")
for bs in result:
    count = result.count(bs)
    if count > 20:
        alpha = sum(int(b) for b in bs[:6])
        beta = sum(int(b) for b in bs[6:])
        print(f"  {bs}: {count} (α={alpha}, β={beta})")

# Test with LiH Hamiltonian
print("\n=== Full QC+SQD test ===")
from src.molecules import get_molecule
from src.methods.qc_sqd import run_qc_sqd

H, info = get_molecule("LiH")
r = run_qc_sqd(H, info, n_samples=5000)
fci_e = -7.8823243789
if r.energy:
    print(f"QC+SQD: E={r.energy:.10f}, err={( r.energy - fci_e)*1000:.3f} mHa")
else:
    print("QC+SQD: FAILED")

print("\n=== HI-VQE test (5 iters) ===")
from src.methods.hi_vqe import run_hi_vqe, HIVQEConfig
cfg = HIVQEConfig(shots=5000, max_iterations=5, configuration_recovery=False)
r = run_hi_vqe(H, info, config=cfg)
if r.energy:
    print(f"HI-VQE: E={r.energy:.10f}, err={(r.energy - fci_e)*1000:.3f} mHa")
else:
    print("HI-VQE: FAILED")
