#!/usr/bin/env python3
"""Quick test for CUDA-Q sampler + IBM SQD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.samplers.cudaq_sampler import CUDAQCircuitSampler, CUDAQSamplerConfig
from src.methods.qc_sqd import run_qc_sqd
from src.methods.hi_vqe import run_hi_vqe, HIVQEConfig

H, info = get_molecule("LiH")
fci_e = -7.8823243789

print("=== CUDA-Q EPA Sampler Test ===")
sampler = CUDAQCircuitSampler(H, CUDAQSamplerConfig(ansatz="epa", n_layers=2, shots=5000))
print(f"  n_params = {sampler.n_params}")
r = sampler.sample(5000)
print(f"  Sampled {len(r.configs)} unique configs in {r.wall_time:.2f}s")

print("\n=== QC+SQD ===")
r = run_qc_sqd(H, info, n_samples=5000)
print(f"  E = {r.energy:.10f}, err = {(r.energy - fci_e)*1000:.3f} mHa" if r.energy else "  FAILED")

print("\n=== HI-VQE (5 iters) ===")
cfg = HIVQEConfig(shots=5000, max_iterations=5, ansatz="epa", n_layers=2, configuration_recovery=False)
r = run_hi_vqe(H, info, config=cfg)
print(f"  E = {r.energy:.10f}, err = {(r.energy - fci_e)*1000:.3f} mHa" if r.energy else "  FAILED")
