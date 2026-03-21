#!/bin/bash
#SBATCH --job-name=beh2-stab
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=beh2_stability_%j.log
#SBATCH --error=beh2_stability_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python -c "
import sys, numpy as np, torch
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

H, info = get_molecule('BeH2')
fci_e = FCISolver().solve(H, info).energy
print(f'BeH2 (14Q), FCI = {fci_e:.10f} Ha\n')

sample_sizes = [3000, 5000, 10000]
n_seeds = 5

for n_samp in sample_sizes:
    print(f'\n--- n_samples = {n_samp} ---')
    errors = []
    times = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        torch.manual_seed(seed * 42 + 7)
        cfg = HINQSSQDConfig(n_samples=n_samp, max_iterations=30, max_basis_size=5000)
        r = run_hi_nqs_sqd(H, info, config=cfg)
        err = (r.energy - fci_e) * 1000
        errors.append(err)
        times.append(r.wall_time)
        print(f'  seed {seed}: err={err:.3f} mHa, basis={r.diag_dim}, t={r.wall_time:.1f}s')

    print(f'  Mean: {np.mean(errors):.3f} ± {np.std(errors):.3f} mHa')
    print(f'  Best: {np.min(errors):.3f} mHa')
    print(f'  Mean time: {np.mean(times):.1f}s')
"
