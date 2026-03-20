#!/bin/bash
#SBATCH --job-name=stability
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=stability_%j.log
#SBATCH --error=stability_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python -c "
import sys, json, numpy as np
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from src.methods.hi_vqe import run_hi_vqe, HIVQEConfig

molecules = ['LiH', 'H2O', 'BeH2']
n_seeds = 5
results = []

for mol_name in molecules:
    H, info = get_molecule(mol_name)
    nq = info['n_qubits']

    # FCI reference
    fci = FCISolver().solve(H, info)
    fci_e = fci.energy
    print(f'\n{\"=\"*80}')
    print(f'{mol_name} ({nq}Q), FCI = {fci_e:.10f} Ha')
    print(f'{\"=\"*80}')

    # HI-VQE (quantum) x n_seeds
    print(f'\n  HI-VQE (quantum):')
    hivqe_energies = []
    hivqe_times = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        cfg = HIVQEConfig(shots=5000, max_iterations=20, configuration_recovery=False)
        r = run_hi_vqe(H, info, config=cfg)
        err = (r.energy - fci_e) * 1000 if r.energy else float('inf')
        hivqe_energies.append(r.energy)
        hivqe_times.append(r.wall_time)
        print(f'    seed {seed}: E={r.energy:.10f}, err={err:.3f} mHa, t={r.wall_time:.1f}s')

    hivqe_errors = [(e - fci_e)*1000 for e in hivqe_energies if e]
    print(f'    Mean error: {np.mean(hivqe_errors):.3f} ± {np.std(hivqe_errors):.3f} mHa')
    print(f'    Best error: {np.min(hivqe_errors):.3f} mHa')

    # HI+NQS+SQD x n_seeds
    print(f'\n  HI+NQS+SQD (ours):')
    nqs_energies = []
    nqs_times = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        import torch; torch.manual_seed(seed * 42 + 7)
        n_samp = 3000 if nq <= 14 else 5000
        cfg = HINQSSQDConfig(n_samples=n_samp, max_iterations=30, max_basis_size=5000)
        r = run_hi_nqs_sqd(H, info, config=cfg)
        err = (r.energy - fci_e) * 1000 if r.energy else float('inf')
        nqs_energies.append(r.energy)
        nqs_times.append(r.wall_time)
        print(f'    seed {seed}: E={r.energy:.10f}, err={err:.3f} mHa, t={r.wall_time:.1f}s, basis={r.diag_dim}')

    nqs_errors = [(e - fci_e)*1000 for e in nqs_energies if e]
    print(f'    Mean error: {np.mean(nqs_errors):.3f} ± {np.std(nqs_errors):.3f} mHa')
    print(f'    Best error: {np.min(nqs_errors):.3f} mHa')

    results.append({
        'molecule': mol_name, 'n_qubits': nq, 'fci': fci_e,
        'hivqe_errors': hivqe_errors, 'hivqe_mean': np.mean(hivqe_errors), 'hivqe_std': np.std(hivqe_errors),
        'nqs_errors': nqs_errors, 'nqs_mean': np.mean(nqs_errors), 'nqs_std': np.std(nqs_errors),
        'hivqe_times': hivqe_times, 'nqs_times': nqs_times,
    })

# Summary
print(f'\n\n{\"=\"*80}')
print('SUMMARY (5 seeds each)')
print(f'{\"Molecule\":<10} {\"Method\":<16} {\"Mean err (mHa)\":>16} {\"Std (mHa)\":>12} {\"Best (mHa)\":>12} {\"Mean time\":>10}')
print(f'{\"=\"*80}')
for r in results:
    print(f'{r[\"molecule\"]:<10} {\"HI-VQE\":<16} {r[\"hivqe_mean\"]:>16.3f} {r[\"hivqe_std\"]:>12.3f} {min(r[\"hivqe_errors\"]):>12.3f} {np.mean(r[\"hivqe_times\"]):>10.1f}')
    print(f'{\"\":<10} {\"HI+NQS+SQD\":<16} {r[\"nqs_mean\"]:>16.3f} {r[\"nqs_std\"]:>12.3f} {min(r[\"nqs_errors\"]):>12.3f} {np.mean(r[\"nqs_times\"]):>10.1f}')

with open('results/stability_test.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
"
