#!/bin/bash
#SBATCH --job-name=nqs-iter2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=benchmark_iter2_%j.log
#SBATCH --error=benchmark_iter2_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

# Only run baselines + iterative methods (skip NF, TF, DCI to save time)
python -c "
import sys
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.solvers.dci_skqd import DCISKQDSolverB, DCISKQDConfig
from src.solvers.iterative_nf_sqd import IterativeNFSQDSolver, IterativeNFSKQDSolver, IterativeNFSQDConfig
from src.flows.particle_conserving_flow import ParticleConservingFlowSampler
import json, time
from pathlib import Path

molecules = ['H2', 'LiH', 'H2O', 'BeH2', 'NH3', 'CH4', 'N2', 'CO']
results = []
fci_energies = {}

for mol_name in molecules:
    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    print(f'\n{\"=\"*80}')
    print(f'{mol_name} ({nq}Q)')
    print(f'  {\"Method\":<16} {\"Energy\":>18} {\"Error(mHa)\":>12} {\"Basis\":>8} {\"Time(s)\":>10}')
    print(f'  {\"-\"*70}')

    # FCI
    fci = FCISolver()
    r = fci.solve(H, info)
    if r.energy is not None:
        fci_energies[mol_name] = r.energy
    fci_e = fci_energies.get(mol_name)
    err = 0.0 if r.energy and fci_e else None
    print(f'  {\"FCI\":<16} {r.energy:.10f} {\"0.000\":>12} {r.diag_dim:>8} {r.wall_time:>10.2f}')
    results.append({'molecule': mol_name, 'method': 'FCI', 'energy': r.energy, 'error_mha': 0.0, 'diag_dim': r.diag_dim, 'wall_time': r.wall_time})

    # DCI-SKQD-B
    dci_cfg = DCISKQDConfig(max_basis_size=5000 if nq <= 14 else 10000)
    dci = DCISKQDSolverB(dci_cfg)
    r = dci.solve(H, info)
    err = (r.energy - fci_e) * 1000 if r.energy and fci_e else None
    err_s = f'{err:.3f}' if err is not None else 'N/A'
    print(f'  {\"DCI-SKQD-B\":<16} {r.energy:.10f} {err_s:>12} {r.diag_dim:>8} {r.wall_time:>10.2f}')
    results.append({'molecule': mol_name, 'method': 'DCI-SKQD-B', 'energy': r.energy, 'error_mha': err, 'diag_dim': r.diag_dim, 'wall_time': r.wall_time})

    # Iter-NF-SQD
    n_samp = 1000 if nq <= 14 else 3000 if nq <= 18 else 5000
    max_basis = 5000 if nq <= 14 else 10000
    flow = ParticleConservingFlowSampler(num_sites=H.num_sites, n_alpha=H.n_alpha, n_beta=H.n_beta, temperature=2.0)
    iter_cfg = IterativeNFSQDConfig(max_outer_iterations=15, n_samples=n_samp, nf_update_steps=30, do_expansion=False, max_basis_size=max_basis, energy_weight=0.1)
    solver = IterativeNFSQDSolver(flow, iter_cfg)
    r = solver.solve(H, info)
    err = (r.energy - fci_e) * 1000 if r.energy and fci_e else None
    err_s = f'{err:.3f}' if err is not None else 'N/A'
    print(f'  {\"Iter-NF-SQD\":<16} {r.energy:.10f} {err_s:>12} {r.diag_dim:>8} {r.wall_time:>10.2f}')
    results.append({'molecule': mol_name, 'method': 'Iter-NF-SQD', 'energy': r.energy, 'error_mha': err, 'diag_dim': r.diag_dim, 'wall_time': r.wall_time})

    # Iter-NF-SKQD
    flow2 = ParticleConservingFlowSampler(num_sites=H.num_sites, n_alpha=H.n_alpha, n_beta=H.n_beta, temperature=2.0)
    iter_cfg2 = IterativeNFSQDConfig(max_outer_iterations=15, n_samples=n_samp, nf_update_steps=30, do_expansion=True, expansion_size=500, n_expansion_configs=200, max_basis_size=max_basis, energy_weight=0.1)
    solver2 = IterativeNFSKQDSolver(flow2, iter_cfg2)
    r = solver2.solve(H, info)
    err = (r.energy - fci_e) * 1000 if r.energy and fci_e else None
    err_s = f'{err:.3f}' if err is not None else 'N/A'
    print(f'  {\"Iter-NF-SKQD\":<16} {r.energy:.10f} {err_s:>12} {r.diag_dim:>8} {r.wall_time:>10.2f}')
    results.append({'molecule': mol_name, 'method': 'Iter-NF-SKQD', 'energy': r.energy, 'error_mha': err, 'diag_dim': r.diag_dim, 'wall_time': r.wall_time})

# Save
Path('results').mkdir(exist_ok=True)
with open('results/iterative_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print('\n\n' + '='*80)
print('SUMMARY')
print(f'{\"Molecule\":<10} {\"Method\":<16} {\"Energy\":>18} {\"Error(mHa)\":>12} {\"Basis\":>8} {\"Time(s)\":>10}')
print('='*80)
for r in results:
    e = r['energy']
    e_s = f'{e:.10f}' if e else 'N/A'
    err_s = f'{r[\"error_mha\"]:.3f}' if r.get('error_mha') is not None else 'N/A'
    print(f'{r[\"molecule\"]:<10} {r[\"method\"]:<16} {e_s:>18} {err_s:>12} {r[\"diag_dim\"]:>8} {r[\"wall_time\"]:>10.2f}')
"

echo "=== Done: $(date) ==="
