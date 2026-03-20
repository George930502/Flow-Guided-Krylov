#!/bin/bash
#SBATCH --job-name=gpu-speed
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=gpu_speed_%j.log
#SBATCH --error=gpu_speed_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python -c "
import sys, numpy as np, torch, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA available: {torch.cuda.is_available()}')

molecules = [
    ('LiH',  12, 5000),
    ('H2O',  14, 5000),
    ('BeH2', 14, 10000),
    ('NH3',  16, 20000),
    ('CH4',  18, 20000),
    ('N2',   20, 20000),
]

for mol_name, nq, n_samp in molecules:
    H, info = get_molecule(mol_name)
    fci_r = FCISolver().solve(H, info)
    fci_e = fci_r.energy

    print(f'\n{\"=\"*60}')
    print(f'{mol_name} ({nq}Q), FCI={fci_e:.10f}')

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samp,
        max_iterations=30,
        max_basis_size=min(15000, n_samp * 2),
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)

    if r.energy and fci_e:
        err = (r.energy - fci_e) * 1000
        print(f'  Result: E={r.energy:.10f}, err={err:.3f} mHa, '
              f'basis={r.diag_dim}, time={r.wall_time:.1f}s')
    else:
        print(f'  Result: E={r.energy}, time={r.wall_time:.1f}s')

print('\nDone!')
"
