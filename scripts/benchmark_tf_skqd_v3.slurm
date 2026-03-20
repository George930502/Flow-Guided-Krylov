#!/bin/bash
#SBATCH --job-name=tf-skqd-v3
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=benchmark_tf_skqd_v3_%j.log
#SBATCH --error=benchmark_tf_skqd_v3_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python -c "
import sys, time
sys.path.insert(0, '.')
from src.solvers.nf_skqd import NFSKQDSolver, NFSKQDConfig
from src.nqs.transformer import AutoregressiveTransformer
from src.molecules import get_molecule

molecules = [
    ('N2', 20),
    ('N2-CAS(10,20)', 40),
    ('N2-CAS(10,26)', 52),
]

for mol_name, nq in molecules:
    print(f'\n{\"=\"*80}')
    print(f'{mol_name} ({nq}Q)')

    t0 = time.time()
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    print(f'  Hamiltonian built in {time.time()-t0:.1f}s')

    if nq <= 20:
        embed, heads, layers = 64, 4, 3
        n_powers, n_samples, nf_steps = 20, 2000, 5
    elif nq <= 40:
        embed, heads, layers = 128, 8, 4
        n_powers, n_samples, nf_steps = 20, 3000, 5
    else:
        embed, heads, layers = 192, 8, 6
        n_powers, n_samples, nf_steps = 20, 5000, 5

    tf = AutoregressiveTransformer(
        n_orbitals=n_orb, n_alpha=H.n_alpha, n_beta=H.n_beta,
        embed_dim=embed, n_heads=heads, n_layers=layers,
    )
    print(f'  Transformer: {sum(p.numel() for p in tf.parameters()):,} params')

    # v3: T=1.0→0.3, no warmup, very few NF steps (5), more powers (20)
    config = NFSKQDConfig(
        n_krylov_powers=n_powers,
        n_samples_per_power=n_samples,
        nf_steps_per_power=nf_steps,    # only 5 steps = gentle nudge
        max_basis_size=15000,
        initial_temperature=1.0,         # not too random
        final_temperature=0.3,           # focus at end
        warmup_powers=0,                 # learn from the start
        entropy_weight=0.05,
        convergence_threshold=1e-6,
    )

    solver = NFSKQDSolver(tf, config)
    result = solver.solve(H, info)

    print(f'\n  E={result.energy:.10f} Ha, basis={result.diag_dim}, '
          f'powers={result.metadata[\"n_krylov_powers\"]}, time={result.wall_time:.1f}s')
    print(f'  samples/power: {result.metadata[\"samples_per_power\"]}')
"
