#!/bin/bash
#SBATCH --job-name=tf-skqd-52
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=benchmark_tf_skqd_52_%j.log
#SBATCH --error=benchmark_tf_skqd_52_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

python -c "
import sys, time
sys.path.insert(0, '.')
from src.solvers.nf_skqd import NFSKQDSolver, NFSKQDConfig
from src.nqs.transformer import AutoregressiveTransformer
from src.molecules import get_molecule

# Test molecules at increasing size
molecules = [
    ('N2', 20),
    ('N2-CAS(10,17)', 34),
    ('N2-CAS(10,20)', 40),
    ('N2-CAS(10,26)', 52),
]

for mol_name, nq in molecules:
    print(f'\n{\"=\"*80}')
    print(f'{mol_name} ({nq}Q)')
    print(f'{\"=\"*80}')

    t0 = time.time()
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    build_time = time.time() - t0
    print(f'  Hamiltonian built in {build_time:.1f}s')

    # Scale transformer and config by system size
    if nq <= 20:
        embed, heads, layers = 64, 4, 3
        n_powers, n_samples, nf_steps = 10, 1000, 15
    elif nq <= 34:
        embed, heads, layers = 128, 4, 4
        n_powers, n_samples, nf_steps = 10, 2000, 20
    elif nq <= 40:
        embed, heads, layers = 128, 8, 4
        n_powers, n_samples, nf_steps = 10, 2000, 20
    else:  # 52Q
        embed, heads, layers = 192, 8, 6
        n_powers, n_samples, nf_steps = 10, 3000, 20

    tf = AutoregressiveTransformer(
        n_orbitals=n_orb, n_alpha=H.n_alpha, n_beta=H.n_beta,
        embed_dim=embed, n_heads=heads, n_layers=layers,
    )
    n_params = sum(p.numel() for p in tf.parameters())
    print(f'  Transformer: embed={embed}, heads={heads}, layers={layers}, params={n_params:,}')

    config = NFSKQDConfig(
        n_krylov_powers=n_powers,
        n_samples_per_power=n_samples,
        nf_steps_per_power=nf_steps,
        max_basis_size=10000,
        entropy_weight=0.05,
    )

    solver = NFSKQDSolver(tf, config)
    result = solver.solve(H, info)

    print(f'\n  Result: E={result.energy:.10f} Ha')
    print(f'  Basis: {result.diag_dim}, Krylov powers: {result.metadata[\"n_krylov_powers\"]}')
    print(f'  Time: {result.wall_time:.1f}s')
    print(f'  Samples/power: {result.metadata[\"samples_per_power\"]}')
    print(f'  Energy history: {[f\"{e:.6f}\" for e in result.metadata[\"energy_history\"]]}')
"

echo "=== Done: $(date) ==="
