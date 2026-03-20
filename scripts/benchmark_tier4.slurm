#!/bin/bash
#SBATCH --job-name=nqs-bench-t4
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=benchmark_tier4_%j.log
#SBATCH --error=benchmark_tier4_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

python scripts/run_benchmark.py --molecules "HCN,C2H2,N2-CAS(10,12),Cr2"

echo "=== Done: $(date) ==="
