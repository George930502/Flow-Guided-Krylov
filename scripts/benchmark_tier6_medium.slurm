#!/bin/bash
#SBATCH --job-name=nqs-bench-t6m
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=benchmark_tier6m_%j.log
#SBATCH --error=benchmark_tier6m_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

python scripts/run_benchmark.py --molecules "N2-CAS(10,17),Cr2-CAS(12,18),N2-CAS(10,20),Cr2-CAS(12,20)"

echo "=== Done: $(date) ==="
