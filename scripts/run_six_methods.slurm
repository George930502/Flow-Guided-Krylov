#!/bin/bash
#SBATCH --job-name=six-methods
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=six_methods_%j.log
#SBATCH --error=six_methods_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

# Start with Tier 1 (H2, LiH, H2O, BeH2) to validate all 6 methods
python scripts/run_six_methods.py --tier 1

echo "=== Done: $(date) ==="
