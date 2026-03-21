#!/bin/bash
#SBATCH --job-name=6methods-v3
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=six_methods_v3_%j.log
#SBATCH --error=six_methods_v3_%j.err

echo "=== Start: $(date) ==="
cd /home/leo07010/NQS-SQD/nqs-sqd

# Only run FCI + QC methods + HI methods (skip slow NQS two-stage for now)
python scripts/run_six_methods.py --molecules "LiH,H2O" --methods "FCI,QC+SQD,QC+SKQD,HI-VQE,HI+NQS+SQD"

echo "=== Done: $(date) ==="
