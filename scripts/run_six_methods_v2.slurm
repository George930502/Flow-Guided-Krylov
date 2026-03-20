#!/bin/bash
#SBATCH --job-name=6methods-v2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=six_methods_v2_%j.log
#SBATCH --error=six_methods_v2_%j.err

echo "=== Start: $(date) ==="
cd /home/leo07010/NQS-SQD/nqs-sqd

# Test molecules: LiH(12Q), H2O(14Q), C2H2(24Q), Benzene(30Q)
python scripts/run_six_methods.py --molecules "LiH,H2O,C2H2,Benzene"

echo "=== Done: $(date) ==="
