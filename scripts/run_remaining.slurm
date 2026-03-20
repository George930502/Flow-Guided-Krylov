#!/bin/bash
#SBATCH --job-name=remaining
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=remaining_%j.log
#SBATCH --error=remaining_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
python scripts/run_remaining.py
