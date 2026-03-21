#!/bin/bash
#SBATCH --job-name=c2h4-v2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=c2h4_v2_%j.log
#SBATCH --error=c2h4_v2_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
python scripts/run_c2h4_v2.py
