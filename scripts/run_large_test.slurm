#!/bin/bash
#SBATCH --job-name=large-test
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=large_test_%j.log
#SBATCH --error=large_test_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python scripts/run_large_test.py
