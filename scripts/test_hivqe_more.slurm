#!/bin/bash
#SBATCH --job-name=hivqe-more
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=hivqe_more_%j.log
#SBATCH --error=hivqe_more_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
python scripts/test_hivqe_more.py
