#!/bin/bash
#SBATCH --job-name=gpu-30q
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=gpu_30q_%j.log
#SBATCH --error=gpu_30q_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python scripts/test_gpu_30q.py
