#!/bin/bash
#SBATCH -J hybrid-anomaly
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH -t 1:00:00
#SBATCH -o logs/hybrid-%j.out
#SBATCH -e logs/hybrid-%j.err
#SBATCH --mail-type=END,FAIL

# Run from the repo root on the cluster: sbatch scripts/run_hybrid.slurm
# Adjust --cpus-per-task / -t to taste; -t is a hard wall-clock limit, so
# leave headroom over your usual local run time.

mkdir -p logs

module load nvidia/latest

source ts-anomaly-env/bin/activate

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -m src.pipelines.run_hybrid
