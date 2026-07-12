#!/bin/bash
#SBATCH -J cnn-anomaly
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH -t 1:00:00
#SBATCH -o logs/cnn-%j.out
#SBATCH -e logs/cnn-%j.err
#SBATCH --mail-type=END,FAIL

# Run from the repo root on the cluster: sbatch scripts/run_cnn.slurm
# Standalone CNN autoencoder (no fusion). Writes outputs/cnn/metrics.json,
# outputs/cnn/predictions.csv (diagnostic) and outputs/cnn_predictions.csv
# (portal submission: run_id/timestep/prediction).

mkdir -p logs

module load nvidia/latest

source ts-anomaly-env/bin/activate

python -u -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# -u: unbuffered stdout, so `tail -f` on the log file shows live progress.
python -u -m src.pipelines.run_cnn
