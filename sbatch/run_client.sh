#!/bin/bash
#SBATCH --job-name=FAIR_client
#SBATCH --output=/home/yujung/gpu-fairness-lab/logs/%x_%j.out
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@0
set -eux

# ✅ 加上 user site-packages 路徑，若沒定義 PYTHONPATH 則預設空字串
export PYTHONPATH="$HOME/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"

echo "=== Job environment check ==="
which python3
python3 --version
python3 -c "import numpy, torch; print('OK numpy', numpy.__version__, 'torch', torch.__version__)"

export FAIR_USER_ID="${SLURM_JOB_USER}"
export FAIR_JOB_ID="${SLURM_JOB_ID}"
export FAIR_RUN_DIR="$HOME/fair_runs/${SLURM_JOB_ID}"
mkdir -p "$FAIR_RUN_DIR"
export FAIR_CKPT_PATH="$FAIR_RUN_DIR/ckpt.pt"

cd ~/gpu-fairness-lab
python3 client/train_gpu_minimal.py

