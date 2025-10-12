#!/bin/bash
#SBATCH --job-name=FAIR_client
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --export=ALL

set -eux

echo "=== Job environment check ==="
echo "SHELL=$SHELL"
hostname
ls -ld /home/yujung/venv/bin/activate || echo "venv missing"

# ✅ 所有指令都放在同一個 bash -c 內，確保在同一個環境中
bash -c "
  set -eux
  source /home/yujung/venv/bin/activate
  echo '=== Activated venv ==='
  which python
  python --version
  echo 'PYTHONPATH(before) = '$PYTHONPATH
  export PYTHONPATH=/home/yujung/gpu-fairness-lab/hooks:\$PYTHONPATH
  echo 'PYTHONPATH(after) = '$PYTHONPATH
  export FAIR_USER_ID=\$SLURM_JOB_USER
  export FAIR_JOB_ID=\$SLURM_JOB_ID
  export FAIR_RUN_DIR=/home/yujung/fair_runs/\$SLURM_JOB_ID
  mkdir -p \$FAIR_RUN_DIR
  export FAIR_CKPT_PATH=\$FAIR_RUN_DIR/ckpt.pt
  cd /home/yujung/gpu-fairness-lab
  python client/train_gpu_minimal.py
"

