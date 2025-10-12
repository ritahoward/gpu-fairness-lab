#!/bin/bash
#SBATCH --job-name=FAIR_client
#SBATCH --output=/home/yujung/gpu-fairness-lab/logs/FAIR_client_%j.out
#SBATCH --error=/home/yujung/gpu-fairness-lab/logs/FAIR_client_%j.err
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@0

# --- 信號轉發（給舊版 Slurm）---
trap 'echo "[trap] Forwarding SIGUSR1 manually"; pkill -USR1 -P $$' USR1

# --- Python 路徑修正 ---
export PYTHONPATH=/home/yujung/.local/lib/python3.8/site-packages:

# --- FAIR 系統環境變數 ---
export FAIR_USER_ID=$USER
export FAIR_JOB_ID=$SLURM_JOB_ID
export FAIR_RUN_DIR=/home/yujung/fair_runs/$SLURM_JOB_ID
export FAIR_CKPT_PATH=$FAIR_RUN_DIR/ckpt.pt
mkdir -p "$FAIR_RUN_DIR"

cd /home/yujung/gpu-fairness-lab

# --- 執行使用者指定的 Python 程式 ---
if [ $# -eq 0 ]; then
    echo "[WARN] No Python script specified. Defaulting to client/train_gpu_minimal.py"
    SCRIPT="client/train_gpu_minimal.py"
else
    SCRIPT=$1
    shift
fi

echo "[INFO] Running $SCRIPT $@"
python3 "$SCRIPT" "$@"

