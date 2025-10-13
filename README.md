instructions:
python3 -m runners.runner_slurm --slice-sec 5 --warn-sec 3 --grace-sec 5
sbatch sbatch/run_client.sh
scancel -u $USER
watch -n 1 squeue
