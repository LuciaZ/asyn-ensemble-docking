#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=robustelli
#SBATCH --job-name=collapse_apo
#SBATCH --array=0-36
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/logs/collapse_apo_%A_%a.out
# One ligand per task: ~100k small-file opens each, which on dartfs is
# minutes per ligand in parallel versus hours serially on the login node.
set -uo pipefail
python3 "${SLURM_SUBMIT_DIR}/collapse_apo_remaining.py" --task "${SLURM_ARRAY_TASK_ID}"
