#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=robustelli
#SBATCH --job-name=gencfg_apo50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-20
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026/logs/gencfg_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026/logs/gencfg_%A_%a.err
# ============================================================================
# Generate the config tree in parallel. 21 tasks each own a 1000-frame slab of
# the ORIGINAL 0..20019 index range; the .py applies stride-4 internally so
# each task actually writes configs for ~250 stride-selected frames.
# CPU-only, standard partition (non-preempt) so this never gets killed midway.
# ============================================================================
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SLAB=1000
FSTART=$(( SLURM_ARRAY_TASK_ID * SLAB ))
FEND=$(( FSTART + SLAB ))

echo "Task ${SLURM_ARRAY_TASK_ID}: frames [${FSTART}, ${FEND})  host=$(hostname)"
python 01_generate_configs.py "${FSTART}" "${FEND}"
echo "Task ${SLURM_ARRAY_TASK_ID} done."
