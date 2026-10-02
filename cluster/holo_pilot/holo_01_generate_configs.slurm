#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=robustelli
#SBATCH --job-name=holo_gencfg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/gencfg_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/gencfg_%A_%a.err
# ============================================================================
# One task per pilot ligand generates that ligand's config tree
# (5000 frames x 20 boxes = 100k configs per ligand; ~1M total).
# ============================================================================
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PILOT=(38 28 5 34 24 3 17 53 9 40)
LIG=${PILOT[$SLURM_ARRAY_TASK_ID]}

echo "Task ${SLURM_ARRAY_TASK_ID}: ligand ${LIG}  host=$(hostname)"
python holo_01_generate_configs.py "${LIG}"
echo "Task ${SLURM_ARRAY_TASK_ID} (lig ${LIG}) done."
