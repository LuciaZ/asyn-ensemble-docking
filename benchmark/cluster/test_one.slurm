#!/usr/bin/env bash
#SBATCH --job-name=vtest
#SBATCH --account=robustelli
#SBATCH --partition=preempt_robust
#SBATCH --nodelist=amp01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/vtest_%j.out
set -euo pipefail; set +o pipefail
cd "$SLURM_SUBMIT_DIR"
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-unset}  TMPDIR=${TMPDIR:-unset}"
srun bash cluster_bench.sh vinagpu 1 8 off 8
