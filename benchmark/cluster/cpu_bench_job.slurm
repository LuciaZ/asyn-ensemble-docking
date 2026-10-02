#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH -t 01:00:00
#SBATCH -o logs/cpubench_%j.out
cd "$SLURM_SUBMIT_DIR"
export SLURM_SUBMIT_DIR=$PWD; source cluster_config.sh
export SLURM_TMPDIR=/scratch/f007ydc/cpuconc_$SLURM_JOB_ID
for combo in "8 8" "16 4" "4 16" "32 2"; do
  set -- $combo
  bash conc_scan_cpu.sh $1 $2 20
done
