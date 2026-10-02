#!/usr/bin/env bash
#SBATCH --job-name=vbench
#SBATCH --account=robustelli
#SBATCH --partition=preempt_robust
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/vbench_%A_%a.out
set -euo pipefail; set +o pipefail
cd "$SLURM_SUBMIT_DIR"
: "${NODETYPE:?NODETYPE must be passed via --export}"
JOBLIST="$SLURM_SUBMIT_DIR/joblist_${NODETYPE}_gpu.tsv"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$JOBLIST")
[ -z "$LINE" ] && { echo "no job for task $SLURM_ARRAY_TASK_ID"; exit 0; }
IFS=$'\t' read -r TOOL NCPU PIN REP <<< "$LINE"
DONE="$SLURM_SUBMIT_DIR/done"; mkdir -p "$DONE" logs
MARK="$DONE/${TOOL}_${NODETYPE}_$(hostname -s)_ncpu${NCPU}_pin${PIN}_rep${REP}.done"
[ -f "$MARK" ] && { echo "SKIP done: $MARK"; exit 0; }
echo "task $SLURM_ARRAY_TASK_ID: $TOOL ncpu=$NCPU pin=$PIN rep=$REP on $(hostname -s)"
srun bash "$SLURM_SUBMIT_DIR/cluster_bench.sh" "$TOOL" 100 "$NCPU" "$PIN" "$REP"
touch "$MARK"
