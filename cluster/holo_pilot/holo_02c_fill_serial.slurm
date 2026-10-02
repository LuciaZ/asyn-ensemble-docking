#!/bin/bash
#SBATCH --partition=preempt_robust
#SBATCH --account=robustelli
#SBATCH --job-name=holo_fills
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --nodelist=amp01,centurion01,centurion03,centurion04
#SBATCH --exclusive
#SBATCH --array=0-3
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/fills_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/fills_%A_%a.err
# ============================================================================
# Serial-safe fill: 4 tasks (one per healthy node), each single-process on its
# own GPU (avoids the OpenCL concurrency crash), batching many boxes per
# apptainer start. 106001 missing / 4 tasks. Precompiled binary.
# ============================================================================
set -uo pipefail

BASE="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot"
SIF="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/vina_gpu_v2.sif"
KM="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/benchmark_cluster/vgpu_make_ampere"
LIST="${BASE}/missing_configs.txt"
DONEDIR="${BASE}/done"
OUTROOT="${BASE}/outputs_vinagpu/_raw"
LOGDIR="${BASE}/logs/dock"
mkdir -p "${DONEDIR}" "${LOGDIR}"

TOTAL=$(wc -l < "${LIST}")
N_TASKS=4
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))

echo "Task ${SLURM_ARRAY_TASK_ID}: lines ${START}-${END} of ${TOTAL}  host=$(hostname)"

# gather this task's configs (skip already-done), build one batched command
mapfile -t CFGS < <(sed -n "${START},${END}p" "${LIST}")

BATCH=50   # boxes per apptainer start
buf=()
flush() {
    [ ${#buf[@]} -eq 0 ] && return
    local cmd=""
    for c in "${buf[@]}"; do
        cmd+="${KM}/AutoDock-Vina-GPU-2-1 --config ${c} --opencl_binary_path ${KM} >/dev/null 2>&1; "
    done
    apptainer exec --nv --pwd "${KM}" \
        --bind /etc/OpenCL/vendors:/etc/OpenCL/vendors \
        --bind /dartfs-hpc:/dartfs-hpc \
        "${SIF}" bash -c "${cmd}"
    # mark done for those whose output exists
    for c in "${buf[@]}"; do
        res_str=$(basename "${c}" .txt | sed 's/res//')
        fid=$(basename "$(dirname "${c}")" | sed 's/frame_//')
        lig=$(basename "$(dirname "$(dirname "${c}")")" | sed 's/lig//')
        outfile="${OUTROOT}/lig${lig}/frame_${fid}_res${res_str}_out.pdbqt"
        marker="${DONEDIR}/lig${lig}_frame_${fid}_res${res_str}.done"
        [ -s "${outfile}" ] && touch "${marker}"
    done
    buf=()
}

n=0
for cfg in "${CFGS[@]}"; do
    [ -z "${cfg}" ] && continue
    res_str=$(basename "${cfg}" .txt | sed 's/res//')
    fid=$(basename "$(dirname "${cfg}")" | sed 's/frame_//')
    lig=$(basename "$(dirname "$(dirname "${cfg}")")" | sed 's/lig//')
    marker="${DONEDIR}/lig${lig}_frame_${fid}_res${res_str}.done"
    [ -f "${marker}" ] && continue
    buf+=("${cfg}")
    n=$(( n + 1 ))
    if [ ${#buf[@]} -ge ${BATCH} ]; then
        flush
        echo "  ...task ${SLURM_ARRAY_TASK_ID}: ${n} processed" 
    fi
done
flush
echo "Task ${SLURM_ARRAY_TASK_ID} finished: ${n} boxes."
