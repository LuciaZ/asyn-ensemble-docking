#!/bin/bash
#SBATCH --partition=preempt_robust
#SBATCH --account=robustelli
#SBATCH --job-name=holo_fill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --nodelist=amp01,centurion01,centurion03,centurion04
#SBATCH --array=0-99%40
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/fill_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/fill_%A_%a.err
# ============================================================================
# Fill missing boxes from missing_configs.txt (zero-skip, no interleaved spin).
# 106001 missing configs, 100 tasks, each a contiguous chunk. Precompiled bin.
# No large-directory ls (uses the list file + single-file ops) -> unaffected
# by dartfs slowness on big dirs.
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
N_TASKS=100
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))

echo "Task ${SLURM_ARRAY_TASK_ID}: lines ${START}-${END} of ${TOTAL}  host=$(hostname)"

mapfile -t CFGS < <(sed -n "${START},${END}p" "${LIST}")

for cfg in "${CFGS[@]}"; do
    [ -z "${cfg}" ] && continue
    res_str=$(basename "${cfg}" .txt | sed 's/res//')
    fid=$(basename "$(dirname "${cfg}")" | sed 's/frame_//')
    lig=$(basename "$(dirname "$(dirname "${cfg}")")" | sed 's/lig//')

    marker="${DONEDIR}/lig${lig}_frame_${fid}_res${res_str}.done"
    outfile="${OUTROOT}/lig${lig}/frame_${fid}_res${res_str}_out.pdbqt"
    boxlog="${LOGDIR}/lig${lig}_frame_${fid}_res${res_str}.log"

    [ -f "${marker}" ] && continue

    apptainer exec --nv --pwd "${KM}" \
        --bind /etc/OpenCL/vendors:/etc/OpenCL/vendors \
        --bind /dartfs-hpc:/dartfs-hpc \
        "${SIF}" \
        "${KM}/AutoDock-Vina-GPU-2-1" --config "${cfg}" --opencl_binary_path "${KM}" \
        > "${boxlog}" 2>&1
    rc=$?
    if [ "${rc}" -eq 0 ] && [ -s "${outfile}" ]; then
        touch "${marker}"
    else
        echo "  FAIL lig${lig} frame_${fid} res${res_str}: rc=${rc}"
    fi
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
