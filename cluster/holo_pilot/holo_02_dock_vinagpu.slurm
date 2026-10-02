#!/bin/bash
#SBATCH --partition=preempt_robust
#SBATCH --account=robustelli
#SBATCH --job-name=holo_dock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=turing01,turing02
#SBATCH --array=0-999%40
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/dock_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot/logs/dock_%A_%a.err
set -uo pipefail

BASE="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot"
SIF="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/vina_gpu_v2.sif"
KM="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/benchmark_cluster/vgpu_make_ampere"
CFGROOT="${BASE}/configs"
RECROOT="${BASE}/holo_receptors"
OUTROOT="${BASE}/outputs_vinagpu/_raw"
DONEDIR="${BASE}/done"
LOGDIR="${BASE}/logs/dock"
mkdir -p "${DONEDIR}" "${LOGDIR}"

PILOT=(38 28 5 34 24 3 17 53 9 40)
N_LIG=10
FRAMES_PER_LIG=5000
N_UNITS=$(( N_LIG * FRAMES_PER_LIG ))
N_TASKS=1000

echo "Task ${SLURM_ARRAY_TASK_ID}  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}"

run_box() {
    local lig="$1" fid="$2" res_str="$3"
    local cfg="${CFGROOT}/lig${lig}/frame_${fid}/res${res_str}.txt"
    local marker="${DONEDIR}/lig${lig}_frame_${fid}_res${res_str}.done"
    local outfile="${OUTROOT}/lig${lig}/frame_${fid}_res${res_str}_out.pdbqt"
    local boxlog="${LOGDIR}/lig${lig}_frame_${fid}_res${res_str}.log"

    [ -f "${marker}" ] && return 0
    [ -f "${cfg}" ] || { echo "  MISSING cfg ${cfg}"; return 0; }

    apptainer exec --nv --pwd "${KM}" \
        --bind /etc/OpenCL/vendors:/etc/OpenCL/vendors \
        --bind /dartfs-hpc:/dartfs-hpc \
        "${SIF}" \
        "${KM}/AutoDock-Vina-GPU-2-1" --config "${cfg}" --opencl_binary_path "${KM}" \
        > "${boxlog}" 2>&1
    local rc=$?
    if [ "${rc}" -eq 0 ] && [ -s "${outfile}" ]; then
        touch "${marker}"
    else
        echo "  FAIL lig${lig} frame_${fid} res${res_str}: rc=${rc} [see ${boxlog}]"
    fi
}

u=${SLURM_ARRAY_TASK_ID}
while [ ${u} -lt ${N_UNITS} ]; do
    lig_idx=$(( u / FRAMES_PER_LIG ))
    frame_idx=$(( u % FRAMES_PER_LIG ))
    lig=${PILOT[${lig_idx}]}

    rec=$(ls -v "${RECROOT}/lig${lig}"/frame_*.pdbqt 2>/dev/null | sed -n "$(( frame_idx + 1 ))p")
    if [ -z "${rec}" ]; then
        u=$(( u + N_TASKS )); continue
    fi
    fid=$(basename "${rec}" .pdbqt | sed 's/frame_//')

    for res in $(seq 1 20); do
        res_str=$(printf "%02d" "${res}")
        run_box "${lig}" "${fid}" "${res_str}"
    done

    u=$(( u + N_TASKS ))
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
