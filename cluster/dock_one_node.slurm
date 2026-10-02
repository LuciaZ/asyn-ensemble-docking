#!/bin/bash
#SBATCH --partition=preempt_robust
#SBATCH --account=robustelli
#SBATCH --job-name=dock1n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-199%9
#SBATCH --requeue
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/logs/dock1n_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/logs/dock1n_%A_%a.err
# ============================================================================
# Single-node Vina-GPU docking. Submit with:
#
# %9 must equal the number of free GPUs on the node(sinfo -O NodeHost,Gres,GresUsed). 
# One process per card is not a preference: several Vina-GPU processes initialising the same card raise
# Err-1001:CL_PLATFORM_NOT_FOUND_KHR,which is how the earlier %40 and even %8 gap-fill runs collapsed.
#
# Three things carried over from the pilot:
#   Precompiled vgpu_make_ampere (with Kernel*_Opt.bin) avoids JIT; in single-ligand mode Vina
#     otherwise recompiles the kernel per box (~11 s vs ~1 s), and concurrent JIT also triggers CL_BUILD_PROGRAM_FAILURE.
#     The kernels are GA102-only and will not load on A100/V100/H200.
#   Batched container starts: apptainer needs ~5 s to mount the 4.1 GB sif from dartfs, far more than the ~1 s of
#     docking. BATCH_FRAMES frames share one exec. Batching only pays on wholly fresh frames, which these are.
#   Local sif copy: many tasks reading the same dartfs sif contend.
#
# One done-marker per box; re-submitting after a timeout or preemption resumes where it stopped.
# ============================================================================
set -uo pipefail

MODE="${MODE:?set MODE=apo or MODE=holo}"
BASE="${BASE:?set BASE=<project dir>}"
SIF="${SIF:-/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/vina_gpu_v2.sif}"
KM="${KM:-/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/benchmark_cluster/vgpu_make_ampere}"
BATCH_FRAMES="${BATCH_FRAMES:-10}"
N_RES=20

MANIFEST="${BASE}/manifest_${MODE}.txt"
CFGROOT="${BASE}/configs"
OUTROOT="${BASE}/outputs_vinagpu/_raw"
DONEDIR="${BASE}/done"
LOGDIR="${BASE}/logs/boxes"
mkdir -p "${DONEDIR}" "${LOGDIR}"

[ -f "${MANIFEST}" ] || { echo "no manifest at ${MANIFEST}"; exit 1; }
TOTAL=$(wc -l < "${MANIFEST}")
NTASK=$(( ${SLURM_ARRAY_TASK_COUNT:-200} ))
TID=${SLURM_ARRAY_TASK_ID}

echo "task ${TID}/${NTASK}  mode=${MODE}  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "manifest: ${TOTAL} ligand-frame units"
# Hard check: %N does not guarantee a card. Last run amp04 had only 9 free, yet SLURM started task
# 10 onward anyway; their nvidia-smi reported "No devices were found", so every box
# failed instantly -- that is where 900k silent failures came from. Exit immediately when there is no GPU.
GPUINFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)
echo "${GPUINFO}"
if ! echo "${GPUINFO}" | grep -qi "RTX A5"; then
    echo "task ${TID}: no usable GPU -- exiting before doing any work"
    exit 0
fi

LOCAL_SIF="${SIF}"
SCRATCH="${SLURM_TMPDIR:-/tmp}"
if [ -w "${SCRATCH}" ]; then
    CAND="${SCRATCH}/vina_gpu_v2_$$.sif"
    if cp "${SIF}" "${CAND}" 2>/dev/null; then
        LOCAL_SIF="${CAND}"; trap 'rm -f "${CAND}"' EXIT
        echo "sif copied to ${CAND}"
    else
        echo "local sif copy failed, reading from dartfs"
    fi
fi

run_batch() {
    [ "$#" -eq 0 ] && return 0
    local cmd="" c
    for c in "$@"; do
        cmd+="${KM}/AutoDock-Vina-GPU-2-1 --config ${c} --opencl_binary_path ${KM} >/dev/null 2>&1; "
    done
    apptainer exec --nv --pwd "${KM}" \
        --bind /etc/OpenCL/vendors:/etc/OpenCL/vendors \
        --bind /dartfs-hpc:/dartfs-hpc \
        "${LOCAL_SIF}" bash -c "${cmd}" \
        >> "${LOGDIR}/task_${TID}.log" 2>&1
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "  apptainer exited ${rc}"
    fi
    for c in "$@"; do
        local res fid lig outf mk
        res=$(basename "${c}" .txt); res="${res#res}"
        fid=$(basename "$(dirname "${c}")"); fid="${fid#frame_}"
        lig=$(basename "$(dirname "$(dirname "${c}")")"); lig="${lig#lig}"
        outf="${OUTROOT}/lig${lig}/frame_${fid}_res${res}_out.pdbqt"
        mk="${DONEDIR}/lig${lig}_frame_${fid}_res${res}.done"
        if [ -s "${outf}" ]; then
            touch "${mk}"; CONSEC_FAIL=0
        else
            echo "  FAIL lig${lig} frame_${fid} res${res}"
            CONSEC_FAIL=$(( CONSEC_FAIL + 1 ))
        fi
    done
    # Circuit breaker: a wholly failed batch means a broken environment (no card, corrupt sif, wrong kernel arch);
    # continuing would only pile up useless failures
    if [ "${CONSEC_FAIL}" -ge 60 ]; then
        echo "task ${TID}: ${CONSEC_FAIL} consecutive failures -- aborting"
        exit 1
    fi
}

n_done=0; n_skip=0; batch=(); CONSEC_FAIL=0
start=$(date +%s)
for (( line = TID + 1; line <= TOTAL; line += NTASK )); do
    read -r lig fid <<< "$(sed -n "${line}p" "${MANIFEST}")"
    [ -z "${lig:-}" ] && continue
    for res in $(seq -w 1 ${N_RES}); do
        mk="${DONEDIR}/lig${lig}_frame_${fid}_res${res}.done"
        [ -f "${mk}" ] && { n_skip=$((n_skip+1)); continue; }
        cfg="${CFGROOT}/lig${lig}/frame_${fid}/res${res}.txt"
        [ -f "${cfg}" ] || continue
        batch+=("${cfg}")
    done
    if [ "${#batch[@]}" -ge $(( BATCH_FRAMES * N_RES )) ]; then
        run_batch "${batch[@]}"
        n_done=$(( n_done + ${#batch[@]} )); batch=()
        el=$(( $(date +%s) - start ))
        [ "${el}" -gt 0 ] && echo "  ${n_done} boxes, ${n_skip} skipped, $(( n_done * 3600 / el ))/hr"
    fi
done
run_batch "${batch[@]}"
n_done=$(( n_done + ${#batch[@]} ))
el=$(( $(date +%s) - start ))
echo "task ${TID} finished: ${n_done} boxes docked, ${n_skip} already done, ${el}s"
