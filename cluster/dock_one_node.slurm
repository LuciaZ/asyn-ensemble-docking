#!/bin/bash
#SBATCH --partition=preempt_robust
#SBATCH --account=robustelli
#SBATCH --job-name=dock1n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-8
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/logs/dock1n_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/logs/dock1n_%A_%a.err
# ============================================================================
# dock_one_node.slurm -- Vina-GPU docking confined to one node
#
#   sbatch --export=ALL,MODE=apo,BASE=<project dir> --nodelist=<node> \
#          --array=0-<N-1> dock_one_node.slurm
#
# WORK DISTRIBUTION. Every task walks the whole manifest and claims a frame
# with an atomic mkdir before docking it. Earlier versions split the manifest
# into fixed stripes by task id; a task that started without a usable GPU
# exited and its stripe -- a tenth of the work -- was never docked by anyone,
# which cost a full five-day round. With claiming, any number of tasks may
# drop out and the rest still finish everything. Claims live under
# claims/<CLAIMS_TAG>/<lig>/<frame>; the tag defaults to the array job id, so
# a later job never inherits the stale claims of one that was killed. Give
# two concurrent jobs the same CLAIMS_TAG to make them share the work.
#
# Keep the array no larger than the node's usable GPUs. Several Vina-GPU
# processes on one card fail with CL_PLATFORM_NOT_FOUND_KHR.
#
# Carried over from the pilot:
#   precompiled kernels -- vgpu_make_ampere holds GA102 binaries (A5000,
#     A5500). Without them single-ligand mode JIT-compiles per box (~11 s vs
#     ~1 s). They do not load on A100/V100/H200; point KM elsewhere for those.
#   batched container starts -- apptainer needs ~5 s to mount the sif, so
#     BATCH_FRAMES frames share one exec.
#   local sif copy -- many tasks reading one dartfs sif contend.
#   done-markers -- one per box; re-submission resumes.
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
TID=${SLURM_ARRAY_TASK_ID:-0}
NTASK=${SLURM_ARRAY_TASK_COUNT:-1}
CLAIMS="${BASE}/claims/${CLAIMS_TAG:-${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "${DONEDIR}" "${LOGDIR}" "${CLAIMS}"

[ -f "${MANIFEST}" ] || { echo "no manifest at ${MANIFEST}"; exit 1; }
TOTAL=$(wc -l < "${MANIFEST}")
echo "task ${TID}/${NTASK}  mode=${MODE}  host=$(hostname)  claims=${CLAIMS}"
echo "manifest: ${TOTAL} ligand-frame units"

# A task without a usable card exits without touching any work; under
# claiming that no longer leaves anything undone.
GPUINFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)
echo "${GPUINFO}"
if [ -z "${GPUINFO}" ] || echo "${GPUINFO}" | grep -qiE "no devices|failed|error"; then
    echo "task ${TID}: no usable GPU -- exiting, other tasks will cover its work"
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

CONSEC_FAIL=0
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
        >> "${LOGDIR}/task_${SLURM_ARRAY_JOB_ID:-x}_${TID}.log" 2>&1
    local rc=$?
    [ "${rc}" -ne 0 ] && echo "  apptainer exited ${rc}"
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
    # a wholly failed stretch means a broken environment (card, sif, kernel
    # architecture); stop rather than pile up failures
    if [ "${CONSEC_FAIL}" -ge 60 ]; then
        echo "task ${TID}: ${CONSEC_FAIL} consecutive failures -- aborting"
        exit 1
    fi
}

n_done=0; n_skip=0; n_claimed=0; batch=()
start=$(date +%s)
# tasks start at different offsets so they do not all contend for line 1
offset=$(( TID * TOTAL / (NTASK > 0 ? NTASK : 1) ))

while read -r lig fid; do
    [ -z "${lig:-}" ] && continue
    todo=()
    for res in $(seq -w 1 ${N_RES}); do
        if [ -f "${DONEDIR}/lig${lig}_frame_${fid}_res${res}.done" ]; then
            n_skip=$((n_skip + 1)); continue
        fi
        cfg="${CFGROOT}/lig${lig}/frame_${fid}/res${res}.txt"
        [ -f "${cfg}" ] && todo+=("${cfg}")
    done
    [ "${#todo[@]}" -eq 0 ] && continue
    mkdir -p "${CLAIMS}/${lig}"
    mkdir "${CLAIMS}/${lig}/${fid}" 2>/dev/null || continue     # someone else has it
    n_claimed=$((n_claimed + 1))
    batch+=("${todo[@]}")
    if [ "${#batch[@]}" -ge $(( BATCH_FRAMES * N_RES )) ]; then
        run_batch "${batch[@]}"
        n_done=$(( n_done + ${#batch[@]} )); batch=()
        el=$(( $(date +%s) - start ))
        [ "${el}" -gt 0 ] && echo "  ${n_done} boxes, ${n_skip} skipped, ${n_claimed} frames claimed, $(( n_done * 3600 / el ))/hr"
    fi
done < <(tail -n +"$(( offset + 1 ))" "${MANIFEST}"; head -n "${offset}" "${MANIFEST}")

run_batch "${batch[@]}"
n_done=$(( n_done + ${#batch[@]} ))
el=$(( $(date +%s) - start ))
echo "task ${TID} finished: ${n_done} boxes docked, ${n_skip} already done, ${n_claimed} frames claimed, ${el}s"
