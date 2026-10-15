#!/bin/bash
#SBATCH --partition=preempt_robust
#SBATCH --account=robustelli
#SBATCH --job-name=dock_apo50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=turing01,turing02
#SBATCH --array=0-500%10
#SBATCH --mail-user=zhihan.zhang.gr@dartmouth.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026/logs/dock_%A_%a.out
#SBATCH -e /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026/logs/dock_%A_%a.err
# ============================================================================
# Scheme A Vina-GPU docking for the 10-ligand apo pilot.  (ROUTE 1: JIT .sif)
#
#   * stride-4 -> 5005 frames; FRAMES_PER_TASK=10 -> 501 array tasks (0..500)
#   * each task loops its 10 frames; each frame runs 20 residue-box configs;
#     each config docks ALL 10 ligands in one ligand_directory call
#   * turing excluded (precompiled binary won't load on 2080 Ti / driver 580)
#   * per-box done-markers -> preemption-resilient reruns
#   * %40 throttle; raise live via
#       scontrol update arraytaskthrottle=<N> jobid=<JID>
#
#   CHANGES vs draft (all three required for the JIT .sif to work at scale):
#     (1) --pwd VGPU_HOME  -> the dir containing OpenCL/ so JIT finds kernels
#     (2) per-box stdout redirected to its own log (Vina-GPU is very verbose;
#         501 tasks x 200 boxes into the SLURM log would blow it up)
#     (3) --time 08:00 -> 24:00  (JIT recompiles per box, ~13.5s/box incl JIT)
# ============================================================================
set -uo pipefail

BASE="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026"
SIF="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/vina_gpu_v2.sif"
# JIT kernels resolve relative to --pwd; this dir contains OpenCL/  (CHANGE 1)
VGPU_HOME="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1"
CFGROOT="${BASE}/configs"
OUTROOT="${BASE}/outputs_vinagpu"
DONEDIR="${BASE}/done"
LOGDIR="${BASE}/logs/dock"
FRAMES_PER_TASK=10
FRAME_STRIDE=4
N_FRAMES_TOTAL=20020

mkdir -p "${DONEDIR}" "${LOGDIR}"

# Build the ordered list of stride-selected frame ids, then index into it.
# The k-th selected frame has original id = k * FRAME_STRIDE.
K_START=$(( SLURM_ARRAY_TASK_ID * FRAMES_PER_TASK ))
K_END=$(( K_START + FRAMES_PER_TASK - 1 ))
N_SELECTED=$(( (N_FRAMES_TOTAL + FRAME_STRIDE - 1) / FRAME_STRIDE ))   # 5005
[ ${K_END} -ge ${N_SELECTED} ] && K_END=$(( N_SELECTED - 1 ))

echo "Task ${SLURM_ARRAY_TASK_ID}: selected-frame idx [${K_START}..${K_END}]  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}"

run_box() {
    local fid_str="$1" res_str="$2"
    local cfg="${CFGROOT}/frame_${fid_str}/res${res_str}.txt"
    local marker="${DONEDIR}/frame_${fid_str}_res${res_str}.done"
    local rawdir="${OUTROOT}/_raw/frame_${fid_str}_res${res_str}"
    local boxlog="${LOGDIR}/frame_${fid_str}_res${res_str}.log"

    [ -f "${marker}" ] && return 0
    if [ ! -f "${cfg}" ]; then
        echo "  MISSING cfg ${cfg}" ; return 0
    fi
    mkdir -p "${rawdir}"

    # CHANGE 1: --pwd VGPU_HOME  |  CHANGE 2: per-box log redirect
    apptainer run --nv \
        --bind /etc/OpenCL/vendors:/etc/OpenCL/vendors \
        --bind /dartfs-hpc:/dartfs-hpc \
        --pwd "${VGPU_HOME}" \
        "${SIF}" --config "${cfg}" \
        > "${boxlog}" 2>&1
    local rc=$?

    # success = all 10 ligand outputs present
    local n_out
    n_out=$(find "${rawdir}" -name '*_charged_out.pdbqt' 2>/dev/null | wc -l)
    if [ "${rc}" -eq 0 ] && [ "${n_out}" -eq 10 ]; then
        touch "${marker}"
    else
        echo "  FAIL frame_${fid_str}_res${res_str}: rc=${rc} n_out=${n_out} (expected 10) [see ${boxlog}]"
    fi
}

for (( k=K_START; k<=K_END; k++ )); do
    fid=$(( k * FRAME_STRIDE ))
    fid_str=$(printf "%05d" "${fid}")
    [ -d "${CFGROOT}/frame_${fid_str}" ] || { echo "  no cfg dir for frame ${fid_str}"; continue; }
    for res in $(seq 1 20); do
        res_str=$(printf "%02d" "${res}")
        run_box "${fid_str}" "${res_str}"
    done
    echo "  frame ${fid_str} complete"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
