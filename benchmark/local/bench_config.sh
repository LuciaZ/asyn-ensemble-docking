#!/usr/bin/env bash
# Central config for the local Vina / Vina-GPU benchmark on prl-desktop1.
# EVERY machine-specific path/param lives here. Edit, then run run_local_bench.sh.

# ---- work + data ----------------------------------------------------------
WORKROOT="/home/zhihanz/data/benchmark/vina_local"   # results land here
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Apo receptor frames (one pdbqt per MD frame). VERIFY THIS EXISTS LOCALLY.
# On Discovery these live at:
#   /dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/.../protein_apo_pdbqt/
# If prl-desktop1 doesn't have them, pull ~100 frames, e.g.:
#   mkdir -p "$FRAMES_DIR" && rsync -av \
#     f007ydc@discovery:/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/VinaGPU/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1/protein_apo_pdbqt/protein_only_frame_000{0,1}*_fasudil_apo.pdbqt \
#     "$FRAMES_DIR"/
FRAMES_DIR="/home/zhihanz/data/DiffDock_copy/data/protein_apo_pdbqt/protein_apo_pdbqt"
FRAME_GLOB="protein_only_frame_*_fasudil_apo.pdbqt"

# Ligand (fasudil = 41). Change LIGAND_PDBQT + EDGE together for another ligand.
LIGAND_PDBQT="/home/zhihanz/data/DiffDock_copy/data/ligand_pdbqt/ligands_pdbqt/47_charged.pdbqt"
EDGE="16.58"

# ---- search parameters (kept identical across ALL cells = equal work) ------
NUM_MODES=20
SEED=42
# Vina (CPU) search effort:
EXHAUSTIVENESS=8
N_CPU=8               # cores given to plain Vina
# Vina-GPU search effort (NOT the same knob as exhaustiveness; report
# throughput + pose quality, don't claim identical search effort):
VINAGPU_THREAD=5000

# ---- GPU ------------------------------------------------------------------
GPU_IDX=0             # RTX 3070 index (check: nvidia-smi -L)

# ---- binaries / containers ------------------------------------------------
# Plain Vina (CPU). Usually on PATH inside the `diffdock` env.
VINA_LOCAL_BIN="vina"
SIF_VINA="/home/zhihanz/data/VinaGPU/vina_cpu.sif"

# Vina-GPU local build. Point at the compiled binary; leave blank if not built yet.
VINAGPU_LOCAL_BIN=""                          # e.g. /home/zhihanz/build/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
VINAGPU_OPENCL_PATH=""                        # dir with Kernel2_Opt.bin etc; pass to --opencl_binary_path

# Vina-GPU container (you already have this one).
SIF_VINAGPU="/home/zhihanz/data/VinaGPU/vinasif/vina_gpu_v2.sif"
VINAGPU_BIN_IN_SIF="AutoDock-Vina-GPU-2-1"    # binary name on PATH inside the sif
VINAGPU_OPENCL_PATH_SIF="/home/zhihanz/data/benchmark/vgpu_home/AutoDock-Vina-GPU-2.1"

# ===========================================================================
# Dock functions: one per (tool, deploy). Each takes:
#   $1 receptor  $2 ligand  $3 cx  $4 cy  $5 cz  $6 edge  $7 out
# Edit the ONE that matches your setup so it mirrors your proven working command.
# ===========================================================================
dock_vina_local() {
  "$VINA_LOCAL_BIN" \
    --receptor "$1" --ligand "$2" \
    --center_x "$3" --center_y "$4" --center_z "$5" \
    --size_x "$6" --size_y "$6" --size_z "$6" \
    --exhaustiveness "$EXHAUSTIVENESS" --num_modes "$NUM_MODES" \
    --seed "$SEED" --cpu "$N_CPU" \
    --out "$7"
}

dock_vina_container() {
  apptainer exec -B "$WORKROOT","$FRAMES_DIR" "$SIF_VINA" \
    vina \
    --receptor "$1" --ligand "$2" \
    --center_x "$3" --center_y "$4" --center_z "$5" \
    --size_x "$6" --size_y "$6" --size_z "$6" \
    --exhaustiveness "$EXHAUSTIVENESS" --num_modes "$NUM_MODES" \
    --seed "$SEED" --cpu "$N_CPU" \
    --out "$7"
}

dock_vinagpu_local() {
  local extra=""; [ -n "$VINAGPU_OPENCL_PATH" ] && extra="--opencl_binary_path $VINAGPU_OPENCL_PATH"
  "$VINAGPU_LOCAL_BIN" \
    --receptor "$1" --ligand "$2" \
    --center_x "$3" --center_y "$4" --center_z "$5" \
    --size_x "$6" --size_y "$6" --size_z "$6" \
    --thread "$VINAGPU_THREAD" --num_modes "$NUM_MODES" \
    $extra --out "$7"
}


dock_vinagpu_container() {
  apptainer exec --nv \
    --pwd "$VINAGPU_OPENCL_PATH_SIF" \
    -B /etc/OpenCL/vendors:/etc/OpenCL/vendors \
    -B "$WORKROOT","$FRAMES_DIR","$VINAGPU_OPENCL_PATH_SIF" \
    "$SIF_VINAGPU" \
    "$VINAGPU_BIN_IN_SIF" \
    --receptor "$1" --ligand "$2" \
    --center_x "$3" --center_y "$4" --center_z "$5" \
    --size_x "$6" --size_y "$6" --size_z "$6" \
    --thread "$VINAGPU_THREAD" --num_modes "$NUM_MODES" \
    --opencl_binary_path "$VINAGPU_OPENCL_PATH_SIF" \
    --out "$7"
}


deploy_available() {  # $1 tool  $2 deploy  -> 0 if runnable
  case "$1_$2" in
    vina_local)        command -v "$VINA_LOCAL_BIN" >/dev/null 2>&1 ;;
    vina_container)    [ -n "$SIF_VINA" ] && [ -f "$SIF_VINA" ] && command -v apptainer >/dev/null 2>&1 ;;
    vinagpu_local)     [ -n "$VINAGPU_LOCAL_BIN" ] && [ -x "$VINAGPU_LOCAL_BIN" ] ;;
    vinagpu_container) [ -f "$SIF_VINAGPU" ] && command -v apptainer >/dev/null 2>&1 ;;
    *) return 1 ;;
  esac
}
