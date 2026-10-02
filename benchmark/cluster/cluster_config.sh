#!/usr/bin/env bash
PROJ=/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan
CONTAINERS="$PROJ/containers"
SIF_VINAGPU="$CONTAINERS/vina_gpu_v2.sif"
SIF_VINA="$CONTAINERS/vina_cpu.sif"
VGPU_BASE="$PROJ/VinaGPU/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1"
FRAMES_DIR="$VGPU_BASE/protein_apo_pdbqt"
FRAME_GLOB="protein_only_frame_*_fasudil_apo.pdbqt"
LIGAND_PDBQT="$VGPU_BASE/ligands_pdbqt/47_charged.pdbqt"
EDGE="16.58"
NUM_MODES=20
SEED=42
EXHAUSTIVENESS=8
RESULTS="$PROJ/benchmark_cluster"

gpu_thread_for_node() {
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)
  case "$name" in
    *A5000*|*A5500*) echo 8000 ;;
    *2080*)          echo 5000 ;;
    *)               echo 4000 ;;
  esac
}
gpu_numa_node() {
  local nnodes; nnodes=$(numactl --hardware 2>/dev/null | awk '/available/{print $2}')
  [ "${nnodes:-1}" -le 1 ] && { echo ""; return; }
  nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}' | tail -1
}

gpu_make_dir() {
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)
  case "$name" in
    *A5000*|*A5500*) echo "$RESULTS/vgpu_make_ampere" ;;
    *2080*)          echo "$RESULTS/vgpu_make_turing" ;;
    *)               echo "" ;;
  esac
}
