#!/usr/bin/env bash
# Vina-GPU fully amortised per-dock floor: one process runs K ligands back to back, so the GPU context and kernels load only once.
# Usage: probe_floor.sh <K>   (K = ligands per batch; 20 matches the number of boxes per frame)


set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bench_config.sh"
K="${1:-20}"

TMP="$WORKROOT/floor_probe"; rm -rf "$TMP"; mkdir -p "$TMP/lig" "$TMP/out"
for i in $(seq -w 1 "$K"); do ln -s "$LIGAND_PDBQT" "$TMP/lig/lig_${i}.pdbqt"; done
FR=$(find "$FRAMES_DIR" -maxdepth 1 -name "$FRAME_GLOB" | sort | head -1)
read -r rs cx cy cz < <(python3 "$SCRIPT_DIR/compute_centers.py" "$FR" | sed -n '10p')
LIGDIR="$(dirname "$LIGAND_PDBQT")"

t0=$(date +%s.%N)
apptainer exec --nv --pwd "$VINAGPU_OPENCL_PATH_SIF" \
  -B /etc/OpenCL/vendors:/etc/OpenCL/vendors \
  -B "$WORKROOT","$FRAMES_DIR","$VINAGPU_OPENCL_PATH_SIF","$LIGDIR" \
  "$SIF_VINAGPU" \
  AutoDock-Vina-GPU-2-1 \
    --receptor "$FR" --ligand_directory "$TMP/lig" --output_directory "$TMP/out" \
    --center_x "$cx" --center_y "$cy" --center_z "$cz" \
    --size_x "$EDGE" --size_y "$EDGE" --size_z "$EDGE" \
    --thread "$VINAGPU_THREAD" --num_modes "$NUM_MODES" \
    --opencl_binary_path "$VINAGPU_OPENCL_PATH_SIF" 2>&1 | tail -4
t1=$(date +%s.%N)
n_out=$(find "$TMP/out" -name '*.pdbqt' | wc -l)
awk "BEGIN{printf \"\n%d ligands in batch, wall clock %.2f s, per-dock floor = %.3f s/dock\n\", $K, $t1-$t0, ($t1-$t0)/$K}"
echo "pose files produced: $n_out (expected $K)"
