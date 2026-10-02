#!/usr/bin/env bash
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster_config.sh"
N="$1"; MODE="$2"; M="${3:-20}"
KM=$(gpu_make_dir); THREAD=$(gpu_thread_for_node)
LIGDIR=$(dirname "$LIGAND_PDBQT")
FR="$FRAMES_DIR/protein_only_frame_00000_fasudil_apo.pdbqt"
SC="${SLURM_TMPDIR}/conc_${MODE}_N${N}"; mkdir -p "$SC"
CEN="$SC/centers.txt"
python3 "$SLURM_SUBMIT_DIR/compute_centers.py" "$FR" | head -n "$M" > "$CEN"
worker() {
  local wid="$1" gpu="$2" i=0
  while read -r rs cx cy cz; do
    i=$((i+1))
    CUDA_VISIBLE_DEVICES=$gpu apptainer exec --nv --pwd "$KM" \
      -B /etc/OpenCL/vendors:/etc/OpenCL/vendors \
      -B "$FRAMES_DIR","$LIGDIR","$KM","$SC" "$SIF_VINAGPU" \
      "$KM/AutoDock-Vina-GPU-2-1" --receptor "$FR" --ligand "$LIGAND_PDBQT" \
        --center_x "$cx" --center_y "$cy" --center_z "$cz" \
        --size_x "$EDGE" --size_y "$EDGE" --size_z "$EDGE" \
        --thread "$THREAD" --num_modes "$NUM_MODES" \
        --opencl_binary_path "$KM" --out "$SC/w${wid}_${i}.pdbqt" > "$SC/w${wid}_${i}.log" 2>&1
  done < "$CEN"
}
t0=$(date +%s.%N)
for w in $(seq 1 "$N"); do
  if [ "$MODE" = "shared" ]; then gpu=0; else gpu=$(( (w-1) % 6 )); fi
  worker "$w" "$gpu" &
done
wait
t1=$(date +%s.%N)
total=$(awk "BEGIN{print $t1-$t0}")
docks=$((N*M))
thr=$(awk "BEGIN{printf \"%.0f\", $docks/$total*3600}")
echo "N=$N mode=$MODE docks=$docks total=$(awk "BEGIN{printf \"%.1f\",$total}")s throughput=${thr} docks/hr"
