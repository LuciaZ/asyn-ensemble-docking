#!/usr/bin/env bash
# Usage: conc_scan_cpu.sh <N_procs> <cores_each> <M_docks>
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster_config.sh"
N="$1"; C="$2"; M="${3:-20}"
LIGDIR=$(dirname "$LIGAND_PDBQT")
FR="$FRAMES_DIR/protein_only_frame_00000_fasudil_apo.pdbqt"
SC="${SLURM_TMPDIR:-/scratch/$USER}/cpuconc_N${N}_C${C}"; mkdir -p "$SC"
CEN="$SC/centers.txt"
python3 "$SLURM_SUBMIT_DIR/compute_centers.py" "$FR" | head -n "$M" > "$CEN"
worker() {
  local wid="$1" i=0
  while read -r rs cx cy cz; do
    i=$((i+1))
    apptainer exec -B "$FRAMES_DIR","$LIGDIR","$SC" "$SIF_VINA" \
      vina --receptor "$FR" --ligand "$LIGAND_PDBQT" \
        --center_x "$cx" --center_y "$cy" --center_z "$cz" \
        --size_x "$EDGE" --size_y "$EDGE" --size_z "$EDGE" \
        --exhaustiveness "$EXHAUSTIVENESS" --num_modes "$NUM_MODES" \
        --seed "$SEED" --cpu "$C" --out "$SC/w${wid}_${i}.pdbqt" > "$SC/w${wid}_${i}.log" 2>&1
  done < "$CEN"
}
t0=$(date +%s.%N)
for w in $(seq 1 "$N"); do worker "$w" & done
wait
t1=$(date +%s.%N)
total=$(awk "BEGIN{print $t1-$t0}")
docks=$((N*M))
thr=$(awk "BEGIN{printf \"%.0f\", $docks/$total*3600}")
echo "N_procs=$N cores_each=$C total_cores=$((N*C)) docks=$docks total=$(awk "BEGIN{printf \"%.1f\",$total}")s throughput=${thr} docks/hr"
