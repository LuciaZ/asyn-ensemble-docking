#!/usr/bin/env bash
# Multi-ligand benchmark: one frame per ligand, all 20 boxes, box edge set per ligand from its Rg.
# Usage: bench_multi_lig.sh <tool> <rep>    tool: vina | vinagpu
set -euo pipefail
set +o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$SCRIPT_DIR/bench_config.sh"; set +a
export -f dock_vina_local dock_vinagpu_container

TOOL="$1"; REP="$2"
LIGAND_DIR="/home/zhihanz/data/DiffDock_copy/data/ligand_pdbqt/asyn_ligands_pdbqt"
DEPLOY=$([ "$TOOL" = "vina" ] && echo "local" || echo "container")
RUNDIR="$WORKROOT/runs_multilig/${TOOL}__rep${REP}"
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR/docks"
TIMING="$RUNDIR/timing.csv"; GPUCSV="$RUNDIR/gpu.csv"; META="$RUNDIR/meta.env"
echo "tool,ligand,edge,n_heavy,respos,resseq,wall_s,exit" > "$TIMING"

FR=$(find "$FRAMES_DIR" -maxdepth 1 -name "$FRAME_GLOB" | sort | head -1)
CENTERS="$RUNDIR/frame.centers"
python3 "$SCRIPT_DIR/compute_centers.py" "$FR" > "$CENTERS"

mapfile -t LIGS < <(find "$LIGAND_DIR" -maxdepth 1 -name '*_charged.pdbqt' | sort -V)

FN="dock_${TOOL}_${DEPLOY}"

( while true; do
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
             --format=csv,noheader,nounits -i "$GPU_IDX" 2>/dev/null | tr -d ' ')"
    sleep 0.5
  done ) > "$GPUCSV" &
SMPID=$!; trap 'kill "$SMPID" 2>/dev/null || true' EXIT
echo "run_start=$(date +%s.%N)" > "$META"

for lig in "${LIGS[@]}"; do
  lname=$(basename "$lig" .pdbqt)
  edge=$(python3 "$SCRIPT_DIR/compute_ligand_edge.py" "$lig")
  nheavy=$(awk '/^(ATOM|HETATM)/ && $NF !~ /^H/' "$lig" | wc -l)
  for pos in $(seq 1 20); do
    read -r rs cx cy cz < <(sed -n "${pos}p" "$CENTERS")
    out="$RUNDIR/docks/${lname}.res${rs}.pdbqt"
    export EDGE="$edge"
    t0=$(date +%s.%N); set +e
    bash -c "$FN \"\$1\" \"\$2\" \"\$3\" \"\$4\" \"\$5\" \"\$6\" \"\$7\"" \
      _ "$FR" "$lig" "$cx" "$cy" "$cz" "$edge" "$out" > "${out}.log" 2>&1
    ec=$?; set -e
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
    echo "$TOOL,$lname,$edge,$nheavy,$pos,$rs,$wall,$ec" >> "$TIMING"
    rm -f "$out"
  done
  echo "  $lname  edge=$edge  nheavy=$nheavy  done"
done
echo "run_end=$(date +%s.%N)" >> "$META"
echo "done: $RUNDIR"
