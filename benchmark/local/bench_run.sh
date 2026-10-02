#!/usr/bin/env bash
# Run one benchmark cell. Records per-dock wall + maxrss, and a GPU time series.
# Usage: bench_run.sh <tool> <deploy> <nframes> <residues> <rep>
#   tool:     vina | vinagpu
#   deploy:   local | container
#   residues: "all"  or comma list of 1-based residue positions, e.g. "10" or "1,10,20"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$SCRIPT_DIR/bench_config.sh"; set +a
export -f dock_vina_local dock_vina_container dock_vinagpu_local dock_vinagpu_container

TOOL="$1"; DEPLOY="$2"; NFRAMES="$3"; RESIDUES="$4"; REP="$5"
RESTAG="${RESIDUES//,/-}"
RUNDIR="$WORKROOT/runs/${TOOL}__${DEPLOY}__n${NFRAMES}__res${RESTAG}__rep${REP}"
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR/docks"
TIMING="$RUNDIR/timing.csv"; GPUCSV="$RUNDIR/gpu.csv"; META="$RUNDIR/meta.env"
echo "tool,deploy,nframes,residues,rep,frame,respos,resseq,wall_s,exit,maxrss_kb" > "$TIMING"

# --- deterministic frame selection: SAME frames for every cell ---
mapfile -t FRAMES < <(find "$FRAMES_DIR" -maxdepth 1 -name "$FRAME_GLOB" 2>/dev/null | sort | head -n "$NFRAMES")
if [ "${#FRAMES[@]}" -lt "$NFRAMES" ]; then
  echo "ERROR: found only ${#FRAMES[@]} frames in $FRAMES_DIR (need $NFRAMES)"; exit 3
fi

# residue positions
if [ "$RESIDUES" = "all" ]; then RES_POS=($(seq 1 20)); else IFS=',' read -ra RES_POS <<< "$RESIDUES"; fi

# timing tool
TIMEBIN="$(command -v /usr/bin/time || true)"

# --- background GPU sampler (epoch-stamped for clean alignment) ---
( while true; do
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used \
             --format=csv,noheader,nounits -i "$GPU_IDX" 2>/dev/null | tr -d ' ')"
    sleep 0.5
  done ) > "$GPUCSV" &
SMPID=$!
trap 'kill "$SMPID" 2>/dev/null || true' EXIT

{ echo "host=$(hostname)"; echo "gpu_idx=$GPU_IDX"; echo "tool=$TOOL"; echo "deploy=$DEPLOY"; } > "$META"
echo "run_start=$(date +%s.%N)" >> "$META"

FN="dock_${TOOL}_${DEPLOY}"
for f in "${FRAMES[@]}"; do
  frame_id="$(basename "$f")"
  centers="$RUNDIR/docks/${frame_id}.centers"
  [ -f "$centers" ] || python3 "$SCRIPT_DIR/compute_centers.py" "$f" > "$centers"
  for pos in "${RES_POS[@]}"; do
    read -r resseq cx cy cz < <(sed -n "${pos}p" "$centers")
    out="$RUNDIR/docks/${frame_id}.res${resseq}.pdbqt"
    log="$RUNDIR/docks/${frame_id}.res${resseq}.log"
    tf="$(mktemp)"
    t0=$(date +%s.%N)
    set +e
    if [ -n "$TIMEBIN" ]; then
      "$TIMEBIN" -v -o "$tf" bash -c "$FN \"\$1\" \"\$2\" \"\$3\" \"\$4\" \"\$5\" \"\$6\" \"\$7\"" \
        _ "$f" "$LIGAND_PDBQT" "$cx" "$cy" "$cz" "$EDGE" "$out" >"$log" 2>&1
      ec=$?
      maxrss=$(grep -i 'Maximum resident' "$tf" | grep -o '[0-9]\+' | tail -1)
    else
      bash -c "$FN \"\$1\" \"\$2\" \"\$3\" \"\$4\" \"\$5\" \"\$6\" \"\$7\"" \
        _ "$f" "$LIGAND_PDBQT" "$cx" "$cy" "$cz" "$EDGE" "$out" >"$log" 2>&1
      ec=$?; maxrss="NA"
    fi
    set -e
    t1=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
    rm -f "$tf"
    echo "$TOOL,$DEPLOY,$NFRAMES,$RESTAG,$REP,$frame_id,$pos,$resseq,$wall,$ec,${maxrss:-NA}" >> "$TIMING"
    # keep only a couple of pose files to save disk; comment out to keep all
    [ "$pos" != "${RES_POS[0]}" ] && rm -f "$out"
  done
done
echo "run_end=$(date +%s.%N)" >> "$META"
echo "done: $RUNDIR"
