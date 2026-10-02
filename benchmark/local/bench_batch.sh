#!/usr/bin/env bash
# Vina-GPU warm-container batch: one apptainer exec loops over every (frame x box) dock.
# Isolates apptainer start-up cost. Usage: bench_batch.sh <nframes> <residues> <rep>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bench_config.sh"

NFRAMES="$1"; RESIDUES="$2"; REP="$3"
RESTAG="${RESIDUES//,/-}"
RUNDIR="$WORKROOT/runs/vinagpu__container_batched__n${NFRAMES}__res${RESTAG}__rep${REP}"
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR/docks"
TIMING="$RUNDIR/timing.csv"; GPUCSV="$RUNDIR/gpu.csv"; META="$RUNDIR/meta.env"
JOBS="$RUNDIR/jobs.tsv"; INNER="$RUNDIR/inner.sh"
echo "tool,deploy,nframes,residues,rep,frame,respos,resseq,wall_s,exit,maxrss_kb" > "$TIMING"

mapfile -t FRAMES < <(find "$FRAMES_DIR" -maxdepth 1 -name "$FRAME_GLOB" 2>/dev/null | sort | head -n "$NFRAMES")
[ "${#FRAMES[@]}" -lt "$NFRAMES" ] && { echo "ERROR: only ${#FRAMES[@]} frames"; exit 3; }
if [ "$RESIDUES" = "all" ]; then RES_POS=($(seq 1 20)); else IFS=',' read -ra RES_POS <<< "$RESIDUES"; fi

# precompute box centres on the host and build the job list
: > "$JOBS"
for f in "${FRAMES[@]}"; do
  fid="$(basename "$f")"
  c="$RUNDIR/docks/${fid}.centers"
  [ -f "$c" ] || python3 "$SCRIPT_DIR/compute_centers.py" "$f" > "$c"
  for pos in "${RES_POS[@]}"; do
    read -r rs cx cy cz < <(sed -n "${pos}p" "$c")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$f" "$cx" "$cy" "$cz" "$RUNDIR/docks/${fid}.res${rs}.pdbqt" "$pos" "$rs" "$fid" >> "$JOBS"
  done
done

# generate the in-container loop script (constants hard-coded to avoid passing them through env)
cat > "$INNER" <<INNEREOF
#!/usr/bin/env bash
while IFS=\$'\t' read -r rec cx cy cz out pos rs fid; do
  t0=\$(date +%s.%N)
  AutoDock-Vina-GPU-2-1 \
    --receptor "\$rec" --ligand "$LIGAND_PDBQT" \
    --center_x "\$cx" --center_y "\$cy" --center_z "\$cz" \
    --size_x "$EDGE" --size_y "$EDGE" --size_z "$EDGE" \
    --thread "$VINAGPU_THREAD" --num_modes "$NUM_MODES" \
    --opencl_binary_path "$VINAGPU_OPENCL_PATH_SIF" \
    --out "\$out" > "\${out}.log" 2>&1
  ec=\$?
  t1=\$(date +%s.%N)
  wall=\$(awk "BEGIN{printf \"%.4f\", \$t1-\$t0}")
  echo "vinagpu,container_batched,$NFRAMES,$RESTAG,$REP,\$fid,\$pos,\$rs,\$wall,\$ec,NA" >> "$TIMING"
done < "$JOBS"
INNEREOF

LIGDIR="$(dirname "$LIGAND_PDBQT")"
( while true; do
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used \
             --format=csv,noheader,nounits -i "$GPU_IDX" 2>/dev/null | tr -d ' ')"
    sleep 0.5
  done ) > "$GPUCSV" &
SMPID=$!; trap 'kill "$SMPID" 2>/dev/null || true' EXIT

echo "host=$(hostname)"; echo "run_start=$(date +%s.%N)" > "$META"
apptainer exec --nv --pwd "$VINAGPU_OPENCL_PATH_SIF" \
  -B /etc/OpenCL/vendors:/etc/OpenCL/vendors \
  -B "$WORKROOT","$FRAMES_DIR","$VINAGPU_OPENCL_PATH_SIF","$LIGDIR" \
  "$SIF_VINAGPU" \
  bash "$INNER"
echo "run_end=$(date +%s.%N)" >> "$META"
echo "done: $RUNDIR"
