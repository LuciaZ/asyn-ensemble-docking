#!/usr/bin/env bash
# Usage: cluster_bench.sh <tool> <nframes> <ncpu> <pin> <rep>
set -euo pipefail
set +o pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/cluster_config.sh"

TOOL="$1"; NFRAMES="$2"; NCPU="$3"; PIN="$4"; REP="$5"
NODE=$(hostname -s)
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | tr ' ' '_')
THREAD=$(gpu_thread_for_node)

RUNDIR="$RESULTS/runs/${TOOL}__${NODE}__ncpu${NCPU}__pin${PIN}__n${NFRAMES}__rep${REP}"
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR/docks"
TIMING="$RUNDIR/timing.csv"; GPUCSV="$RUNDIR/gpu.csv"; META="$RUNDIR/meta.env"
echo "tool,node,gpu,thread,ncpu,pin,nframes,rep,frame,respos,resseq,wall_s,exit" > "$TIMING"

SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/vbench_${SLURM_JOB_ID:-manual}}"; mkdir -p "$SCRATCH"

NUMA_PREFIX=""
if [ "$PIN" = "on" ]; then
  NODE_ID=$(gpu_numa_node)
  if [ -n "$NODE_ID" ]; then
    NUMA_PREFIX="numactl --cpunodebind=$NODE_ID --membind=$NODE_ID"
    echo "pin: GPU on NUMA node $NODE_ID"
  else
    echo "pin=on but single socket / no affinity, skipping (=off)"
  fi
fi

mapfile -t FRAMES < <(find "$FRAMES_DIR" -maxdepth 1 -name "$FRAME_GLOB" 2>/dev/null | sort | head -n "$NFRAMES")
[ "${#FRAMES[@]}" -lt "$NFRAMES" ] && { echo "ERROR: not enough frames ${#FRAMES[@]}/$NFRAMES"; exit 3; }

run_vina() {
  $NUMA_PREFIX apptainer exec -B "$RESULTS","$FRAMES_DIR","$SCRATCH","$(dirname "$LIGAND_PDBQT")" "$SIF_VINA" \
    vina --receptor "$1" --ligand "$2" \
      --center_x "$3" --center_y "$4" --center_z "$5" \
      --size_x "$6" --size_y "$6" --size_z "$6" \
      --exhaustiveness "$EXHAUSTIVENESS" --num_modes "$NUM_MODES" \
      --seed "$SEED" --cpu "$NCPU" --out "$7"
}
run_vinagpu() {
  local KM; KM=$(gpu_make_dir)
  [ -z "$KM" ] && { echo "ERROR: no make build directory for this GPU architecture"; return 1; }
  $NUMA_PREFIX apptainer exec --nv --pwd "$KM" \
    -B /etc/OpenCL/vendors:/etc/OpenCL/vendors \
    -B "$RESULTS","$FRAMES_DIR","$SCRATCH","$(dirname "$LIGAND_PDBQT")","$KM" "$SIF_VINAGPU" \
    "$KM/AutoDock-Vina-GPU-2-1" --receptor "$1" --ligand "$2" \
      --center_x "$3" --center_y "$4" --center_z "$5" \
      --size_x "$6" --size_y "$6" --size_z "$6" \
      --thread "$THREAD" --num_modes "$NUM_MODES" \
      --opencl_binary_path "$KM" --out "$7"
}

( while true; do
    echo "$(date +%s.%N),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
             --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')"
    sleep 0.5
  done ) > "$GPUCSV" &
SMPID=$!; trap 'kill "$SMPID" 2>/dev/null || true' EXIT
{ echo "node=$NODE"; echo "gpu=$GPUNAME"; echo "thread=$THREAD"; echo "ncpu=$NCPU"; \
  echo "pin=$PIN"; echo "run_start=$(date +%s.%N)"; } > "$META"

for f in "${FRAMES[@]}"; do
  fid=$(basename "$f")
  cen="$RUNDIR/docks/${fid}.centers"
  [ -f "$cen" ] || python3 "$SCRIPT_DIR/compute_centers.py" "$f" > "$cen"
  for pos in $(seq 1 20); do
    read -r rs cx cy cz < <(sed -n "${pos}p" "$cen")
    out="$RUNDIR/docks/${fid}.res${rs}.pdbqt"
    t0=$(date +%s.%N); set +e
    if [ "$TOOL" = "vina" ]; then run_vina "$f" "$LIGAND_PDBQT" "$cx" "$cy" "$cz" "$EDGE" "$out" >"${out}.log" 2>&1
    else run_vinagpu "$f" "$LIGAND_PDBQT" "$cx" "$cy" "$cz" "$EDGE" "$out" >"${out}.log" 2>&1; fi
    ec=$?; set -e
    t1=$(date +%s.%N); wall=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
    echo "$TOOL,$NODE,$GPUNAME,$THREAD,$NCPU,$PIN,$NFRAMES,$REP,$fid,$pos,$rs,$wall,$ec" >> "$TIMING"
    rm -f "$out"
  done
done
echo "run_end=$(date +%s.%N)" >> "$META"
echo "done: $RUNDIR"
