#!/usr/bin/env bash
# Sweep tool x deploy x n_frames x reps on prl-desktop1, then build the table.
#   SMOKE=1 ./run_local_bench.sh    # quick check first (your standard practice)
#   ./run_local_bench.sh            # full local pass
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bench_config.sh"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  NFRAMES_LIST="1 5"; RESIDUES="all"; REPS=2
else
  NFRAMES_LIST="1 10 100"; RESIDUES="all"; REPS=3   # RESIDUES=10 isolates engine speed
fi
# For real per-frame PIPELINE throughput (20 boxes/frame), set RESIDUES="all"
# and expect ~20x the wall per frame.

TOOLS="vina vinagpu"
DEPLOYS="local container"

echo "== local benchmark (SMOKE=$SMOKE) =="
for tool in $TOOLS; do
  for deploy in $DEPLOYS; do
    if ! deploy_available "$tool" "$deploy"; then
      echo "SKIP  $tool/$deploy  (binary or container not found — see bench_config.sh)"
      continue
    fi
    for n in $NFRAMES_LIST; do
      for rep in $(seq 1 "$REPS"); do
        echo ">> $tool/$deploy  n=$n  res=$RESIDUES  rep=$rep"
        bash "$SCRIPT_DIR/bench_run.sh" "$tool" "$deploy" "$n" "$RESIDUES" "$rep"
      done
    done
  done
done

echo "== aggregating =="
python3 "$SCRIPT_DIR/parse_bench.py" "$WORKROOT/runs" "$WORKROOT"
echo "table -> $WORKROOT/benchmark_table.md  (+ .csv)"
