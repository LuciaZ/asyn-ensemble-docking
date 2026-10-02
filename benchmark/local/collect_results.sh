#!/usr/bin/env bash
# collect every benchmark result on this machine plus its environment into one Markdown archive
set -uo pipefail; set +o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bench_config.sh"
ARCH=~/data/benchmark/vina_local/archive
mkdir -p "$ARCH"
OUT="$ARCH/BENCHMARK_local_$(hostname)_$(date +%Y%m%d).md"

{
echo "# Benchmark archive — $(hostname) — $(date +%Y-%m-%d)"
echo
echo "## Environment"
echo '```'
cat "$ARCH"/env_$(hostname)_*.txt 2>/dev/null | head -60
echo '```'
echo
echo "## Results (frame scaling / res=all / warm container)"
echo
cat ~/data/benchmark/vina_local/benchmark_table.md 2>/dev/null
echo
echo "## Response to ligand complexity (50 ligands)"
echo
cat ~/data/benchmark/vina_local/multilig_table.md 2>/dev/null
echo
echo "## Key findings"
cat <<'NOTES'
- single box: Vina-GPU ~1.43x faster
- per frame (20 boxes): Vina-GPU ~1.30x faster; GPU utilisation only ~30%
- context floor: 83% of a single dock (~1.1 s) is GPU context rebuild, which Vina-GPU cannot amortise (no batch-receptor or batch-box interface)
- container overhead: ~0.1 s/dock (~5%), negligible
- 50 ligands: CPU time depends strongly on ligand size (33-76 s), GPU is flat (~29 s); speed-up grows with ligand size (small ~1.03x, large ~2.6x)
- the cluster phase should compare against AutoDock-GPU (native --filelist batching amortises the context cost)
NOTES
} > "$OUT"
echo "archive -> $OUT"
