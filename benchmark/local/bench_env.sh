#!/usr/bin/env bash
# Phase 0 environment inventory. Run once per machine and keep with the results.
# Usage: ./bench_env.sh > env_$(hostname)_$(date +%Y%m%d).txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== HOST ==="; hostname; date; uname -a
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv 2>/dev/null
nvidia-smi 2>/dev/null | grep -i "CUDA Version"
echo; echo "=== processes already on the GPU (should be empty before benchmarking) ==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null
echo; echo "=== CPU ==="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz|NUMA"
echo; echo "=== NUMA ==="
numactl --hardware 2>/dev/null | grep -E "available|node 0 size" || echo "single node / no numactl"
echo; echo "=== MEMORY ==="; free -g
echo; echo "=== LIMITS ==="
echo "memlock (ulimit -l): $(ulimit -l)"
echo "stack   (ulimit -s): $(ulimit -s)"
echo; echo "=== SOFTWARE (host) ==="
echo "gcc: $(gcc --version 2>/dev/null | head -1)"
echo "apptainer: $(apptainer --version 2>/dev/null)"
echo "python/conda env: $(python --version 2>&1) / $CONDA_DEFAULT_ENV"
echo "vina (host, local): $(vina --version 2>/dev/null | head -1)"
echo; echo "=== OpenCL (host) ==="
clinfo 2>/dev/null | grep -iE "platform name|device name" | head -6 || echo "no clinfo"
echo; echo "=== CONTAINERS ==="
for s in "$SIF_VINAGPU" "$SIF_VINA" /home/zhihanz/data/VinaGPU/vina_cpu.sif; do
  [ -f "$s" ] && { echo "$s"; md5sum "$s"; }
done
echo "vina (container): $(apptainer exec /home/zhihanz/data/VinaGPU/vina_cpu.sif vina --version 2>/dev/null | head -1)"
echo; echo "=== BENCHMARK PARAMS (bench_config.sh) ==="
grep -E "^(EXHAUSTIVENESS|N_CPU|VINAGPU_THREAD|NUM_MODES|SEED|GPU_IDX|LIGAND_PDBQT|FRAMES_DIR)=" \
  "$SCRIPT_DIR/bench_config.sh" 2>/dev/null
echo; echo "=== Vina-GPU commit (container) ==="
apptainer exec "$SIF_VINAGPU" bash -c 'cd /opt/Vina-GPU-2.1 && git log -1 --format="%h %ci" 2>/dev/null' 2>/dev/null || echo "n/a"
