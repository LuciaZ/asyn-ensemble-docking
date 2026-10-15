#!/bin/bash
# ============================================================================
# 00_setup.sh  --  one-time layout + pilot ligand selection
# Run once on the login node (andes) before generating configs.
# ============================================================================
set -euo pipefail

BASE="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026"
LIGSRC="/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/autodock_vina/ligands_pdbqt"

# 10 pilot ligands: cross-size sample, none previously docked except 5 (anchor)
PILOT=(38 28 5 34 24 3 17 53 9 40)

mkdir -p "${BASE}"/{configs,outputs_vinagpu/_raw,outputs_vinagpu/best,done,logs,ligands_pilot,scripts}

# symlink pilot ligands into a clean folder for ligand_directory batch mode
rm -f "${BASE}/ligands_pilot"/*.pdbqt 2>/dev/null || true
for n in "${PILOT[@]}"; do
    src="${LIGSRC}/${n}_charged.pdbqt"
    if [ -f "${src}" ]; then
        ln -sf "${src}" "${BASE}/ligands_pilot/${n}_charged.pdbqt"
    else
        echo "WARNING: missing ${src}"
    fi
done

echo "Pilot ligands linked:"
ls -l "${BASE}/ligands_pilot/"
echo
echo "Count: $(find "${BASE}/ligands_pilot" -name '*.pdbqt' | wc -l) (expected 10)"
echo "Base tree ready at ${BASE}"
