# Ensemble docking against the alpha-synuclein C-terminus

Reproduction and extension of *J. Chem. Inf. Model.* `acs.jcim.5c00370`:
ensemble docking of charged small molecules against residues D121-A140.

Data (docked ensembles, receptor ensembles, MD reference, analysis):
`/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/SHARED_asyn_docking/` -- see its
`CONTENTS.md`. This repository holds code only.

## Layout

    ligand_prep/        SMILES -> SDF -> charged ligands -> PDBQT; protein frames -> PDBQT
    workstation/        receptor extraction, fingerprints, scoring, RMSD, errors, figures
    cluster/            config generation and single-node Vina-GPU docking
    cluster/holo_pilot/ the 10-ligand holo pilot: configs, docking, gap-fill, collapse
    benchmark/          engine and throughput benchmarks (local and cluster)
    legacy/             superseded scripts, kept for the record; not part of the pipeline

## Pipeline

1. `ligand_prep/` -> ligand and protein PDBQT
2. `workstation/extract_holo_receptors.py` -> holo receptors, PDBQT (docking)
3. `workstation/extract_holo_receptors_pdb.py` -> same frames, all-atom PDB (analysis)
4. `cluster/gen_configs_remaining.py` -> config tree + work manifest
5. `cluster/dock_one_node.slurm` -> docking, one process per GPU
6. `cluster/holo_pilot/holo_collapse_v2.py` -> best-of-20 winner per frame
7. `workstation/analyze_holo_v2.py` -> per-residue interaction fingerprints
8. `workstation/md_reference_fingerprint.py` -> MD reference on the same frames
9. `workstation/analyze_docking_scores.py`, `holo_rmsd.py`, `error_analysis.py`, `plot_*.py`

`docking_tools` (the lab's interaction-analysis module) is imported by the
analysis scripts and is not vendored here.

## Protocol

20 boxes per frame, centred on each residue's heavy-atom centroid (unweighted),
uniform 18.75 A edge, `num_modes = 20`, best-of-20 collapse. Apo: 5005 frames
(stride 4 of 20020), shared by all ligands. Holo: 5000 bound frames per ligand
(minimum protein-ligand heavy-atom distance < 6 A).

The published trio (47 / 41 = fasudil / 23) used Rg-scaled boxes
(16.58 / 13.01 / 13.17 A) and 20,020 frames. Larger boxes search more space
and find more favourable poses, so raw scores from the two protocols are not
comparable; normalise against a shared reference and say so.

## Pitfalls

**Two receptor formats.** `obabel -xr -p 7.4` removes all 128 hydrogens and
restores ~29 polar ones. Vina docks correctly against that, but
`docking_tools.hbond` needs explicit donor hydrogens, so analysing against
PDBQT under-counts H-bonds about threefold. Dock with PDBQT, analyse with PDB.

**AutoDock types are not element symbols.** `A` is aromatic carbon, `NA/OA/SA`
are acceptor N/O/S, `HD` is polar hydrogen.

**Vina-GPU single-ligand mode ignores `output_directory`**; give every config a
unique `out =` or all boxes overwrite one file.

**One Vina-GPU process per GPU.** Several processes on one card fail with
`CL_PLATFORM_NOT_FOUND_KHR`. `--array=...%N` does not guarantee each task a
card: surplus tasks started anyway, saw "No devices were found", and produced
~900k silent failures in an hour. Size the array to the free GPUs and check
`nvidia-smi` at task start (the script now does).

**Precompiled kernels are architecture-specific.** `vgpu_make_ampere` is GA102
(A5000/A5500) only. Without precompiled kernels single-ligand mode JIT-compiles
per box (~11 s vs ~1 s).

**Most of a single dock is GPU context setup** (~83% of ~1.1 s in the local
benchmark), which is why batching container starts helps and why more than one
process per card may help further -- untested at production scale.

**dartfs stalls on large directories.** Listing a directory with 10^5+ entries
hangs for minutes. Drive work from a manifest file and build paths directly.

**Done-markers make it survivable.** A multi-day Slurm outage, preemption and
repeated wall-clock timeouts lost no finished work.

**Frames are correlated.** Bound fraction: g = 45-76. Winner scores from
consecutive frames: g > 1000. Report `sd / sqrt(N / g)`, not `sd / sqrt(N)`.

The C-terminus has **three** tyrosines: Y125, Y133, Y136.
