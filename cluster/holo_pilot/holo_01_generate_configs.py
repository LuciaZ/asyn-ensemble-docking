#!/usr/bin/env python3
# ============================================================================
# holo_01_generate_configs.py  (v2: single-ligand uses `out =`, not output_directory)
#
# Vina-GPU single-ligand mode IGNORES output_directory and writes to the
# ligand's source dir. Must use `out = <unique file>` per box instead.
#
# Tree:  configs/lig{N}/frame_{fid}/res{NN}.txt
# Output: outputs_vinagpu/_raw/lig{N}/frame_{fid}_res{NN}_out.pdbqt  (flat)
#
# Usage:  python holo_01_generate_configs.py <LIG>
# ============================================================================
import sys, os, glob

BASE     = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot"
RECROOT  = BASE + "/holo_receptors"
LIGDIR   = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/autodock_vina/ligands_pdbqt"
CFGROOT  = BASE + "/configs"
OUTROOT  = BASE + "/outputs_vinagpu"

BOX_EDGE = 18.75
N_RES    = 20
THREAD   = 8000
NUM_MODES = 20
SKIP_EXISTING = True

def residue_centers(pdbqt_path):
    acc = {}
    with open(pdbqt_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = line[76:78].strip() if len(line) >= 78 else ""
            if elem == "H":
                continue
            try:
                resSeq = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            a = acc.setdefault(resSeq, [0.0, 0.0, 0.0, 0])
            a[0] += x; a[1] += y; a[2] += z; a[3] += 1
    return {r: (a[0]/a[3], a[1]/a[3], a[2]/a[3]) for r, a in acc.items() if a[3] > 0}

def write_config(cfg_path, receptor, ligand, out_file, cx, cy, cz):
    with open(cfg_path, "w") as f:
        f.write(f"receptor = {receptor}\n")
        f.write(f"ligand = {ligand}\n")
        f.write(f"out = {out_file}\n")            # single-ligand mode: use out=, NOT output_directory
        f.write(f"center_x = {cx:.3f}\n")
        f.write(f"center_y = {cy:.3f}\n")
        f.write(f"center_z = {cz:.3f}\n")
        f.write(f"size_x = {BOX_EDGE}\n")
        f.write(f"size_y = {BOX_EDGE}\n")
        f.write(f"size_z = {BOX_EDGE}\n")
        f.write(f"thread = {THREAD}\n")
        f.write(f"num_modes = {NUM_MODES}\n")

def main():
    lig = int(sys.argv[1])
    ligand_pdbqt = os.path.join(LIGDIR, f"{lig}_charged.pdbqt")
    if not os.path.exists(ligand_pdbqt):
        sys.exit(f"ERROR: ligand {ligand_pdbqt} missing")

    recs = sorted(glob.glob(os.path.join(RECROOT, f"lig{lig}", "frame_*.pdbqt")))
    print(f"lig{lig}: {len(recs)} receptors", flush=True)

    out_lig_dir = os.path.join(OUTROOT, "_raw", f"lig{lig}")
    os.makedirs(out_lig_dir, exist_ok=True)

    made = skipped = 0
    for rec in recs:
        fid = os.path.basename(rec).replace("frame_", "").replace(".pdbqt", "")
        centers = residue_centers(rec)
        cfg_dir = os.path.join(CFGROOT, f"lig{lig}", f"frame_{fid}")
        os.makedirs(cfg_dir, exist_ok=True)
        for res in sorted(centers)[:N_RES]:
            cfg_path = os.path.join(cfg_dir, f"res{res:02d}.txt")
            if SKIP_EXISTING and os.path.exists(cfg_path):
                skipped += 1; continue
            out_file = os.path.join(out_lig_dir, f"frame_{fid}_res{res:02d}_out.pdbqt")
            cx, cy, cz = centers[res]
            write_config(cfg_path, rec, ligand_pdbqt, out_file, cx, cy, cz)
            made += 1
        if (made + skipped) % 20000 == 0:
            print(f"  ...{made} made, {skipped} skipped", flush=True)
    print(f"lig{lig}: DONE made={made} skipped={skipped}", flush=True)

if __name__ == "__main__":
    main()
