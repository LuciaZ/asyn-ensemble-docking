#!/usr/bin/env python3
# ============================================================================
# 01_generate_configs.py
#
# Build the shared Vina-GPU config tree for the 10-ligand apo pilot.
#
#   * 10 ligands, docked together via ligand_directory batch mode (scheme A)
#   * stride-4 frames  -> 5005 frames  (indices 0,4,8,...,20016)
#   * unified cubic box, edge = 18.75 A for every box (largest-ligand policy)
#   * 20 residue boxes per frame, center = geometric mean of that residue's
#     heavy atoms in the apo receptor PDBQT (matches lab generate_gpf.py)
#
# One config per (frame, residue). Each config points ligand_directory at the
# shared 10-ligand folder and output_directory at a UNIQUE per-box folder so
# the 10 "{N}_charged_out.pdbqt" outputs of different boxes never collide.
#
# numpy-free (compute nodes lack it); resumable via SKIP_EXISTING.
#
# Usage:   python 01_generate_configs.py <FRAME_START> <FRAME_END>
#   generates configs for stride-selected frames whose ORIGINAL index is in
#   [FRAME_START, FRAME_END).  Called by 01_generate_configs.slurm in chunks.
#   Run with no args -> does all 5005 frames single-process (slow; for testing).
# ============================================================================
import sys, os

# ---- paths (EDIT ONLY IF LAYOUT CHANGES) -----------------------------------
BASE      = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026"
FRAMES    = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/autodock_vina/protein_apo_pdbqt"
FRAME_FMT = "protein_only_frame_{fid:05d}_fasudil_apo.pdbqt"   # apo receptor per frame
LIGDIR    = BASE + "/ligands_pilot"          # 10-ligand folder (symlinks), scheme A
CFGROOT   = BASE + "/configs"
OUTROOT   = BASE + "/outputs_vinagpu"        # per-box _raw dirs live here

# ---- protocol constants ----------------------------------------------------
N_FRAMES_TOTAL = 20020
FRAME_STRIDE   = 4                # 20020/4 -> 5005 frames
BOX_EDGE       = 18.75           # A, unified (npts=50 @0.375 for the AD-GPU engine)
N_RES          = 20              # residues 1..20 (paper renumbering of 121..140)
THREAD         = 8000            # Vina-GPU --thread (as in holo pipeline)
NUM_MODES      = 20
SKIP_EXISTING  = True

# ---- per-residue heavy-atom geometric center from a receptor PDBQT ---------
def residue_centers(pdbqt_path):
    """Return {resSeq: (cx,cy,cz)} geometric mean of heavy atoms per residue.
    Hydrogens skipped. Matches the lab generate_gpf.py convention."""
    acc = {}   # resSeq -> [sx,sy,sz,n]
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

def write_config(cfg_path, receptor, out_dir):
    with open(cfg_path, "w") as f:
        f.write(f"receptor = {receptor}\n")
        f.write(f"ligand_directory = {LIGDIR}\n")
        f.write(f"output_directory = {out_dir}\n")
        f.write(f"center_x = {CX:.3f}\n")
        f.write(f"center_y = {CY:.3f}\n")
        f.write(f"center_z = {CZ:.3f}\n")
        f.write(f"size_x = {BOX_EDGE}\n")
        f.write(f"size_y = {BOX_EDGE}\n")
        f.write(f"size_z = {BOX_EDGE}\n")
        f.write(f"thread = {THREAD}\n")
        f.write(f"num_modes = {NUM_MODES}\n")

def main():
    fstart = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fend   = int(sys.argv[2]) if len(sys.argv) > 2 else N_FRAMES_TOTAL
    made = skipped = missing = 0
    for fid in range(fstart, min(fend, N_FRAMES_TOTAL)):
        if fid % FRAME_STRIDE != 0:          # stride selection on ORIGINAL index
            continue
        receptor = os.path.join(FRAMES, FRAME_FMT.format(fid=fid))
        if not os.path.exists(receptor):
            missing += 1
            print(f"  MISSING receptor frame {fid:05d}", flush=True)
            continue
        centers = residue_centers(receptor)
        if len(centers) < N_RES:
            print(f"  WARN frame {fid:05d}: only {len(centers)} residues parsed", flush=True)
        cfg_dir = os.path.join(CFGROOT, f"frame_{fid:05d}")
        os.makedirs(cfg_dir, exist_ok=True)
        for res in sorted(centers)[:N_RES]:
            cfg_path = os.path.join(cfg_dir, f"res{res:02d}.txt")
            out_dir  = os.path.join(OUTROOT, "_raw", f"frame_{fid:05d}_res{res:02d}")
            if SKIP_EXISTING and os.path.exists(cfg_path):
                skipped += 1
                continue
            os.makedirs(out_dir, exist_ok=True)
            global CX, CY, CZ
            CX, CY, CZ = centers[res]
            write_config(cfg_path, receptor, out_dir)
            made += 1
    print(f"[{fstart}-{fend}] configs made={made} skipped={skipped} missing_frames={missing}", flush=True)

if __name__ == "__main__":
    main()
