#!/usr/bin/env python3
# ============================================================================
# extract_holo_receptors_pdb.py -- holo bound-frame receptors as PROTEIN PDB
#
# WHY: the PDBQT receptors (obabel -xr) carry no/merged hydrogens, so
# docking_tools.hbond found ~1/3 the H-bonds apo did -- an artifact, not
# chemistry. Analysis needs all-atom PDBs matching what apo used, straight
# from the MD topology (hydrogens intact). Docking still uses the PDBQTs.
#
# For each pilot ligand: read its holo trajectory, pull the 5000 bound frames
# selected earlier (bound_out/selected_frame_idx.npy), strip the ligand so
# only the 20-residue protein remains, write a protein PDB, and convert to a
# rigid receptor PDBQT with the SAME obabel recipe apo used (-xr).
#
# Scheme A: these receptors will later be docked with the apo 20-residue-box,
# 18.75 A protocol on the cluster (config generator + docking reused).
#
# Ligand located by exclusion (non-protein/non-ion/non-water, >=6 heavy atoms),
# same as the bound-frame analysis (residue name is "<1>", unreliable).
#
# Run (diffdock env, workstation):
#   python extract_holo_receptors.py --smoke        # 5 frames/ligand, lig 38 only
#   python extract_holo_receptors.py                # 10 pilot ligands, 5000 each
# ============================================================================
import os, sys, re, glob, argparse, subprocess
import numpy as np
import mdtraj as md

MDROOT   = "/home/zhihanz/data/MDdata"
OUT_ROOT = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/holo_receptors_pdb"
PILOT    = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

STD_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
             "SER THR TRP TYR VAL HID HIE HIP HISE HISD HISP".split())

def ligand_map():
    m = {}
    for f in glob.glob(os.path.join(MDROOT, "*.tar")) + glob.glob(os.path.join(MDROOT, "*.tar.gz")):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)", os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)   # lig -> desres_id
    return m

def find_traj_dir(desres_id):
    hits = glob.glob(os.path.join(
        MDROOT, f"DESRES-Trajectory_biorxiv2021-{desres_id}-no-water-glue", "*",
        "top_from_local_mae.pdb"))
    return os.path.dirname(hits[0]) if hits else None

def obabel_receptor(pdb_in, pdbqt_out):
    # apo-matched: rigid receptor (-xr), pH 7.4 protonation (-p 7.4)
    cmd = ["obabel", pdb_in, "-O", pdbqt_out, "-xr", "-p", "7.4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(pdbqt_out)

def extract_ligand(lig, desres_id, n_smoke=None):
    tdir = find_traj_dir(desres_id)
    if tdir is None:
        print(f"  lig{lig}: no traj dir for {desres_id}"); return
    top_pdb = os.path.join(tdir, "top_from_local_mae.pdb")
    dcd     = os.path.join(tdir, "traj_all.dcd")
    sel_npy = os.path.join(tdir, "bound_out", "selected_frame_idx.npy")
    for f in (top_pdb, dcd, sel_npy):
        if not os.path.exists(f):
            print(f"  lig{lig}: missing {os.path.basename(f)}"); return

    sel = np.load(sel_npy)
    if n_smoke:
        sel = sel[:n_smoke]
    sel = np.sort(sel)

    top = md.load_topology(top_pdb)
    prot_atoms = top.select("protein")
    if prot_atoms.size == 0:
        # fallback: standard-AA residues by name
        prot_atoms = np.array([a.index for a in top.atoms
                               if a.residue.name.upper() in STD_AA], int)
    print(f"  lig{lig}: {sel.size} bound frames, protein atoms={prot_atoms.size}", flush=True)

    outdir = os.path.join(OUT_ROOT, f"lig{lig}")
    os.makedirs(outdir, exist_ok=True)

    # stream only the selected frames (iterload then index is wasteful; use
    # md.load with frame indices via a slice per chunk). Simplest robust way:
    # load in chunks, keep frames whose global index is in `sel`.
    sel_set = set(int(x) for x in sel)
    written = 0
    gidx = 0
    for chunk in md.iterload(dcd, top=top_pdb, chunk=2000):
        for k in range(chunk.n_frames):
            if gidx in sel_set:
                out_pdb = os.path.join(outdir, f"frame_{gidx:06d}.pdb")
                if not os.path.exists(out_pdb):
                    fr = chunk.slice(k)
                    prot = fr.atom_slice(prot_atoms)
                    prot.save_pdb(out_pdb)   # keeps hydrogens from the MD topology
                written += 1
            gidx += 1
        if written and written % 500 == 0:
            print(f"    ...{written} PDBs written", flush=True)
    print(f"  lig{lig}: DONE {written} PDBs -> {outdir}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=PILOT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    lmap = ligand_map()
    ligs = [38] if args.smoke else args.ligs
    n_smoke = 5 if args.smoke else None
    for lig in ligs:
        did = lmap.get(lig)
        if did is None:
            print(f"lig{lig}: no DESRES id in tar names, skip"); continue
        print(f"=== lig {lig} (DESRES {did}) ===", flush=True)
        extract_ligand(lig, did, n_smoke=n_smoke)
    print(f"\nreceptor PDBs -> {OUT_ROOT}/lig*/frame_*.pdb")

if __name__ == "__main__":
    main()
