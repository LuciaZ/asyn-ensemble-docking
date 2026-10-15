#!/usr/bin/env python3
# ============================================================================
# export_docked_xtc.py -- docked complex trajectories as xtc + pdb
#
# build_docked_trajectories.py stores each docked ensemble as a coordinate
# array plus a topology PDB, which suits the notebooks but is not a format
# anyone else can open directly. This writes the same ensembles as
#
#   lig{N}.xtc        protein + docked ligand, one frame per docked pose
#   lig{N}.pdb        topology (frame 0), ligand bonds as CONECT
#   lig{N}_frames.txt the source frame id of every trajectory frame
#
# i.e. the layout of the lab's example notebook (AutoDock_apo.xtc / .pdb), so
# md.load(xtc, top=pdb) just works.
#
# Residue numbering: protein 121-140 as in the construct, ligand as the last
# residue. The ligand's resSeq equals its topology index (20), so both
# "residue 20" and "resid 20" select it -- docking_tools.dual_contact uses the
# former, most other selections the latter.
#
# xtc stores coordinates at ~1/1000 nm precision, which is far finer than any
# docking pose warrants and roughly halves the size against float32 .npy.
#
# Run (diffdock env):
#   python export_docked_xtc.py --jobs apo_pilot10 --ligs 38      # one, to check
#   python export_docked_xtc.py                                   # everything built
# ============================================================================
import os, re, glob, json, argparse
import numpy as np
import mdtraj as md

SRC = os.path.expanduser("~/data/docked_trajectories")
OUT = os.path.expanduser("~/data/docked_trajectories_xtc")

JOB_NOTE = {
    "apo_pilot10": "Vina-GPU apo docking, 10 pilot ligands, 5005 apo frames "
                   "(stride 4 of 20020), uniform 18.75 A boxes",
    "apo_remaining37": "Vina-GPU apo docking, the other 37 ligands, same "
                       "protocol as apo_pilot10",
    "holo_pilot10": "Vina-GPU holo docking, 10 pilot ligands, 5000 bound "
                    "frames per ligand, uniform 18.75 A boxes",
    "holo_remaining37": "Vina-GPU holo docking, the other 37 ligands, same "
                        "protocol as holo_pilot10",
    "trio_apo": "Vina-GPU apo docking of ligands 47 / 41 (fasudil) / 23, all "
                "20020 apo frames, Rg-scaled boxes (16.58 / 13.01 / 13.17 A)",
}


def convert(job, lig, outdir, force=False):
    s = os.path.join(SRC, job)
    xyz_p = os.path.join(s, f"lig{lig}_xyz.npy")
    top_p = os.path.join(s, f"lig{lig}_top.pdb")
    if not (os.path.exists(xyz_p) and os.path.exists(top_p)):
        return None
    out_xtc = os.path.join(outdir, f"lig{lig}.xtc")
    if os.path.exists(out_xtc) and not force:
        print(f"  lig{lig}: already converted, skipping")
        return None

    top = md.load_topology(top_p)
    xyz = np.load(xyz_p, mmap_mode="r")
    if xyz.shape[1] != top.n_atoms:
        print(f"  lig{lig}: {xyz.shape[1]} coords vs {top.n_atoms} topology "
              f"atoms -- skipping")
        return None

    # write in chunks: a 37-ligand pass would otherwise hold several GB at once
    tmp = out_xtc + ".tmp"
    with md.formats.XTCTrajectoryFile(tmp, "w") as fh:
        for i in range(0, xyz.shape[0], 500):
            fh.write(np.asarray(xyz[i:i + 500], dtype=np.float32))
    os.replace(tmp, out_xtc)

    md.Trajectory(np.asarray(xyz[0:1]), top).save_pdb(
        os.path.join(outdir, f"lig{lig}.pdb"))

    fr = os.path.join(s, f"lig{lig}_frames.npy")
    if os.path.exists(fr):
        np.savetxt(os.path.join(outdir, f"lig{lig}_frames.txt"),
                   np.load(fr), fmt="%d")

    n_prot = len(top.select("protein"))
    size = os.path.getsize(out_xtc) / 1e6
    print(f"  lig{lig}: {xyz.shape[0]} frames, {n_prot}+{top.n_atoms - n_prot} "
          f"atoms, {size:.0f} MB", flush=True)
    return dict(lig=lig, frames=int(xyz.shape[0]), n_protein=n_prot,
                n_ligand=int(top.n_atoms - n_prot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--outroot", default=OUT)
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--ligs", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = args.jobs or sorted(
        d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    for job in jobs:
        outdir = os.path.join(args.outroot, job)
        os.makedirs(outdir, exist_ok=True)
        ligs = args.ligs or sorted(
            int(re.search(r"lig(\d+)_xyz", os.path.basename(f)).group(1))
            for f in glob.glob(os.path.join(args.src, job, "lig*_xyz.npy")))
        print(f"\n=== {job}: {len(ligs)} ligands -> {outdir}")
        rows = [r for lig in ligs
                if (r := convert(job, lig, outdir, args.force)) is not None]
        if rows:
            with open(os.path.join(outdir, "README.txt"), "w") as fh:
                fh.write(f"{job}\n{JOB_NOTE.get(job, '')}\n\n")
                fh.write("Each lig{N}.xtc holds the receptor conformation that was\n"
                         "docked against plus the best-of-20 docked pose, one frame\n"
                         "per docked frame. Load with:\n\n"
                         "    import mdtraj as md\n"
                         "    t = md.load('lig38.xtc', top='lig38.pdb')\n\n"
                         "Protein residues are numbered 121-140; the ligand is the\n"
                         "last residue. lig{N}_frames.txt lists the source MD frame\n"
                         "id of every trajectory frame, in order.\n\n")
                fh.write("ligand\tframes\tprotein_atoms\tligand_atoms\n")
                for r in rows:
                    fh.write(f"{r['lig']}\t{r['frames']}\t{r['n_protein']}\t"
                             f"{r['n_ligand']}\n")
            print(f"  wrote README.txt ({len(rows)} ligands)")


if __name__ == "__main__":
    main()
