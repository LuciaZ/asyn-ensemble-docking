#!/usr/bin/env python3
# ============================================================================
# gen_configs_remaining.py -- Vina-GPU configs + work manifest for the
#                             ligands not yet docked
#
# Writes one config per (ligand, frame, residue box) and a manifest listing
# every (ligand, frame) pair. The SLURM script indexes the manifest by line
# rather than globbing the config tree: on dartfs, listing a directory with
# 10^5 entries stalls for minutes, while reading N lines of a text file is
# instant. Same reason the earlier gap-fill used missing_configs.txt.
#
# Box geometry matches the pilot exactly so the new ligands are comparable:
# 20 boxes per frame centred on each residue's centre of mass, uniform
# 18.75 A edge (npts=50), num_modes=20, best-of-20 collapse afterwards.
#
# CRITICAL: Vina-GPU in single-ligand mode ignores `output_directory` and
# writes next to the ligand file, so every box would overwrite the same
# output. Each config therefore carries an explicit unique `out =`.
#
# Run on the cluster:
#   python gen_configs_remaining.py --mode apo  --list-only
#   python gen_configs_remaining.py --mode apo
#   python gen_configs_remaining.py --mode holo
# ============================================================================
import os, re, glob, argparse
# Pure stdlib: numpy is absent from the login node's base env and only a mean is needed here

BASE_APO  = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_remaining"
BASE_HOLO = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_remaining"
LIG_PDBQT = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/autodock_vina/ligands_pdbqt"
APO_FRAMES = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/autodock_vina/protein_apo_pdbqt"
APO_TMPL   = "protein_only_frame_{fid:05d}_fasudil_apo.pdbqt"
HOLO_RECEPTORS = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_remaining/holo_receptors"

# already docked: the 10-ligand pilot plus the three published ligands
DONE = {38, 28, 5, 34, 24, 3, 17, 53, 9, 40, 23, 41, 47}

BOX_EDGE  = 18.75          # A, uniform across ligands (pilot decision)
N_RES     = 20             # D121..A140
NUM_MODES = 20
THREAD    = 8000
APO_STRIDE = 4             # 20020 frames -> 5005


def available_ligands(lig_dir):
    out = []
    for f in glob.glob(os.path.join(lig_dir, "*_charged.pdbqt")):
        m = re.match(r"(\d+)_charged\.pdbqt$", os.path.basename(f))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def residue_coms(pdb_path):
    """Per-residue centre of mass for the 20 peptide residues, in Angstrom.
    Read straight from the PDB text: mdtraj is not always importable in the
    cluster's base env, and this only needs coordinates."""
    res = {}
    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
            if el == "H":
                continue
            try:
                rs = int(ln[22:26])
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            res.setdefault(rs, []).append(xyz)
    keys = sorted(res)[:N_RES]
    out = []
    for k in keys:
        pts = res[k]
        n = len(pts)
        out.append((sum(p[0] for p in pts) / n,
                    sum(p[1] for p in pts) / n,
                    sum(p[2] for p in pts) / n))
    return out


def write_config(path, receptor, ligand, out_pdbqt, centre):
    with open(path, "w") as f:
        f.write(f"receptor = {receptor}\n")
        f.write(f"ligand = {ligand}\n")
        f.write(f"out = {out_pdbqt}\n")
        f.write(f"center_x = {centre[0]:.3f}\n")
        f.write(f"center_y = {centre[1]:.3f}\n")
        f.write(f"center_z = {centre[2]:.3f}\n")
        f.write(f"size_x = {BOX_EDGE}\nsize_y = {BOX_EDGE}\nsize_z = {BOX_EDGE}\n")
        f.write(f"thread = {THREAD}\nnum_modes = {NUM_MODES}\n")


def apo_frames():
    fids = []
    for f in glob.glob(os.path.join(APO_FRAMES, "protein_only_frame_*.pdbqt")):
        m = re.search(r"frame_(\d+)_", os.path.basename(f))
        if m:
            fids.append(int(m.group(1)))
    fids.sort()
    return [f for f in fids if f % APO_STRIDE == 0]


def holo_frames(lig):
    d = os.path.join(HOLO_RECEPTORS, f"lig{lig}")
    if not os.path.isdir(d):
        return []
    out = []
    with os.scandir(d) as it:
        for e in it:
            m = re.match(r"frame_(\d+)\.pdbqt$", e.name)
            if m:
                out.append(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["apo", "holo"], required=True)
    ap.add_argument("--ligs", nargs="*", type=int, default=None,
                    help="default: every ligand with a PDBQT, minus those "
                         "already docked")
    ap.add_argument("--base", default=None)
    ap.add_argument("--lig-dir", default=LIG_PDBQT)
    ap.add_argument("--list-only", action="store_true",
                    help="report the ligand list and box count, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="rewrite configs that already exist")
    args = ap.parse_args()

    base = args.base or (BASE_APO if args.mode == "apo" else BASE_HOLO)
    ligs = args.ligs
    if ligs is None:
        ligs = [l for l in available_ligands(args.lig_dir) if l not in DONE]
    ligs = sorted(ligs)
    print(f"mode={args.mode}  {len(ligs)} ligands: {ligs}")

    if args.mode == "apo":
        frames = apo_frames()
        if not frames:
            print(f"no apo frames found in {APO_FRAMES}"); return
        print(f"apo frames: {len(frames)} (stride {APO_STRIDE}, "
              f"{frames[0]}..{frames[-1]})")
        per_lig = {l: frames for l in ligs}
    else:
        per_lig = {l: holo_frames(l) for l in ligs}
        empty = [l for l, v in per_lig.items() if not v]
        if empty:
            print(f"NO RECEPTORS for ligand(s) {empty} -- run the holo "
                  f"receptor extraction first")
            ligs = [l for l in ligs if per_lig[l]]
        for l in ligs[:3]:
            print(f"  lig{l}: {len(per_lig[l])} receptor frames")

    total = sum(len(per_lig[l]) for l in ligs) * N_RES
    print(f"total boxes: {total:,}")
    hrs = total / 35000.0
    print(f"at ~35,000 boxes/hr (one 10-GPU node, batched): "
          f"{hrs:.0f} h = {hrs/24:.1f} days")
    if args.list_only:
        return

    cfg_root = os.path.join(base, "configs")
    raw_root = os.path.join(base, "outputs_vinagpu", "_raw")
    os.makedirs(cfg_root, exist_ok=True)
    os.makedirs(os.path.join(base, "done"), exist_ok=True)
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)

    manifest = os.path.join(base, f"manifest_{args.mode}.txt")
    n_cfg = 0
    with open(manifest, "w") as mf:
        for lig in ligs:
            ligand = os.path.join(args.lig_dir, f"{lig}_charged.pdbqt")
            if not os.path.exists(ligand):
                print(f"  lig{lig}: no ligand PDBQT, skipped"); continue
            os.makedirs(os.path.join(raw_root, f"lig{lig}"), exist_ok=True)
            for fid in per_lig[lig]:
                if args.mode == "apo":
                    rec = os.path.join(APO_FRAMES, APO_TMPL.format(fid=fid))
                    tag = f"{fid:05d}"
                else:
                    rec = os.path.join(HOLO_RECEPTORS, f"lig{lig}",
                                       f"frame_{fid:06d}.pdbqt")
                    tag = f"{fid:06d}"
                if not os.path.exists(rec):
                    continue
                fdir = os.path.join(cfg_root, f"lig{lig}", f"frame_{tag}")
                os.makedirs(fdir, exist_ok=True)
                coms = None
                for res in range(1, N_RES + 1):
                    cfg = os.path.join(fdir, f"res{res:02d}.txt")
                    if os.path.exists(cfg) and not args.force:
                        continue
                    if coms is None:
                        coms = residue_coms(rec)
                        if len(coms) < N_RES:
                            print(f"  lig{lig} frame {tag}: only "
                                  f"{len(coms)} residues, skipped")
                            break
                    out_pdbqt = os.path.join(
                        raw_root, f"lig{lig}",
                        f"frame_{tag}_res{res:02d}_out.pdbqt")
                    write_config(cfg, rec, ligand, out_pdbqt, coms[res - 1])
                    n_cfg += 1
                mf.write(f"{lig} {tag}\n")
            print(f"  lig{lig}: {len(per_lig[lig])} frames queued", flush=True)

    with open(manifest) as f:
        n_units = sum(1 for _ in f)
    print(f"\nwrote {n_cfg:,} new configs")
    print(f"manifest: {manifest}  ({n_units:,} ligand-frame units)")
    print(f"submit with:  sbatch --export=ALL,MODE={args.mode},"
          f"BASE={base} dock_one_node.slurm")


if __name__ == "__main__":
    main()
