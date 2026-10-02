#!/usr/bin/env python3
# ============================================================================
# md_reference_fingerprint.py -- MD-ensemble interaction fingerprints
#
# Produces the "MD Ensemble" reference panel (JCIM Fig 7A left / Fig 8A left):
# per-residue hydrophobic / aromatic / H-bond / charge contact probabilities
# and the dual-residue contact matrix, computed from the LIGAND'S OWN MD
# POSITION rather than from a docked pose. Everything downstream -- the
# docked-vs-MD correlation tables, <score> vs bound-fraction plots, the
# three-panel MD/holo/apo comparison -- is measured against this.
#
# Frames: the SAME frames the docking used, so the comparison is paired.
#   --frames selected  (default)  read bound_out/selected_frame_idx.npy
#                                 = the 5000 stride-sampled bound frames
#   --frames stride --nframes N   stride-sample N frames instead
#                                 (--pool bound|all chooses the pool)
#
# The trajectory is sliced to protein + ligand before analysis. This matters:
# the raw topology may retain ions, and docking_tools.dual_contact's fallback
# selector is "not protein and not water and mass>2", which would pick up
# Na/Cl as ligand atoms. Slicing also puts the 20 protein residues first so
# the [:, :20] per-residue slicing is valid, and makes the topology mirror
# the docked complex (20 residues + 1 ligand) so the two npz files are
# directly comparable.
#
# Output npz keys match analyze_holo_v2.py exactly (bf, hphob, aro, hbond,
# charge, dual, aro_groups, donors, charges) so existing comparison and
# plotting code needs no changes.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python md_reference_fingerprint.py --smoke --ligs 38
#   python md_reference_fingerprint.py                      # 10 pilot ligands
#   python md_reference_fingerprint.py --ligs 23 41 47 \
#          --frames stride --nframes 20020 --pool all
# ============================================================================
import os, re, sys, glob, argparse
import numpy as np
import mdtraj as md

sys.path.insert(0, "/home/zhihanz/data/scripts")
from docking_tools import trajectory_analysis

MDROOT   = "/home/zhihanz/data/MDdata"
OUT_ROOT = "/home/zhihanz/data/VinaGPU/hpc_results/md_reference"
PILOT    = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

N_RES_PROT = 20            # D121..A140
CONTACT_NM = 0.6           # 6 A, JCIM bound definition
HYDROGEN   = {"H"}
CHUNK      = 5000

STD_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
             "SER THR TRP TYR VAL HID HIE HIP HISE HISD HISP".split())
ION    = set("NA CL K SOD CLA NA+ CL- K+ MG CA ZN".split())
WATER  = set("HOH WAT TIP3 TIP4 TIP3P TIP4P SOL T3P T4P".split())


# ---------------------------------------------------------------- topology ---
def lig_to_desres(mdroot):
    """ligand id -> DESRES id, parsed from the tar names (leading number)."""
    m = {}
    for f in (glob.glob(os.path.join(mdroot, "*.tar"))
              + glob.glob(os.path.join(mdroot, "*.tar.gz"))):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)",
                      os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)
    return m


def desres_dir_for(desres_id, mdroot):
    hits = glob.glob(os.path.join(
        mdroot, f"DESRES-Trajectory_biorxiv2021-{desres_id}-no-water-glue",
        "*", "top_from_local_mae.pdb"))
    return os.path.dirname(hits[0]) if hits else None


def classify(top):
    """Protein-residue atom indices (in residue order) and ligand atoms,
    by residue-name exclusion -- the ligand is named '<1>' so it can't be
    selected by name."""
    prot_atoms, lig_atoms, lig_names = [], [], set()
    n_prot_res = 0
    for r in top.residues:
        nm = r.name.upper()
        idx = [a.index for a in r.atoms]
        heavy = [a.index for a in r.atoms
                 if a.element is not None and a.element.symbol != "H"]
        if nm in STD_AA:
            prot_atoms.extend(idx); n_prot_res += 1
        elif nm in ION or nm in WATER:
            continue
        elif len(heavy) >= 6:
            lig_atoms.extend(idx); lig_names.add(r.name)
    return (np.array(prot_atoms, int), np.array(lig_atoms, int),
            lig_names, n_prot_res)


# ------------------------------------------------------------ frame picking ---
def pick_frames(desres_dir, mode, nframes, pool):
    """Return the frame indices to analyse, or None to mean 'all'."""
    if mode == "selected":
        p = os.path.join(desres_dir, "bound_out", "selected_frame_idx.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no selected_frame_idx.npy in {desres_dir}"
                                    f"/bound_out -- run analyze_bound.py first,"
                                    f" or use --frames stride")
        return np.sort(np.load(p))

    # stride mode
    if pool == "bound":
        p = os.path.join(desres_dir, "bound_out", "bound_frame_idx.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no bound_frame_idx.npy in {desres_dir}"
                                    f"/bound_out -- needed for --pool bound")
        cand = np.load(p)
    else:                                    # pool == "all"
        n_tot = md.load(os.path.join(desres_dir, "traj_all.dcd"),
                        top=os.path.join(desres_dir, "top_from_local_mae.pdb"),
                        frame=0).n_frames    # cheap; real count comes below
        # count frames without loading coordinates for all atoms
        n_tot = 0
        top_pdb = os.path.join(desres_dir, "top_from_local_mae.pdb")
        for ch in md.iterload(os.path.join(desres_dir, "traj_all.dcd"),
                              top=top_pdb, chunk=CHUNK, atom_indices=[0]):
            n_tot += ch.n_frames
        cand = np.arange(n_tot)

    if cand.size <= nframes:
        print(f"    NOTE: pool has {cand.size} frames (<= {nframes}); taking all")
        return np.sort(cand)
    step = cand.size / nframes
    return np.sort(cand[(np.arange(nframes) * step).astype(int)])


def load_frames(desres_dir, want_idx, keep_atoms):
    """Stream the DCD, keep only the wanted frames and only protein+ligand
    atoms. Streaming avoids holding a 333k-frame trajectory in memory."""
    top_pdb = os.path.join(desres_dir, "top_from_local_mae.pdb")
    dcd     = os.path.join(desres_dir, "traj_all.dcd")
    want = set(int(i) for i in want_idx)
    pieces, seen = [], 0
    for chunk in md.iterload(dcd, top=top_pdb, chunk=CHUNK,
                             atom_indices=keep_atoms):
        local = [i for i in range(chunk.n_frames) if (seen + i) in want]
        if local:
            pieces.append(chunk.slice(local))
        seen += chunk.n_frames
        if seen % (CHUNK * 10) == 0:
            print(f"    ...scanned {seen} frames", flush=True)
        if seen > max(want):          # every wanted frame already collected
            break
    if not pieces:
        return None
    traj = pieces[0] if len(pieces) == 1 else pieces[0].join(pieces[1:])
    return traj


# -------------------------------------------------- fingerprints (as docked) ---
def ligand_residue(top):
    res = list(top.residues)[-1]
    heavy = [a.index for a in res.atoms if a.element.symbol not in HYDROGEN]
    return res.index, np.array(heavy, int)


def bound_fraction_uniform(traj, lig_heavy, n_res, cutoff_nm):
    bf = np.zeros(n_res)
    lig_xyz = traj.xyz[:, lig_heavy, :]
    for ri, res in enumerate(traj.top.residues):
        if ri >= n_res or not res.is_protein:
            continue
        pidx = [a.index for a in res.atoms]
        pxyz = traj.xyz[:, pidx, :]
        diff = pxyz[:, :, None, :] - lig_xyz[:, None, :, :]
        mind = np.sqrt((diff**2).sum(-1)).reshape(traj.n_frames, -1).min(1)
        bf[ri] = (mind < cutoff_nm).mean()
    return bf


def detect_groups(traj):
    """Ring / donor / charge detection -- same logic as the docked-pose
    analysis so the two ensembles get identical group definitions."""
    import itertools, networkx as nx
    top = traj.topology
    lig_resid, _ = ligand_residue(top)
    res = list(top.residues)[lig_resid]
    idx = [a.index for a in res.atoms]
    elem = {a.index: a.element.symbol for a in res.atoms}
    xyz = traj.xyz[0] * 10.0
    adj = {i: set() for i in idx}
    for i, j in itertools.combinations(idx, 2):
        d = np.linalg.norm(xyz[i] - xyz[j])
        cut = 1.30 if ("H" in (elem[i], elem[j])) else 1.85
        if d < cut:
            adj[i].add(j); adj[j].add(i)
    heavy = [i for i in idx if elem[i] != "H"]
    G = nx.Graph()
    for i in heavy:
        for j in adj[i]:
            if elem[j] != "H":
                G.add_edge(i, j)
    rings = [sorted(c) for c in nx.minimum_cycle_basis(G) if 4 <= len(c) <= 6]

    def planar(a):
        P = xyz[a]; Q = P - P.mean(0)
        n = np.linalg.svd(Q)[2][-1]
        return float(np.sqrt(((Q @ n)**2).mean()))

    aro = [(planar(r), r) for r in rings if planar(r) < 0.10]
    aro_ring = [min(aro, key=lambda x: x[0])[1]] if aro else None
    donors, charges = [], []
    for n in [i for i in heavy if elem[i] == "N"]:
        h = [j for j in adj[n] if elem[j] == "H"]
        bonded_S = any(elem[j] == "S" for j in adj[n])
        in_aro = bool(aro_ring and n in aro_ring[0])
        donors += [[n, hh] for hh in h]
        if h and not bonded_S and not in_aro:
            charges.append(n)
    return aro_ring, (donors or None), (charges or None), lig_resid


def perframe(name, fn, n_frames, **kw):
    try:
        arr = np.asarray(fn(**kw), dtype=float)
        if arr.ndim == 2:
            return arr[:, :N_RES_PROT]
        print(f"    {name}: unexpected ndim {arr.ndim}, zeros")
        return np.zeros((n_frames, N_RES_PROT))
    except Exception as e:
        print(f"    {name} ERR: {e}")
        return np.zeros((n_frames, N_RES_PROT))


# ---------------------------------------------------------------- per ligand ---
def run_ligand(lig, desres_id, args):
    print(f"\n=== lig {lig} (DESRES {desres_id}) ===", flush=True)
    d = desres_dir_for(desres_id, args.mdroot)
    if d is None:
        print("  trajectory dir not found, skip"); return None

    top = md.load_topology(os.path.join(d, "top_from_local_mae.pdb"))
    prot_atoms, lig_atoms, lig_names, n_prot_res = classify(top)
    if lig_atoms.size == 0:
        print("  no ligand found (apo trajectory?), skip"); return None
    if n_prot_res != N_RES_PROT:
        print(f"  WARNING: {n_prot_res} protein residues, expected {N_RES_PROT}")
    print(f"  topology: {top.n_atoms} atoms; protein {prot_atoms.size} "
          f"({n_prot_res} res); ligand {lig_atoms.size} {lig_names}")

    idx = pick_frames(d, args.frames, args.nframes, args.pool)
    if args.smoke:
        idx = idx[:50]
    print(f"  frames: {idx.size} (range {idx.min()}-{idx.max()})", flush=True)

    # protein first, then ligand -> ligand is the last residue, no ions present
    keep = np.concatenate([prot_atoms, lig_atoms])
    traj = load_frames(d, idx, keep)
    if traj is None:
        print("  no frames loaded, skip"); return None
    print(f"  traj: {traj.n_frames} frames, {traj.n_atoms} atoms, "
          f"{traj.n_residues} residues", flush=True)

    aro, donors, charges, lig_resid = detect_groups(traj)
    print(f"  aro={aro} donors={donors} charges={charges}", flush=True)
    _, heavy = ligand_residue(traj.top)
    nf = traj.n_frames

    bf = bound_fraction_uniform(traj, heavy, N_RES_PROT, CONTACT_NM)
    hphob_pf = perframe("hphob", trajectory_analysis.hphob_contacts, nf,
                        trj=traj, ligand_residue_index=lig_resid)
    aro_pf   = (perframe("aro", trajectory_analysis.aro_contacts, nf,
                         trj=traj, ligand_rings=aro)
                if aro else np.zeros((nf, N_RES_PROT)))
    hbond_pf = (perframe("hbond", trajectory_analysis.hbond, nf,
                         trj=traj, ligand_residue_index=lig_resid,
                         lig_hbond_donors=donors)
                if donors else np.zeros((nf, N_RES_PROT)))
    chg_pf   = (perframe("charge", trajectory_analysis.charge_contacts, nf,
                         trj=traj, Ligand_Pos_Charges=charges)
                if charges else np.zeros((nf, N_RES_PROT)))
    try:
        dual = np.asarray(trajectory_analysis.dual_contact(traj, residue_idx=99999),
                          dtype=float)[:N_RES_PROT, :N_RES_PROT]
    except Exception as e:
        print(f"    dual ERR: {e}"); dual = np.zeros((N_RES_PROT, N_RES_PROT))

    print(f"  sums: bf={bf.sum():.2f} hphob={hphob_pf.mean(0).sum():.2f} "
          f"aro={aro_pf.mean(0).sum():.2f} hbond={hbond_pf.mean(0).sum():.2f} "
          f"charge={chg_pf.mean(0).sum():.2f} dual={dual.sum():.2f}", flush=True)

    return dict(lig=lig, n_frames=nf, frame_idx=idx, bf=bf,
                hphob=hphob_pf, aro=aro_pf, hbond=hbond_pf, charge=chg_pf,
                dual=dual, aro_groups=aro, donors=donors, charges=charges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=PILOT)
    ap.add_argument("--mdroot", default=MDROOT)
    ap.add_argument("--outdir", default=OUT_ROOT)
    ap.add_argument("--frames", choices=["selected", "stride"], default="selected",
                    help="'selected' = the frames docking used (default)")
    ap.add_argument("--nframes", type=int, default=20020,
                    help="stride mode: how many frames")
    ap.add_argument("--pool", choices=["bound", "all"], default="all",
                    help="stride mode: sample from bound frames or all frames")
    ap.add_argument("--smoke", action="store_true", help="first 50 frames only")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mapping = lig_to_desres(args.mdroot)
    missing = [l for l in args.ligs if l not in mapping]
    if missing:
        print(f"WARNING: no DESRES id for ligand(s) {missing}")

    for lig in args.ligs:
        if lig not in mapping:
            continue
        try:
            r = run_ligand(lig, mapping[lig], args)
        except Exception as e:
            print(f"  lig{lig} FAILED: {e}")
            continue
        if r is None:
            continue
        np.savez(os.path.join(args.outdir, f"lig{lig}_md_fingerprint.npz"),
                 lig=r["lig"], n_frames=r["n_frames"], frame_idx=r["frame_idx"],
                 bf=r["bf"], hphob=r["hphob"], aro=r["aro"], hbond=r["hbond"],
                 charge=r["charge"], dual=r["dual"],
                 aro_groups=np.array(r["aro_groups"], dtype=object),
                 donors=np.array(r["donors"], dtype=object),
                 charges=np.array(r["charges"], dtype=object))
        print(f"  saved lig{lig}_md_fingerprint.npz", flush=True)

    print(f"\nnpz -> {args.outdir}/lig*_md_fingerprint.npz")


if __name__ == "__main__":
    main()

