#!/usr/bin/env python3
# ============================================================================
# analyze_pilot.py (v2) -- per-frame fingerprints + dual contact
#
# Change vs v1: fingerprints are saved as per-frame matrices (n_frames, 20)
# instead of pre-averaged (20,), so they can be fed to docking_tools.plotting
# .plot_ensemble_ii (which expects (n_cluster, n_frames, n_res) and does the
# averaging itself) and so dual contact (20,20) can be computed. Uniform
# ensemble (decision B): the whole 5005-frame set is treated as ONE cluster
# with weight 1.0 at plot time.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python analyze_pilot.py --smoke --ligs 38     # quick check
#   python analyze_pilot.py                         # full 10 ligands
# ============================================================================
import os, re, sys, glob, argparse, tempfile, shutil
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/zhihanz/data/scripts")
from docking_tools import trajectory_analysis   # noqa

# HOLO: winners come from the holo pilot; receptor is each pose's OWN holo
# bound-frame receptor (per-ligand, per-frame PDBQT), not a shared apo PDB.
WINNERS_DIR = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/winners"
RECEPTOR_ROOT = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/holo_receptors_pdb"
RECEPTOR_TMPL = "lig{lig}/frame_{fid:06d}.pdb"   # all-atom PDB (H intact)
OUT_ROOT    = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/analysis_v2"

PILOT_LIGS  = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]
N_RES_PROT  = 20
CONTACT_NM  = 0.6
HYDROGEN    = {"H"}

def read_winner_models(pdbqt_path, csv_path):
    fids = []
    with open(csv_path) as fh:
        next(fh)
        for ln in fh:
            fids.append(int(ln.split(",")[0]))
    models, cur, in_model = [], [], False
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                in_model, cur = True, []; continue
            if line.startswith("ENDMDL"):
                models.append(cur); in_model = False; continue
            if in_model:
                cur.append(line)
    n = min(len(models), len(fids))
    return list(zip(fids[:n], models[:n]))

def _pdbqt_atom_to_pdb(line):
    """Convert a receptor PDBQT ATOM/HETATM line to a plain PDB ATOM line.
    PDBQT has extra trailing columns (charge + AutoDock type); PDB stops at
    col 66 (occupancy/tempfactor) + element. Keep the fixed-width prefix and
    drop the AutoDock-specific tail so MDTraj parses it cleanly."""
    # columns 1-54 (record..z) are identical layout; keep through col 54,
    # then pad standard occupancy/bfactor and element guessed from atom name.
    head = line[:54]
    # element: PDBQT last token is AD type (e.g. 'NA','C','OA'); map to element
    ad_type = line[76:78].strip() if len(line) >= 78 else ""
    elem = ad_type[0] if ad_type else line[12:16].strip()[0]
    return f"{head}  1.00  0.00          {elem:>2}\n"

def build_complex_traj(lig, models, workdir, smoke=None):
    from openbabel import openbabel
    os.makedirs(workdir, exist_ok=True)
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("pdbqt", "pdb")
    merged_files, frames_used = [], []
    items = models[:smoke] if smoke else models
    for fid, atom_lines in items:
        # HOLO receptor: this ligand's own bound-frame receptor (PDBQT)
        rec = os.path.join(RECEPTOR_ROOT, RECEPTOR_TMPL.format(lig=lig, fid=fid))
        if not os.path.exists(rec):
            continue
        # ligand pose: pdbqt -> pdb +H
        lig_pdbqt = os.path.join(workdir, f"lig_{fid:06d}.pdbqt")
        lig_pdb   = os.path.join(workdir, f"lig_{fid:06d}.pdb")
        with open(lig_pdbqt, "w") as f:
            f.writelines(atom_lines)
        mol = openbabel.OBMol()
        conv.ReadFile(mol, lig_pdbqt); mol.AddHydrogens()
        conv.WriteFile(mol, lig_pdb)
        # receptor: convert protein PDBQT ATOM lines -> PDB style
        with open(rec) as f:
            pro = [ln for ln in f if ln.startswith(("ATOM", "HETATM"))]
        with open(lig_pdb) as f:
            lg = [ln for ln in f if ln.startswith(("ATOM", "HETATM"))]
        merged = os.path.join(workdir, f"cx_{fid:06d}.pdb")
        with open(merged, "w") as f:
            f.writelines(pro); f.write("TER\n"); f.writelines(lg); f.write("TER\nEND\n")
        merged_files.append(merged); frames_used.append(fid)
    if not merged_files:
        return None, []
    return md.load(merged_files), frames_used

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

def analyze_ligand(lig, smoke=None):
    pdbqt = os.path.join(WINNERS_DIR, f"lig{lig}_winners.pdbqt")
    csv   = os.path.join(WINNERS_DIR, f"lig{lig}_winners.csv")
    print(f"\n=== lig {lig} ===", flush=True)
    models = read_winner_models(pdbqt, csv)
    print(f"  {len(models)} winner poses", flush=True)
    workdir = tempfile.mkdtemp(prefix=f"lig{lig}_", dir="/tmp")
    try:
        traj, fids = build_complex_traj(lig, models, workdir, smoke=smoke)
        if traj is None:
            print("  no complexes built"); return None
        nf = traj.n_frames
        print(f"  complex traj: {nf} frames, {traj.n_atoms} atoms", flush=True)
        aro, donors, charges, lig_resid = detect_groups(traj)
        print(f"  aro={aro} donors={donors} charges={charges}", flush=True)
        _, heavy = ligand_residue(traj.top)

        bf = bound_fraction_uniform(traj, heavy, N_RES_PROT, CONTACT_NM)

        hphob_pf  = perframe("hphob", trajectory_analysis.hphob_contacts, nf,
                             trj=traj, ligand_residue_index=lig_resid)
        aro_pf    = perframe("aro", trajectory_analysis.aro_contacts, nf,
                             trj=traj, ligand_rings=aro) if aro else np.zeros((nf, N_RES_PROT))
        hbond_pf  = perframe("hbond", trajectory_analysis.hbond, nf,
                             trj=traj, ligand_residue_index=lig_resid,
                             lig_hbond_donors=donors) if donors else np.zeros((nf, N_RES_PROT))
        charge_pf = perframe("charge", trajectory_analysis.charge_contacts, nf,
                             trj=traj, Ligand_Pos_Charges=charges) if charges else np.zeros((nf, N_RES_PROT))

        # dual contact: force fallback ligand-select with an out-of-range idx
        try:
            dual = np.asarray(trajectory_analysis.dual_contact(traj, residue_idx=99999), dtype=float)
            dual = dual[:N_RES_PROT, :N_RES_PROT]
        except Exception as e:
            print(f"    dual ERR: {e}"); dual = np.zeros((N_RES_PROT, N_RES_PROT))

        print(f"  sums: bf={bf.sum():.2f} hphob={hphob_pf.mean(0).sum():.2f} "
              f"aro={aro_pf.mean(0).sum():.2f} hbond={hbond_pf.mean(0).sum():.2f} "
              f"charge={charge_pf.mean(0).sum():.2f} dual={dual.sum():.2f}", flush=True)

        return dict(lig=lig, n_frames=nf, bf=bf,
                    aro_groups=aro, donors=donors, charges=charges,
                    hphob=hphob_pf, aro=aro_pf, hbond=hbond_pf, charge=charge_pf,
                    dual=dual)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=PILOT_LIGS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    smoke = 50 if args.smoke else None
    for lig in args.ligs:
        r = analyze_ligand(lig, smoke=smoke)
        if r is None:
            continue
        # save: per-frame matrices + reshaped (1, nf, 20) for plot_ensemble_ii
        np.savez(os.path.join(OUT_ROOT, f"lig{lig}_fingerprint.npz"),
                 lig=r["lig"], n_frames=r["n_frames"], bf=r["bf"],
                 hphob=r["hphob"], aro=r["aro"], hbond=r["hbond"],
                 charge=r["charge"], dual=r["dual"],
                 aro_groups=np.array(r["aro_groups"], dtype=object),
                 donors=np.array(r["donors"], dtype=object),
                 charges=np.array(r["charges"], dtype=object))
        print(f"  saved lig{lig}_fingerprint.npz", flush=True)
    print(f"\nnpz -> {OUT_ROOT}/lig*_fingerprint.npz")

if __name__ == "__main__":
    main()
