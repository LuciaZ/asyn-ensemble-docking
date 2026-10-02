#!/usr/bin/env python3
# ============================================================================
# diagnose_best_matched.py -- why does best-matched RMSD differ so much
#                             between ligands of the same size?
#
# holo_rmsd.py gave best-matched floors from 0.94 A (lig3) to 4.21 A (lig9)
# for chemically similar, equally sized ligands. Protein-superposition
# quality, symmetry-mapping count, single-pair conformational difference and
# centroid placement were each checked and are near-identical between lig9
# and lig40, so none of them explains it. This decomposes the metric instead
# of guessing further.
#
# For a subsample of docked x reference pose pairs it reports:
#
#   protein-frame RMSD   what holo_rmsd.py measures: superpose the proteins,
#                        then compare ligands in place
#   conformational RMSD  superpose each ligand pair onto ITSELF (Kabsch on the
#                        ligand), discarding position and orientation. The
#                        minimum over all pairs is the CONFORMATIONAL FLOOR --
#                        no amount of correct placement can beat it.
#   gap                  protein-frame minus conformational, i.e. the part
#                        attributable to position/orientation
#
# If floor ~= protein-frame minimum, the ligand simply cannot be reproduced
# conformationally and the metric says nothing about binding-site accuracy.
# If floor << protein-frame minimum, docking places the ligand somewhere the
# MD never puts it, and the number is meaningful.
#
# Also writes PyMOL-ready PDBs (chain A protein, B docked, C MD same-frame,
# D best-match MD) so the geometry can be seen rather than inferred.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python diagnose_best_matched.py --ligs 9 40
#   python diagnose_best_matched.py --ligs 9 40 --n-dock 300 --n-ref 300
# ============================================================================
import os, re, glob, csv, argparse, itertools
import numpy as np
import mdtraj as md

WINNERS = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/winners"
MDROOT  = "/home/zhihanz/data/MDdata"
OUT     = "/home/zhihanz/data/VinaGPU/hpc_results/holo_rmsd/diagnose"

AD2EL = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
         "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
         "H": "H", "HD": "H", "HS": "H", "P": "P",
         "F": "F", "CL": "CL", "BR": "BR", "I": "I"}
STD_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
             "SER THR TRP TYR VAL HID HIE HIP HISE HISD HISP".split())
ION   = set("NA CL K SOD CLA NA+ CL- K+ MG CA ZN".split())
WATER = set("HOH WAT TIP3 TIP4 TIP3P TIP4P SOL T3P T4P".split())


def lig_to_desres(mdroot):
    m = {}
    for f in (glob.glob(os.path.join(mdroot, "*.tar"))
              + glob.glob(os.path.join(mdroot, "*.tar.gz"))):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)",
                      os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)
    return m


def read_winners(win_dir, lig, limit=None):
    csv_p = os.path.join(win_dir, f"lig{lig}_winners.csv")
    pdb_p = os.path.join(win_dir, f"lig{lig}_winners.pdbqt")
    with open(csv_p) as f:
        fids = [int(r["frame_id"]) for r in csv.DictReader(f)]
    models, names, cur, elems = [], None, [], None
    for line in open(pdb_p):
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith(("ATOM", "HETATM")):
            ad = line[77:79].strip().upper() if len(line) >= 79 else ""
            el = AD2EL.get(ad, ad[:1] if ad else line[12:16].strip()[:1].upper())
            if el == "H":
                continue
            cur.append((el, line[12:16].strip(),
                        float(line[30:38]), float(line[38:46]),
                        float(line[46:54])))
        elif line.startswith("ENDMDL"):
            if elems is None:
                elems = [c[0] for c in cur]
                names = [c[1] for c in cur]
            if len(cur) == len(elems):
                models.append([[c[2], c[3], c[4]] for c in cur])
            cur = []
            if limit and len(models) >= limit:
                break
    n = len(models)
    return np.array(fids[:n]), np.array(models, float), elems, names


def classify(top):
    prot, lig = [], []
    for r in top.residues:
        nm = r.name.upper()
        heavy = [a.index for a in r.atoms
                 if a.element is not None and a.element.symbol != "H"]
        if nm in STD_AA:
            prot.extend(heavy)
        elif nm in ION or nm in WATER:
            continue
        elif len(heavy) >= 6:
            lig.extend(heavy)
    return np.array(prot, int), np.array(lig, int)


def load_md(d, want, keep):
    top_pdb = os.path.join(d, "top_from_local_mae.pdb")
    dcd = os.path.join(d, "traj_all.dcd")
    wset = set(int(i) for i in want)
    hi = max(wset)
    pieces, ids, seen = [], [], 0
    for ch in md.iterload(dcd, top=top_pdb, chunk=5000, atom_indices=keep):
        loc = [i for i in range(ch.n_frames) if (seen + i) in wset]
        if loc:
            pieces.append(ch.xyz[loc] * 10.0)
            ids.extend(seen + i for i in loc)
        seen += ch.n_frames
        if seen > hi:
            break
    return (np.concatenate(pieces, 0), np.array(ids, int)) if pieces else (None, None)


def bond_graph(xyz, el):
    import networkx as nx
    G = nx.Graph()
    for i, e in enumerate(el):
        G.add_node(i, el=e)
    for i, j in itertools.combinations(range(len(el)), 2):
        if np.linalg.norm(xyz[i] - xyz[j]) < 1.95:
            G.add_edge(i, j)
    return G


def atom_mappings(dx, de, mx, me, cap=24):
    from networkx.algorithms import isomorphism as iso
    if sorted(de) != sorted(me):
        return []
    gm = iso.GraphMatcher(bond_graph(dx, de), bond_graph(mx, me),
                          node_match=lambda a, b: a["el"] == b["el"])
    out = []
    for m in gm.isomorphisms_iter():
        out.append(np.array([m[i] for i in range(len(de))], int))
        if len(out) >= cap:
            break
    return out


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def conf_rmsd(P, Q):
    """RMSD after optimal superposition of the pair -- conformation only."""
    pc, qc = P.mean(0), Q.mean(0)
    A, B = P - pc, Q - qc
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt((((A @ R.T) - B) ** 2).sum(-1).mean()))


def write_pdb(path, blocks):
    """blocks: [(chain, resname, resseq, [(name, element, xyz)])]"""
    with open(path, "w") as f:
        n = 0
        for chain, resn, resi, atoms in blocks:
            for nm, el, xyz in atoms:
                n += 1
                f.write(f"ATOM  {n:5d} {nm:<4s}{resn:>3s} {chain}{resi:4d}    "
                        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                        f"  1.00  0.00          {el:>2s}\n")
            f.write("TER\n")
        f.write("END\n")


def run(lig, desres_id, args):
    print(f"\n=== lig {lig} ===", flush=True)
    fids, dock, del_, dnames = read_winners(args.winners, lig,
                                            limit=args.n_dock)
    d = glob.glob(os.path.join(
        args.mdroot, f"DESRES-Trajectory_biorxiv2021-{desres_id}-no-water-glue",
        "*", "top_from_local_mae.pdb"))
    if not d:
        print("  no trajectory"); return None
    d = os.path.dirname(d[0])
    top = md.load_topology(os.path.join(d, "top_from_local_mae.pdb"))
    pidx, lidx = classify(top)
    mel = [top.atom(i).element.symbol.upper() for i in lidx]
    keep = np.concatenate([pidx, lidx]); npro = len(pidx)

    xyz, got = load_md(d, fids, keep)
    if xyz is None:
        print("  no frames"); return None
    pos = {f: k for k, f in enumerate(got)}
    sel = [(k, pos[f]) for k, f in enumerate(fids) if f in pos]
    dk = np.array([s[0] for s in sel]); mk = np.array([s[1] for s in sel])
    fids, dock = fids[dk], dock[dk]
    prot, mdlig = xyz[mk, :npro, :], xyz[mk, npro:, :]

    maps = atom_mappings(dock[0], del_, mdlig[0], mel)
    if not maps:
        print("  no mapping"); return None
    print(f"  {len(dock)} docked, {len(mdlig)} MD, {len(maps)} mapping(s)",
          flush=True)

    # common frame via protein superposition
    ref = prot[0]
    dref = np.empty_like(dock); mref = np.empty_like(mdlig)
    for i in range(len(fids)):
        R, t = kabsch(prot[i], ref)
        dref[i] = dock[i] @ R.T + t
        mref[i] = mdlig[i] @ R.T + t

    nref = min(args.n_ref, len(mref))
    step = max(1, len(mref) // nref)
    ridx = np.arange(0, len(mref), step)[:nref]
    refset = mref[ridx]

    # protein-frame best match
    best = np.full(len(fids), np.inf); bj = np.zeros(len(fids), int)
    for m in maps:
        D = dref[:, None, m, :]
        R_ = refset[None, :, :, :]
        block = np.sqrt(((D - R_) ** 2).sum(-1).mean(-1))
        j = block.argmin(1); v = block[np.arange(len(fids)), j]
        upd = v < best
        best[upd] = v[upd]; bj[upd] = j[upd]

    # conformational floor: free superposition, min over sampled pairs
    rng = np.random.default_rng(0)
    npair = min(args.n_pairs, len(fids) * len(refset))
    ii = rng.integers(0, len(fids), npair)
    jj = rng.integers(0, len(refset), npair)
    floor = np.inf
    for a, b in zip(ii, jj):
        for m in maps:
            floor = min(floor, conf_rmsd(dref[a], refset[b][m]))

    # conformational RMSD specifically for each frame's best-matching pair
    conf_at_best = np.array([
        min(conf_rmsd(dref[i], refset[bj[i]][m]) for m in maps)
        for i in range(0, len(fids), max(1, len(fids) // 200))])

    print(f"  protein-frame best-matched : min {best.min():.2f}  "
          f"median {np.median(best):.2f}")
    print(f"  conformational floor       : {floor:.2f}   "
          f"({npair} random pairs, free superposition)")
    print(f"  conformational at best pair: median {np.median(conf_at_best):.2f}")
    gap = np.median(best) - np.median(conf_at_best)
    print(f"  -> position/orientation gap: {gap:.2f} A "
          f"({100*gap/max(np.median(best),1e-9):.0f}% of the metric)",
          flush=True)

    # ---- export a few representatives for PyMOL ----
    os.makedirs(args.outdir, exist_ok=True)
    order = np.argsort(best)
    picks = [("best", order[0]), ("median", order[len(order) // 2]),
             ("worst", order[-1])]
    pnames = [top.atom(i).name for i in pidx]
    pels = [top.atom(i).element.symbol.upper() for i in pidx]
    mnames = [top.atom(i).name for i in lidx]
    for tag, i in picks:
        R, t = kabsch(prot[i], ref)
        blocks = [
            ("A", "ALA", 1, [(pnames[k], pels[k], prot[i][k] @ R.T + t)
                             for k in range(npro)]),
            ("B", "DOK", 1, [(dnames[k], del_[k], dref[i][k])
                             for k in range(len(del_))]),
            ("C", "MDS", 1, [(mnames[k], mel[k], mref[i][k])
                             for k in range(len(mel))]),
            ("D", "MDB", 1, [(mnames[k], mel[k], refset[bj[i]][k])
                             for k in range(len(mel))]),
        ]
        p = os.path.join(args.outdir,
                         f"lig{lig}_{tag}_frame{fids[i]}_rmsd{best[i]:.2f}.pdb")
        write_pdb(p, blocks)
        print(f"    wrote {os.path.basename(p)}")

    return dict(lig=lig, best_min=best.min(), best_med=np.median(best),
                floor=floor, conf_med=np.median(conf_at_best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=[9, 40])
    ap.add_argument("--winners", default=WINNERS)
    ap.add_argument("--mdroot", default=MDROOT)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--n-dock", type=int, default=500)
    ap.add_argument("--n-ref", type=int, default=500)
    ap.add_argument("--n-pairs", type=int, default=3000,
                    help="random pairs for the conformational floor")
    args = ap.parse_args()

    mapping = lig_to_desres(args.mdroot)
    rows = []
    for lig in args.ligs:
        if lig not in mapping:
            print(f"lig{lig}: no DESRES id"); continue
        try:
            r = run(lig, mapping[lig], args)
        except Exception as e:
            print(f"  lig{lig} FAILED: {type(e).__name__}: {e}"); continue
        if r:
            rows.append(r)

    if rows:
        print("\n=== decomposition ===")
        print(f"{'lig':>4} {'best min':>9} {'best med':>9} {'conf floor':>11} "
              f"{'conf@best':>10} {'verdict':>28}")
        for r in rows:
            v = ("conformation-limited" if r["floor"] > 0.8 * r["best_min"]
                 else "placement-limited")
            print(f"{r['lig']:>4} {r['best_min']:>9.2f} {r['best_med']:>9.2f} "
                  f"{r['floor']:>11.2f} {r['conf_med']:>10.2f} {v:>28}")
        print(f"\nPDBs -> {args.outdir}/  (chain A protein, B docked, "
              f"C MD same-frame, D MD best-match)")
        print("PyMOL:  load lig9_best_*.pdb; show sticks, chain B+C+D; "
              "color red, chain B; color grey, chain C; color yellow, chain D")


if __name__ == "__main__":
    main()
