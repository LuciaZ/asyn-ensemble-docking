#!/usr/bin/env python3
# ============================================================================
# holo_rmsd.py -- frame-matched and best-matched ligand RMSD (JCIM Fig 9 /
#                 Table 2 / Supp. Table 5)
#
# FRAME-MATCHED: docked pose in frame i vs the ligand's actual MD position in
# that same frame i. For holo docking the receptor WAS the MD protein of frame
# i, so both sit in one coordinate system and no superposition is applied.
# Answers: did docking find where the ligand really was in this frame?
#
# BEST-MATCHED: docked pose in frame i vs EVERY reference MD bound pose, take
# the minimum. Reference poses and docked poses are first brought into a
# common frame by superposing each frame's protein heavy atoms onto a
# reference frame's protein (Kabsch), then applying that transform to the
# ligand. Answers: does the docked pose at least resemble some pose the
# ligand genuinely adopts?
#
# ATOM ORDER is the main correctness trap. PDBQT atom order (Vina output) does
# not match the MD topology order, and comparing them element-by-index gives
# meaningless RMSD. Rather than hard-coding a permutation, the mapping is
# derived once per ligand by graph isomorphism on the heavy-atom connectivity
# (elements as node labels). All valid isomorphisms are enumerated and the
# minimum RMSD over them is taken, which simultaneously handles atom ordering
# AND topological symmetry (a flipped phenyl is the same pose).
#
# Outputs per ligand:
#   lig{N}_rmsd.npz     frame_ids, frame_matched, best_matched, best_match_ref
#   lig{N}_rmsd.png     the two KDEs with <3 A / <5 A percentages
#   rmsd_table.csv      Table 2 layout: "%<3A (%<5A)" per ligand
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python holo_rmsd.py --smoke --ligs 38
#   python holo_rmsd.py
#   python holo_rmsd.py --ref-stride 4        # quarter the reference set
# ============================================================================
import os, re, glob, csv, argparse, itertools
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINNERS = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/winners"
MDROOT  = "/home/zhihanz/data/MDdata"
OUT     = "/home/zhihanz/data/VinaGPU/hpc_results/holo_rmsd"
PILOT   = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

STD_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
             "SER THR TRP TYR VAL HID HIE HIP HISE HISD HISP".split())
ION   = set("NA CL K SOD CLA NA+ CL- K+ MG CA ZN".split())
WATER = set("HOH WAT TIP3 TIP4 TIP3P TIP4P SOL T3P T4P".split())
CHUNK = 5000

# AutoDock atom type -> element
AD2EL = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
         "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
         "H": "H", "HD": "H", "HS": "H", "P": "P",
         "F": "F", "CL": "CL", "BR": "BR", "I": "I",
         "MG": "MG", "MN": "MN", "ZN": "ZN", "CA": "CA", "FE": "FE"}


# ------------------------------------------------------------------ loading ---
def lig_to_desres(mdroot):
    m = {}
    for f in (glob.glob(os.path.join(mdroot, "*.tar"))
              + glob.glob(os.path.join(mdroot, "*.tar.gz"))):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)",
                      os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)
    return m


def desres_dir(desres_id, mdroot):
    hits = glob.glob(os.path.join(
        mdroot, f"DESRES-Trajectory_biorxiv2021-{desres_id}-no-water-glue",
        "*", "top_from_local_mae.pdb"))
    return os.path.dirname(hits[0]) if hits else None


def read_winners(win_dir, lig):
    """Docked heavy-atom coords per frame, in Angstrom.
    Returns (frame_ids, coords (n_frames, n_heavy, 3), elements)."""
    csv_p = os.path.join(win_dir, f"lig{lig}_winners.csv")
    pdb_p = os.path.join(win_dir, f"lig{lig}_winners.pdbqt")
    if not (os.path.exists(csv_p) and os.path.exists(pdb_p)):
        return None, None, None
    with open(csv_p) as f:
        fids = [int(r["frame_id"]) for r in csv.DictReader(f)]

    models, cur, elems = [], [], None
    for line in open(pdb_p):
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith(("ATOM", "HETATM")):
            ad = line[77:79].strip().upper() if len(line) >= 79 else ""
            # AutoDock types are not element symbols: A = aromatic carbon,
            # NA/OA/SA = H-bond-accepting N/O/S, HD = polar hydrogen. Taking
            # the first character turns every aromatic ring carbon into "A"
            # and breaks the element match against the MD topology.
            el = AD2EL.get(ad, ad[:1] if ad else
                           line[12:16].strip()[:1].upper())
            if el == "H":
                continue
            cur.append((el, float(line[30:38]), float(line[38:46]),
                        float(line[46:54])))
        elif line.startswith("ENDMDL"):
            if elems is None:
                elems = [c[0] for c in cur]
            if len(cur) == len(elems):
                models.append([[c[1], c[2], c[3]] for c in cur])
            cur = []
    if not models or len(models) != len(fids):
        print(f"    lig{lig}: {len(models)} MODELs vs {len(fids)} csv rows "
              f"-- cannot pair, skipping")
        return None, None, None
    return np.array(fids), np.array(models, float), elems


def classify(top):
    prot, lig = [], []
    for r in top.residues:
        nm = r.name.upper()
        idx = [a.index for a in r.atoms]
        heavy = [a.index for a in r.atoms
                 if a.element is not None and a.element.symbol != "H"]
        if nm in STD_AA:
            prot.extend(heavy)
        elif nm in ION or nm in WATER:
            continue
        elif len(heavy) >= 6:
            lig.extend(heavy)
    return np.array(prot, int), np.array(lig, int)


def load_md_frames(d, want, keep):
    """Stream the DCD; return coords (n, len(keep), 3) in Angstrom, ordered
    by sorted frame index, plus the frame ids actually found."""
    top_pdb = os.path.join(d, "top_from_local_mae.pdb")
    dcd = os.path.join(d, "traj_all.dcd")
    wset = set(int(i) for i in want)
    pieces, ids, seen = [], [], 0
    hi = max(wset) if wset else -1
    for chunk in md.iterload(dcd, top=top_pdb, chunk=CHUNK, atom_indices=keep):
        local = [i for i in range(chunk.n_frames) if (seen + i) in wset]
        if local:
            pieces.append(chunk.xyz[local] * 10.0)
            ids.extend(seen + i for i in local)
        seen += chunk.n_frames
        if seen > hi:
            break
    if not pieces:
        return None, None
    return np.concatenate(pieces, 0), np.array(ids, int)


# --------------------------------------------------------------- atom mapping ---
def bond_graph(coords, elements):
    import networkx as nx
    n = len(elements)
    G = nx.Graph()
    for i, e in enumerate(elements):
        G.add_node(i, el=e)
    for i, j in itertools.combinations(range(n), 2):
        if np.linalg.norm(coords[i] - coords[j]) < 1.95:   # heavy-heavy bond
            G.add_edge(i, j)
    return G


def atom_mappings(dock_xyz, dock_el, md_xyz, md_el, max_maps=24):
    """All isomorphisms docked->MD as index arrays. Handles both the PDBQT/MD
    ordering difference and topological symmetry in one step."""
    import networkx as nx
    from networkx.algorithms import isomorphism as iso
    if sorted(dock_el) != sorted(md_el):
        print(f"    element mismatch: docked {sorted(dock_el)} vs "
              f"MD {sorted(md_el)}")
        return []
    Gd = bond_graph(dock_xyz, dock_el)
    Gm = bond_graph(md_xyz, md_el)
    gm = iso.GraphMatcher(Gd, Gm,
                          node_match=lambda a, b: a["el"] == b["el"])
    maps = []
    for m in gm.isomorphisms_iter():
        maps.append(np.array([m[i] for i in range(len(dock_el))], int))
        if len(maps) >= max_maps:
            break
    return maps


# -------------------------------------------------------------------- kabsch ---
def kabsch(P, Q):
    """Rotation+translation taking P onto Q (both (n,3))."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rmsd_pairs(A, B):
    """A (n,k,3) vs B (n,k,3) rowwise -> (n,)"""
    return np.sqrt(((A - B) ** 2).sum(-1).mean(-1))


# ---------------------------------------------------------------- per ligand ---
def run_ligand(lig, desres_id, args):
    print(f"\n=== lig {lig} (DESRES {desres_id}) ===", flush=True)
    fids, dock_all, dock_el = read_winners(args.winners, lig)
    if fids is None:
        return None
    d = desres_dir(desres_id, args.mdroot)
    if d is None:
        print("  trajectory not found"); return None

    top = md.load_topology(os.path.join(d, "top_from_local_mae.pdb"))
    prot_idx, lig_idx = classify(top)
    md_el = [top.atom(i).element.symbol.upper() for i in lig_idx]
    print(f"  docked {dock_all.shape[1]} heavy atoms, MD {len(lig_idx)} heavy; "
          f"protein {len(prot_idx)} heavy", flush=True)

    if args.smoke:
        fids, dock_all = fids[:args.smoke_n], dock_all[:args.smoke_n]

    keep = np.concatenate([prot_idx, lig_idx])
    n_prot = len(prot_idx)
    md_xyz, got = load_md_frames(d, fids, keep)
    if md_xyz is None:
        print("  no MD frames loaded"); return None

    # keep only docked frames whose MD frame was recovered, same order
    pos = {f: k for k, f in enumerate(got)}
    sel = [(k, pos[f]) for k, f in enumerate(fids) if f in pos]
    if len(sel) < len(fids):
        print(f"  matched {len(sel)}/{len(fids)} frames to MD")
    dk = np.array([s[0] for s in sel]); mk = np.array([s[1] for s in sel])
    fids = fids[dk]; dock = dock_all[dk]
    prot = md_xyz[mk, :n_prot, :]
    mdlig = md_xyz[mk, n_prot:, :]

    maps = atom_mappings(dock[0], dock_el, mdlig[0], md_el)
    if not maps:
        print("  no atom mapping found -- RMSD would be meaningless, skipping")
        return None
    print(f"  {len(maps)} symmetry-equivalent atom mapping(s)", flush=True)

    # ---- frame-matched: same coordinate frame, no superposition ----
    fm = np.full(len(fids), np.inf)
    for m in maps:
        fm = np.minimum(fm, rmsd_pairs(dock, mdlig[:, m, :]))

    # ---- bring everything into one frame via protein superposition ----
    ref = prot[0]
    dock_ref = np.empty_like(dock)
    mdlig_ref = np.empty_like(mdlig)
    for i in range(len(fids)):
        R, t = kabsch(prot[i], ref)
        dock_ref[i] = dock[i] @ R.T + t
        mdlig_ref[i] = mdlig[i] @ R.T + t

    refset = mdlig_ref[::args.ref_stride]
    print(f"  best-matched against {len(refset)} reference poses "
          f"(stride {args.ref_stride})", flush=True)

    # ---- best-matched: min over reference poses, over mappings ----
    bm = np.full(len(fids), np.inf)
    bm_ref = np.zeros(len(fids), int)
    step = max(1, int(2e7 / (len(refset) * dock.shape[1] * 3)))
    for s in range(0, len(fids), step):
        e = min(s + step, len(fids))
        block = np.full((e - s, len(refset)), np.inf, dtype=np.float32)
        for m in maps:
            D = dock_ref[s:e, None, m, :].astype(np.float32)
            Rf = refset[None, :, :, :].astype(np.float32)
            block = np.minimum(
                block, np.sqrt(((D - Rf) ** 2).sum(-1).mean(-1)))
        j = block.argmin(1)
        bm[s:e] = block[np.arange(e - s), j]
        bm_ref[s:e] = j * args.ref_stride
        if (s // step) % 10 == 0:
            print(f"    ...{e}/{len(fids)}", flush=True)

    def pct(x, c):
        return 100.0 * float((x < c).mean())

    print(f"  frame-matched: median {np.median(fm):.2f} A, "
          f"<3A {pct(fm,3):.2f}%, <5A {pct(fm,5):.2f}%")
    print(f"  best-matched : median {np.median(bm):.2f} A, "
          f"<3A {pct(bm,3):.2f}%, <5A {pct(bm,5):.2f}%", flush=True)

    return dict(lig=lig, frame_ids=fids, frame_matched=fm, best_matched=bm,
                best_match_ref=bm_ref,
                fm3=pct(fm, 3), fm5=pct(fm, 5),
                bm3=pct(bm, 3), bm5=pct(bm, 5))


def kde(x, grid):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std() < 1e-9:
        return np.zeros_like(grid)
    h = max(1.06 * x.std() * x.size ** (-1 / 5), 1e-6)
    out = np.zeros_like(grid)
    for s in range(0, x.size, 20000):          # chunk to bound memory
        c = x[s:s + 20000]
        z = (grid[:, None] - c[None, :]) / h
        out += np.exp(-0.5 * z ** 2).sum(1)
    return out / (x.size * h * np.sqrt(2 * np.pi))


def figure(r, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8))
    for ax, key, ttl, colour, xmax in (
            (axes[0], "frame_matched", "Frame-matched RMSD", "#C0504D", 50),
            (axes[1], "best_matched", "Best-matched RMSD", "#065A82", 8)):
        x = r[key]
        g = np.linspace(0, max(xmax, float(np.nanmax(x)) * 1.05), 400)
        y = kde(x, g)
        ax.plot(g, y, color=colour, lw=1.8)
        ax.fill_between(g, y, color=colour, alpha=0.25)
        p3 = r["fm3"] if key == "frame_matched" else r["bm3"]
        p5 = r["fm5"] if key == "frame_matched" else r["bm5"]
        # inside the axes with a backing box: at y=0.78 the block collided
        # with the axes title once the font grew
        ax.text(0.97, 0.94,
                f"RMSD < 3\u00c5:  {p3:.2f}% frames\n"
                f"RMSD < 5\u00c5:  {p5:.2f}% frames",
                transform=ax.transAxes, fontsize=11, color=colour,
                ha="right", va="top", linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor=colour, alpha=0.85, linewidth=0.8))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
        ax.set_title(ttl, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel("Density", fontsize=11)
        ax.tick_params(labelsize=9)
    axes[1].set_xlabel("RMSD (\u00c5)", fontsize=12)
    fig.suptitle(f"Ligand {r['lig']} — holo docking vs MD reference poses",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    p = os.path.join(outdir, f"lig{r['lig']}_rmsd.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=PILOT)
    ap.add_argument("--winners", default=WINNERS)
    ap.add_argument("--mdroot", default=MDROOT)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--ref-stride", type=int, default=1,
                    help="use every Nth MD pose as a best-match reference")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mapping = lig_to_desres(args.mdroot)
    rows = []
    for lig in args.ligs:
        if lig not in mapping:
            print(f"lig{lig}: no DESRES id"); continue
        try:
            r = run_ligand(lig, mapping[lig], args)
        except Exception as e:
            print(f"  lig{lig} FAILED: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        np.savez(os.path.join(args.outdir, f"lig{lig}_rmsd.npz"),
                 lig=r["lig"], frame_ids=r["frame_ids"],
                 frame_matched=r["frame_matched"],
                 best_matched=r["best_matched"],
                 best_match_ref=r["best_match_ref"])
        figure(r, args.outdir)
        rows.append(r)
        print(f"  saved lig{lig}_rmsd.npz + .png", flush=True)

    if rows:
        p = os.path.join(args.outdir, "rmsd_table.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ligand", "n_frames",
                        "frame_matched_pct_lt3", "frame_matched_pct_lt5",
                        "best_matched_pct_lt3", "best_matched_pct_lt5"])
            for r in rows:
                w.writerow([r["lig"], len(r["frame_ids"]),
                            f"{r['fm3']:.2f}", f"{r['fm5']:.2f}",
                            f"{r['bm3']:.2f}", f"{r['bm5']:.2f}"])
        print("\n=== Table 2 layout:  %<3A (%<5A) ===")
        print(f"{'lig':>5} {'frame-matched':>18} {'best-matched':>18}")
        for r in rows:
            print(f"{r['lig']:>5} {r['fm3']:>10.2f} ({r['fm5']:.2f})"
                  f" {r['bm3']:>10.2f} ({r['bm5']:.2f})")
        print(f"\noutputs -> {args.outdir}/")


if __name__ == "__main__":
    main()
