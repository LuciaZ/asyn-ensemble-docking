#!/usr/bin/env python3
# ============================================================================
# build_docked_trajectories.py -- docked protein-ligand trajectories for the
#                                 docking_tools analysis notebooks
#
# The example notebook in Dhar_IDP_ensemble_docking_25 starts from a docked
# trajectory (xtc + pdb) in which every frame is receptor + docked ligand.
# This project's docking jobs instead produced best-of-20 winner poses
# (ligand only, PDBQT) plus one receptor file per frame, so this script joins
# the two, one trajectory per ligand per job.
#
# Outputs, in OUTROOT/<job>/:
#   lig{N}_top.pdb      complex topology (frame 0) with CONECT for the ligand
#   lig{N}_xyz.npy      coordinates, (n_frames, n_atoms, 3), float32, nm
#   lig{N}_frames.npy   frame ids, in trajectory order
#   lig{N}_groups.json  ligand atom groups for docking_tools (global indices)
#
# Three things are handled here so the notebooks can stay as short as the
# example:
#
#   Ligand atom groups. The example hard-codes ring, donor and charge indices
#   for ligand 47. With 50 ligands they are detected from the pose: aromatic
#   rings are 5/6-membered rings whose carbons all carry AutoDock type A;
#   donors are polar hydrogens bonded to N or O; cationic centres are
#   ammonium nitrogens (four substituents), amidinium/guanidinium NH2, and
#   protonated pyridine-type ring nitrogens. Every detection is printed --
#   check them against the chemistry before trusting the charge panel.
#
#   Receptor residue order. save_pdb wrote two N-terminal hydrogens after the
#   last residue in the holo receptors, which MDTraj reads as a 21st residue.
#   Atoms are regrouped by residue number, and matched across frames by
#   (residue, atom name) rather than by file position.
#
#   Ligand residue selection. dual_contact selects the ligand with
#   "residue {idx}", which matches resSeq, not the topology index. Protein
#   residues are numbered 121-140 and the ligand's resSeq is set equal to its
#   topology index (20), so "residue 20" and "resid 20" both pick the ligand.
#
# Run (diffdock env, from ~/data/scripts or the repo's workstation/):
#   python build_docked_trajectories.py --job apo_pilot10 --smoke
#   python build_docked_trajectories.py --job apo_pilot10
#   python build_docked_trajectories.py --job holo_pilot10
#   python build_docked_trajectories.py --job trio_apo
#   python build_docked_trajectories.py --job apo_remaining37   # after collapse
# ============================================================================
import os, re, csv, json, argparse, itertools
import numpy as np

H = os.path.expanduser("~/data/VinaGPU/hpc_results")
APO_REC = os.path.expanduser(
    "~/data/autodock_vina/protein_frames/protein_frames_apo/"
    "protein_only_frame_{fid:05d}_fasudil_apo.pdb")
OUTROOT = os.path.expanduser("~/data/docked_trajectories")

PILOT = [38, 5, 28, 34, 3, 24, 17, 53, 9, 40]
REMAINING = [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
             25, 26, 27, 29, 30, 33, 35, 36, 37, 39, 46, 48, 49, 50, 51, 52,
             54, 55, 56]

JOBS = {
    "apo_pilot10": dict(
        kind="winners", ligs=PILOT, receptor=APO_REC,
        winners=f"{H}/08142026/winners"),
    "holo_pilot10": dict(
        kind="winners", ligs=PILOT, winners=f"{H}/holo_08192026/winners",
        receptor=f"{H}/08142026/holo_receptors_pdb/lig{{lig}}/frame_{{fid:06d}}.pdb"),
    "apo_remaining37": dict(
        kind="winners", ligs=REMAINING, receptor=APO_REC,
        winners=f"{H}/apo_remaining/winners"),
    # the 20,020-frame runs were never collapsed into winner poses; the best
    # box per frame is read from the extracted score table instead
    "trio_apo": dict(
        kind="raw", ligs=[47, 41, 23], receptor=APO_REC, min_boxes=15,
        scores=f"{H}/scores_474123/apo",
        raw={47: f"{H}/05122026/outputs",
             41: f"{H}/06242026/ligand41_ensemble/outputs",
             23: f"{H}/06242026/ligand23_ensemble/outputs"},
        raw_tmpl="frame_{fid:05d}_res{res:02d}_out.pdbqt"),
}

AD2EL = {"A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
         "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
         "H": "H", "HD": "H", "HS": "H", "P": "P",
         "F": "F", "CL": "Cl", "BR": "Br", "I": "I"}


# ------------------------------------------------------------------ readers ---
def read_receptor(path):
    """[(resSeq, resName, name, element, (x, y, z))] for ATOM records."""
    out = []
    with open(path) as fh:
        for ln in fh:
            if not ln.startswith("ATOM"):
                continue
            name = ln[12:16].strip()
            try:
                rs = int(ln[22:26])
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            except ValueError:
                continue
            el = ln[76:78].strip() if len(ln) >= 78 else ""
            if not el:
                el = name.lstrip("0123456789")[:1]
            out.append((rs, ln[17:20].strip(), name, el.capitalize(), xyz))
    return out


def atom_keys(atoms):
    """(resSeq, name, k): k counts earlier atoms with the same resSeq and name.
    Some writers reuse a name within a residue (generic hydrogen names); frames
    written by the same tool keep the same in-residue order, so the
    occurrence index still identifies each atom across frames."""
    seen, keys = {}, []
    for a in atoms:
        base = (a[0], a[2])
        k = seen.get(base, 0)
        seen[base] = k + 1
        keys.append((a[0], a[2], k))
    return keys


def receptor_layout(atoms):
    """Canonical order: residues ascending, atoms in file order within each.
    Regrouping by residue number absorbs the stray N-terminal hydrogens."""
    order = sorted(range(len(atoms)), key=lambda i: (atoms[i][0], i))
    keys_all = atom_keys(atoms)
    keys = [keys_all[i] for i in order]
    dups = sum(1 for k in keys if k[2] > 0)
    if dups:
        ex = sorted({(k[0], k[1]) for k in keys if k[2] > 0})[:4]
        print(f"    note: {dups} receptor atom name(s) repeat within a residue "
              f"(e.g. {ex}); matched by in-residue order")
    return order, keys


def receptor_coords(path, keys):
    atoms = read_receptor(path)
    pos = dict(zip(atom_keys(atoms), (a[4] for a in atoms)))
    missing = [k for k in keys if k not in pos]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing {missing[:3]}")
    return np.array([pos[k] for k in keys], dtype=np.float64)


def read_pdbqt_models(path, first_only=False):
    """[(model_number, [(adtype, (x, y, z))])], heavy atoms plus HD."""
    models, cur, num = [], [], 1
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("MODEL"):
                cur = []
                try:
                    num = int(ln.split()[1])
                except (IndexError, ValueError):
                    num = len(models) + 1
            elif ln.startswith(("ATOM", "HETATM")):
                ad = ln[77:79].strip().upper() if len(ln) >= 79 else ""
                if not ad:
                    continue
                if AD2EL.get(ad, ad[:1]) == "H" and ad != "HD":
                    continue                      # keep only polar H
                cur.append((ad, (float(ln[30:38]), float(ln[38:46]),
                                 float(ln[46:54]))))
            elif ln.startswith("ENDMDL"):
                models.append((num, cur))
                if first_only:
                    break
    if not models and cur:
        models.append((1, cur))
    return models


def read_scores(path):
    """{frame_id: (res_box, n_boxes)} from a winners CSV."""
    out = {}
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                fid = int(r["frame_id"])
                res = int(r["res_box"])
                nb = int(r.get("n_boxes_found") or 20)
            except (KeyError, ValueError, TypeError):
                continue
            out[fid] = (res, nb)
    return out


# ----------------------------------------------------------- group detection ---
def detect_groups(types, xyz):
    """Ring / donor / charge groups from one ligand pose (local indices)."""
    import networkx as nx
    el = [AD2EL.get(t, t[:1]) for t in types]
    heavy = [i for i, e in enumerate(el) if e != "H"]
    hyd = [i for i, e in enumerate(el) if e == "H"]
    d = lambda i, j: float(np.linalg.norm(xyz[i] - xyz[j]))

    G = nx.Graph()
    G.add_nodes_from(heavy)
    for i, j in itertools.combinations(heavy, 2):
        if d(i, j) < 1.95:
            G.add_edge(i, j)
    h_on = {}
    for h in hyd:
        near = min(heavy, key=lambda k: d(h, k))
        if d(h, near) < 1.25:
            h_on[h] = near
    nH = {i: sum(1 for v in h_on.values() if v == i) for i in heavy}

    try:
        cycles = nx.minimum_cycle_basis(G)
    except Exception:
        cycles = nx.cycle_basis(G)
    rings = []
    for c in cycles:
        if len(c) not in (5, 6):
            continue
        carbons = [a for a in c if el[a] == "C"]
        if carbons and all(types[a] == "A" for a in carbons) \
                and sum(types[a] == "A" for a in c) >= 3:
            rings.append(sorted(c))
    in_arom = {a for r in rings for a in r}

    donors = [[h_on[h], h] for h in hyd
              if h in h_on and el[h_on[h]] in ("N", "O")]

    pos = []
    for i in heavy:
        if el[i] != "N":
            continue
        deg, h = G.degree(i), nH[i]
        if deg + h >= 4 and h >= 1:                         # ammonium
            pos.append(i)
        elif h == 2 and deg == 1:                           # amidinium/guanidinium
            c = next(iter(G.neighbors(i)))
            if el[c] == "C" and sum(el[n] == "N" for n in G.neighbors(c)) >= 2:
                pos.append(i)
        elif i in in_arom and h == 1 and deg == 2:          # pyridinium-type
            ring = next(r for r in rings if i in r)
            if len(ring) == 6:
                pos.append(i)

    neg = []
    for i in heavy:
        if el[i] not in ("C", "S", "P"):
            continue
        term_o = [n for n in G.neighbors(i)
                  if el[n] == "O" and G.degree(n) == 1 and nH.get(n, 0) == 0]
        if (el[i] == "C" and len(term_o) == 2) or \
           (el[i] in ("S", "P") and len(term_o) >= 3):
            neg.extend(term_o)

    bonds = [list(e) for e in G.edges()] + [[v, h] for h, v in h_on.items()]
    return dict(aromatic_rings=rings, hbond_donors=donors,
                positive_charges=sorted(pos), negative_charges=sorted(neg),
                bonds=bonds)


def to_global(g, offset):
    shift = lambda xs: [x + offset for x in xs]
    return dict(
        aromatic_rings=[shift(r) for r in g["aromatic_rings"]],
        hbond_donors=[shift(p) for p in g["hbond_donors"]],
        positive_charges=shift(g["positive_charges"]),
        negative_charges=shift(g["negative_charges"]))


# ------------------------------------------------------------------- writer ---
def pdb_line(rec, serial, name, resn, chain, resi, xyz, el):
    ln = list(" " * 80)
    ln[0:6] = f"{rec:<6s}"
    ln[6:11] = f"{serial:>5d}"
    nm = name if len(name) == 4 else f" {name:<3s}"
    ln[12:16] = nm[:4]
    ln[17:20] = f"{resn:>3s}"[:3]
    ln[21] = chain
    ln[22:26] = f"{resi:>4d}"
    ln[30:38] = f"{xyz[0]:>8.3f}"
    ln[38:46] = f"{xyz[1]:>8.3f}"
    ln[46:54] = f"{xyz[2]:>8.3f}"
    ln[54:60] = f"{1.0:>6.2f}"
    ln[60:66] = f"{0.0:>6.2f}"
    ln[76:78] = f"{el:>2s}"[:2]
    return "".join(ln).rstrip()


def write_topology(path, rec_atoms, rec_order, rec_xyz, lig_types, lig_xyz,
                   lig_bonds, n_res):
    """Protein residues renumbered 121.. ; ligand resSeq = its topology index."""
    lines, serial = [], 0
    res_map, nxt = {}, 0
    for k, i in enumerate(rec_order):
        rs, resn, name, el, _ = rec_atoms[i]
        if rs not in res_map:
            res_map[rs] = 121 + nxt
            nxt += 1
        serial += 1
        lines.append(pdb_line("ATOM", serial, name, resn, "A", res_map[rs],
                              rec_xyz[k], el))
    lines.append("TER")
    n_rec = serial
    for j, t in enumerate(lig_types):
        el = AD2EL.get(t, t[:1])
        serial += 1
        lines.append(pdb_line("HETATM", serial, f"{el.upper()}{j + 1}"[:4],
                              "UNL", "B", n_res, lig_xyz[j], el))
    lines.append("TER")
    nb = {}
    for a, b in lig_bonds:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    for a in sorted(nb):
        partners = sorted(nb[a])
        for s in range(0, len(partners), 4):
            chunk = partners[s:s + 4]
            lines.append("CONECT" + f"{n_rec + a + 1:>5d}" +
                         "".join(f"{n_rec + p + 1:>5d}" for p in chunk))
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return n_rec


# --------------------------------------------------------------------- jobs ---
def poses_from_winners(job, lig):
    base = os.path.join(job["winners"], f"lig{lig}_winners")
    if not (os.path.exists(base + ".csv") and os.path.exists(base + ".pdbqt")):
        raise FileNotFoundError(f"{base}.csv/.pdbqt not found -- for "
                                f"apo_remaining37, collapse on the cluster "
                                f"first and copy the winners here")
    with open(base + ".csv") as fh:
        fids = [int(r["frame_id"]) for r in csv.DictReader(fh)]
    models = read_pdbqt_models(base + ".pdbqt")
    nums = [m[0] for m in models]
    if sorted(n - 1 for n in nums) == sorted(fids):        # MODEL = fid + 1
        pairs = [(n - 1, atoms) for n, atoms in models]
    elif len(models) == len(fids):                          # same order
        pairs = list(zip(fids, (m[1] for m in models)))
    else:
        raise ValueError(f"lig{lig}: {len(models)} MODELs vs {len(fids)} rows")
    return sorted(pairs, key=lambda p: p[0])


def poses_from_raw(job, lig):
    table = read_scores(os.path.join(job["scores"], f"lig{lig}_winners.csv"))
    pairs = []
    for fid in sorted(table):
        res, nb = table[fid]
        if nb < job.get("min_boxes", 0):
            continue
        p = os.path.join(job["raw"][lig], job["raw_tmpl"].format(fid=fid, res=res))
        if not os.path.exists(p):
            continue
        m = read_pdbqt_models(p, first_only=True)
        if m:
            pairs.append((fid, m[0][1]))
    return pairs


def build(job_name, lig, args):
    job = JOBS[job_name]
    pairs = (poses_from_winners if job["kind"] == "winners"
             else poses_from_raw)(job, lig)
    if args.smoke:
        pairs = pairs[:args.smoke]
    if not pairs:
        print(f"  lig{lig}: no poses"); return None

    types = [a[0] for a in pairs[0][1]]
    rec0 = job["receptor"].format(lig=lig, fid=pairs[0][0])
    if not os.path.exists(rec0):
        raise FileNotFoundError(f"receptor not found: {rec0}")
    rec_atoms = read_receptor(rec0)
    order, keys = receptor_layout(rec_atoms)
    n_res = len({k[0] for k in keys})
    n_rec = len(keys)

    xyz, fids, skipped = [], [], 0
    for fid, atoms in pairs:
        if [a[0] for a in atoms] != types:
            skipped += 1; continue
        rp = job["receptor"].format(lig=lig, fid=fid)
        try:
            r = receptor_coords(rp, keys)
        except (OSError, ValueError):
            skipped += 1; continue
        l = np.array([a[1] for a in atoms], dtype=np.float64)
        xyz.append(np.vstack([r, l]))
        fids.append(fid)
    if not xyz:
        print(f"  lig{lig}: nothing assembled"); return None
    xyz = np.array(xyz)

    lig0 = xyz[0, n_rec:]
    g_local = detect_groups(types, lig0)
    g = to_global(g_local, n_rec)
    g.update(ligand_residue_index=n_res, n_protein_atoms=n_rec,
             n_ligand_atoms=len(types), n_frames=len(fids),
             ligand_autodock_types=types)

    out = os.path.join(args.outroot, job_name)
    os.makedirs(out, exist_ok=True)
    write_topology(os.path.join(out, f"lig{lig}_top.pdb"), rec_atoms, order,
                   xyz[0, :n_rec], types, lig0, g_local["bonds"], n_res)
    np.save(os.path.join(out, f"lig{lig}_xyz.npy"),
            (xyz / 10.0).astype(np.float32))        # A -> nm for MDTraj
    np.save(os.path.join(out, f"lig{lig}_frames.npy"), np.array(fids))
    with open(os.path.join(out, f"lig{lig}_groups.json"), "w") as fh:
        json.dump(g, fh, indent=1)

    print(f"  lig{lig}: {len(fids)} frames ({skipped} skipped), "
          f"{n_res} residues, {n_rec}+{len(types)} atoms | "
          f"rings {len(g['aromatic_rings'])}, donors {len(g['hbond_donors'])}, "
          f"cation {g['positive_charges']}, anion {g['negative_charges']}",
          flush=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--ligs", nargs="*", type=int, default=None)
    ap.add_argument("--outroot", default=OUTROOT)
    ap.add_argument("--smoke", type=int, nargs="?", const=20, default=0,
                    help="only the first N frames (default 20)")
    args = ap.parse_args()
    ligs = args.ligs or JOBS[args.job]["ligs"]
    print(f"job {args.job}: {len(ligs)} ligands -> "
          f"{os.path.join(args.outroot, args.job)}")
    for lig in ligs:
        try:
            build(args.job, lig, args)
        except Exception as e:
            print(f"  lig{lig} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
