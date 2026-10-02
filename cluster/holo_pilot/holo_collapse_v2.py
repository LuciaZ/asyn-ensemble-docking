#!/usr/bin/env python3
# ============================================================================
# holo_collapse_v2.py -- best-of-20 collapse, dartfs-safe.
#
# WHY v2: v1 used glob("_raw/lig{N}/frame_*_res*_out.pdbqt"), which enumerates
# a ~100k-file directory. On dartfs large-directory metadata ops stall (procs
# wedge in D state), exactly like scan_missing did. v2 never lists a big dir:
# it reads the frame list from the receptor dir (or config tree) and then
# CONSTRUCTS each of the 20 output paths and opens them directly. Single-file
# opens are fast on dartfs even when the containing dir is huge.
#
# Usage: python holo_collapse_v2.py                 # all 10 pilot ligands
#        python holo_collapse_v2.py --ligs 38 40
#        python holo_collapse_v2.py --progress 500  # print every N frames
# ============================================================================
import os, re, sys, argparse

BASE   = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot"
RAW    = os.path.join(BASE, "outputs_vinagpu", "_raw")
CFGROOT= os.path.join(BASE, "configs")
OUTDIR = os.path.join(BASE, "outputs_vinagpu", "winners")
PILOT  = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]
N_RES  = 20

_RE_ENERGY = re.compile(r'^REMARK\s+VINA\s+RESULT[:\s]+\s*(-?\d+\.?\d*)')

def frame_ids(lig):
    """Frame ids for this ligand, from the config tree.
    configs/lig{N}/ holds 5000 small subdirs -- listing it is fine; the
    problem dir is _raw/lig{N} with 100k flat files, which we never list."""
    d = os.path.join(CFGROOT, f"lig{lig}")
    try:
        names = os.listdir(d)
    except OSError:
        return []
    fids = [n[len("frame_"):] for n in names if n.startswith("frame_")]
    return sorted(fids)

def best_model(path):
    """Lowest-energy MODEL in one output pdbqt -> (energy, [lines])."""
    best_e, best_lines = None, None
    cur, cur_e, in_model = [], None, False
    try:
        fh = open(path)
    except OSError:
        return None, None          # box missing -> just skip it
    with fh:
        for line in fh:
            if line.startswith("MODEL"):
                in_model, cur, cur_e = True, [line], None; continue
            if not in_model:
                continue
            cur.append(line)
            if cur_e is None:
                m = _RE_ENERGY.match(line)
                if m: cur_e = float(m.group(1))
            if line.startswith("ENDMDL"):
                if cur_e is not None and (best_e is None or cur_e < best_e):
                    best_e, best_lines = cur_e, list(cur)
                in_model, cur, cur_e = False, [], None
    return best_e, best_lines

def collapse_ligand(lig, outdir, progress=0):
    fids = frame_ids(lig)
    if not fids:
        return None
    ligraw = os.path.join(RAW, f"lig{lig}")
    pdbqt_out, csv_rows = [], []
    n_ok = n_missing = n_partial = 0
    for i, fid in enumerate(fids, 1):
        best_e, best_lines, best_res, found = None, None, None, 0
        for res in range(1, N_RES + 1):
            # CONSTRUCT the path -- no glob, no listdir on the big dir
            p = os.path.join(ligraw, f"frame_{fid}_res{res:02d}_out.pdbqt")
            e, lines = best_model(p)
            if e is None:
                continue
            found += 1
            if best_e is None or e < best_e:
                best_e, best_lines, best_res = e, lines, res
        if best_lines is None:
            n_missing += 1
        else:
            if found < N_RES:
                n_partial += 1
            fid_int = int(fid)
            pdbqt_out.append(f"MODEL {fid_int+1}\n")
            pdbqt_out.extend(ln for ln in best_lines if not ln.startswith("MODEL"))
            csv_rows.append((fid_int, best_res, best_e, found))
            n_ok += 1
        if progress and i % progress == 0:
            print(f"    lig{lig}: {i}/{len(fids)} frames", flush=True)

    with open(os.path.join(outdir, f"lig{lig}_winners.pdbqt"), "w") as f:
        f.write("".join(pdbqt_out))
    with open(os.path.join(outdir, f"lig{lig}_winners.csv"), "w") as f:
        f.write("frame_id,res_box,energy,n_boxes_found\n")
        for fid, res, e, found in csv_rows:
            f.write(f"{fid},{res},{e:.3f},{found}\n")

    es = [r[2] for r in csv_rows]
    stats = (min(es), sum(es)/len(es), max(es)) if es else (float('nan'),)*3
    return n_ok, n_missing, n_partial, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=PILOT)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--progress", type=int, default=1000,
                    help="print every N frames (0 to disable)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summary = []
    for lig in args.ligs:
        print(f"[lig{lig}] collapsing...", flush=True)
        r = collapse_ligand(lig, args.outdir, args.progress)
        if r is None:
            line = f"lig{lig:>3}: NO FRAMES FOUND"
        else:
            n_ok, n_missing, n_partial, (emin, emean, emax) = r
            line = (f"lig{lig:>3}: winners={n_ok} no_output={n_missing} "
                    f"partial_boxes={n_partial} "
                    f"E[min/mean/max]={emin:.2f}/{emean:.2f}/{emax:.2f}")
        print(line, flush=True)
        summary.append(line)
    with open(os.path.join(args.outdir, "collapse_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nwinners -> {args.outdir}/lig{{N}}_winners.pdbqt (+ .csv)")

if __name__ == "__main__":
    main()
