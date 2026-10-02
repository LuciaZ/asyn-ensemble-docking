#!/usr/bin/env python3
# ============================================================================
# extract_scores_474123.py -- best-of-20 collapse for the 20,020-frame runs
#
# The lig23/41/47 ensembles were analysed for fingerprints only; no winner
# scores were ever saved. This walks their raw per-box outputs and writes
# lig{N}_winners.csv in the same format the pilot uses (frame_id, res_box,
# energy, n_boxes_found), so analyze_docking_scores.py can pool all ligands
# onto one min-max scale.
#
# Two directory layouts are handled and auto-detected:
#   flat    DIR/frame_XXXXX_resNN_out.pdbqt          (holo runs, lig23/41 apo)
#   nested  DIR/frame_XXXXX/<something>resNN<...>.pdbqt   (lig47 apo)
#
# Only the head of each file is read: Vina writes poses sorted by energy, so
# the first "REMARK VINA RESULT" line is that box's best pose. At ~2 million
# files across all runs, reading whole files would dominate the runtime.
#
# COMPLETENESS. These runs are partial (lig23 holo ~66%, others 88-94%), so
# per-frame box coverage varies. n_boxes_found records it per frame, and the
# summary prints the distribution -- a frame with 3 boxes is a weaker
# best-of-20 estimate than one with 20, and if you use these ligands to set
# a normalisation scale, that unevenness propagates into the scale.
#
# Run (from /home/zhihanz/data/scripts):
#   python extract_scores_474123.py --smoke        # 200 frames per run
#   python extract_scores_474123.py --workers 8
# ============================================================================
import os, re, csv, glob, argparse, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

HPC = "/home/zhihanz/data/VinaGPU/hpc_results"
OUT = os.path.join(HPC, "scores_474123")

# label -> (ligand, directory)
RUNS = [
    ("holo", 23, f"{HPC}/07052026/holo_docking/ligand23_holo_outputs"),
    ("holo", 41, f"{HPC}/07052026/holo_docking/ligand41_holo_outputs"),
    ("holo", 47, f"{HPC}/07052026/holo_docking/ligand47_holo_outputs"),
    ("apo",  23, f"{HPC}/06242026/ligand23_ensemble/outputs"),
    ("apo",  41, f"{HPC}/06242026/ligand41_ensemble/outputs"),
    ("apo",  47, f"{HPC}/05122026/outputs"),   # per-residue, 400400 files (complete); o_outputs is the superseded run
]

RE_VINA  = re.compile(rb"REMARK VINA RESULT:\s*(-?\d+\.?\d*)")
RE_FLAT  = re.compile(r"frame_(\d+)_res(\d+)_out\.pdbqt$")
RE_RES   = re.compile(r"res(\d+)")
RE_FRAME = re.compile(r"frame_(\d+)")
HEAD     = 4096          # first VINA RESULT is within the first few lines


def first_energy(path):
    """Best-pose energy for one box, or None."""
    try:
        with open(path, "rb") as f:
            m = RE_VINA.search(f.read(HEAD))
    except OSError:
        return None
    return float(m.group(1)) if m else None


def list_flat(d):
    """[(frame, res, path)] for DIR/frame_XXXXX_resNN_out.pdbqt."""
    out = []
    with os.scandir(d) as it:
        for e in it:
            if not e.is_file():
                continue
            m = RE_FLAT.search(e.name)
            if m:
                out.append((m.group(1), int(m.group(2)), e.path))
    return out


def list_nested(d):
    """[(frame, res, path)] for DIR/frame_XXXXX/*.pdbqt."""
    out = []
    with os.scandir(d) as it:
        subs = [e for e in it if e.is_dir() and e.name.startswith("frame_")]
    for sub in subs:
        mf = RE_FRAME.search(sub.name)
        if not mf:
            continue
        fid = mf.group(1)
        try:
            with os.scandir(sub.path) as it2:
                for e in it2:
                    if not (e.is_file() and e.name.endswith(".pdbqt")):
                        continue
                    if "out" not in e.name.lower():
                        continue                      # skip inputs/configs
                    mr = RE_RES.search(e.name)
                    if mr:
                        out.append((fid, int(mr.group(1)), e.path))
        except OSError:
            continue
    return out


def detect_and_list(d):
    if not os.path.isdir(d):
        return None, []
    flat = list_flat(d)
    if flat:
        return "flat", flat
    nested = list_nested(d)
    if nested:
        return "nested", nested
    return None, []


def collapse(entries, workers):
    """entries: [(frame, res, path)] -> {frame: (best_energy, best_res, n)}"""
    paths = [p for _, _, p in entries]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            energies = list(ex.map(first_energy, paths, chunksize=512))
    else:
        energies = [first_energy(p) for p in paths]

    best = {}
    for (fid, res, _), e in zip(entries, energies):
        if e is None:
            continue
        cur = best.get(fid)
        if cur is None:
            best[fid] = [e, res, 1]
        else:
            cur[2] += 1
            if e < cur[0]:
                cur[0], cur[1] = e, res
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="only the first 200 frames of each run")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to labels, e.g. --only holo")
    args = ap.parse_args()

    for lbl in ("holo", "apo"):
        os.makedirs(os.path.join(args.outdir, lbl), exist_ok=True)

    print(f"{'run':>12} {'layout':>7} {'files':>9} {'frames':>7} "
          f"{'mean':>7} {'best':>7} {'worst':>7} {'boxes/frame':>12}")
    for lbl, lig, d in RUNS:
        if args.only and lbl not in args.only:
            continue
        layout, entries = detect_and_list(d)
        if not entries:
            print(f"{lbl+' lig'+str(lig):>12} {'-':>7} {'0':>9}   "
                  f"(no outputs found in {d})")
            continue

        if args.smoke:
            keep = sorted({f for f, _, _ in entries})[:200]
            kset = set(keep)
            entries = [t for t in entries if t[0] in kset]

        best = collapse(entries, args.workers)
        if not best:
            print(f"{lbl+' lig'+str(lig):>12} {layout:>7} "
                  f"{len(entries):>9}   (no energies parsed)")
            continue

        rows = sorted((int(f), v[1], v[0], v[2]) for f, v in best.items())
        p = os.path.join(args.outdir, lbl, f"lig{lig}_winners.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "res_box", "energy", "n_boxes_found"])
            for fid, res, e, n in rows:
                w.writerow([fid, res, f"{e:.3f}", n])

        es = [r[2] for r in rows]
        nb = [r[3] for r in rows]
        import statistics as st
        print(f"{lbl+' lig'+str(lig):>12} {layout:>7} {len(entries):>9} "
              f"{len(rows):>7} {st.fmean(es):>7.2f} {min(es):>7.2f} "
              f"{max(es):>7.2f} {st.fmean(nb):>7.1f} (min {min(nb)})")

    print(f"\nwinners CSV -> {args.outdir}/{{holo,apo}}/lig*_winners.csv")
    print("\nfeed them to the score analysis, published trio setting the scale:")
    print("  python analyze_docking_scores.py \\\n"
          "    --ligs 47 41 23 38 28 5 34 24 3 17 53 9 40 \\\n"
          f"    --holo ~/data/VinaGPU/hpc_results/holo_08192026/winners "
          f"{args.outdir}/holo \\\n"
          f"    --apo  ~/data/VinaGPU/hpc_results/08142026/winners "
          f"{args.outdir}/apo \\\n"
          "    --pool-ligs 47 41 23")


if __name__ == "__main__":
    main()
