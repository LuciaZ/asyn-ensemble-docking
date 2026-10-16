#!/usr/bin/env python3
# ============================================================================
# make_manifest_1000.py -- per-ligand 1000-frame manifest for docking
#
# Reads manifest_<mode>.txt ("lig frame" lines) and keeps, for each ligand,
# N evenly spaced frames out of that ligand's frames sorted by id. The
# selection reproduces numpy.round(numpy.linspace(0, n-1, N)) exactly, i.e.
# the frames stride_test.py evaluated, so the convergence test applies to
# what gets docked.
#
#   python3 make_manifest_1000.py --base <project dir> --mode holo
#   -> <project dir>/manifest_holo1000.txt
#
# Dock it with the usual script; configs and done-markers are shared with the
# full manifest, so boxes already docked are skipped:
#   sbatch --export=ALL,MODE=holo1000,BASE=<project dir> ... dock_one_node.slurm
#
# --count-done stats the done-marker of every selected box to report how much
# of the subset is already finished (one stat per box; slow on dartfs).
# Standard library only.
# ============================================================================
import os, argparse

N_RES = 20


def linspace_round(n, k):
    """Same values as numpy.unique(numpy.round(numpy.linspace(0, n-1, k)))."""
    if n <= k:
        return list(range(n))
    step = (n - 1) / (k - 1)
    ys = [i * step for i in range(k)]
    ys[-1] = float(n - 1)            # numpy pins the endpoint exactly
    return sorted({int(round(y)) for y in ys})   # both round half to even


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mode", default="holo")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--count-done", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.base, f"manifest_{args.mode}.txt")
    dst = os.path.join(args.base, f"manifest_{args.mode}{args.n}.txt")
    per = {}
    with open(src) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) == 2:
                per.setdefault(int(p[0]), []).append(p[1])

    lines, done, total = [], 0, 0
    for lig in sorted(per):
        tags = sorted(per[lig], key=int)
        keep = [tags[i] for i in linspace_round(len(tags), args.n)]
        for t in keep:
            lines.append(f"{lig} {t}")
            if args.count_done:
                for r in range(1, N_RES + 1):
                    total += 1
                    if os.path.exists(os.path.join(
                            args.base, "done", f"lig{lig}_frame_{t}_res{r:02d}.done")):
                        done += 1
        print(f"lig{lig}: {len(keep)} of {len(tags)} frames")

    with open(dst, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n{len(lines)} ligand-frame units ({len(lines) * N_RES:,} boxes) -> {dst}")
    if args.count_done:
        print(f"already docked: {done:,} / {total:,} boxes ({100 * done / max(total, 1):.1f}%), "
              f"{total - done:,} to go")


if __name__ == "__main__":
    main()
