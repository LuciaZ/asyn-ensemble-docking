#!/usr/bin/env python3
# ============================================================================
# collapse_apo_remaining.py -- best-of-20 winner per frame for apo_remaining
#
# For every (ligand, frame) in the manifest, read the 20 per-residue box
# outputs, keep the one with the lowest energy, and write
#
#   outputs_vinagpu/winners/lig{N}_winners.pdbqt   one MODEL per frame,
#                                                  numbered frame_id + 1
#   outputs_vinagpu/winners/lig{N}_winners.csv     frame_id,res_box,energy,
#                                                  n_boxes_found
#   outputs_vinagpu/winners/lig{N}_incomplete.txt  "lig frame" for frames
#                                                  with fewer than 20 boxes
#
# Same MODEL numbering and CSV columns as the pilot winners, so the score
# analysis and build_docked_trajectories.py read these unchanged.
#
# dartfs-safe: every path is built from the manifest and opened directly. No
# directory under _raw/ is ever listed or globbed -- those hold ~100k files
# each and a listing stalls for minutes in D state. Only the head of each file
# is read for the energy (Vina writes poses best-first); the full pose is read
# once per frame, from the winning box only. Standard library only, since the
# base env on login and compute nodes has no numpy.
#
# Run as an array, one ligand per task:
#   sbatch collapse_apo_remaining.slurm
# or one ligand by hand:
#   python3 collapse_apo_remaining.py --lig 1
# then merge the refill lists and summaries:
#   python3 collapse_apo_remaining.py --merge
# ============================================================================
import os, re, sys, csv, argparse, statistics

BASE = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_remaining"
N_RES = 20
HEAD = 4096
RE_VINA = re.compile(rb"REMARK VINA RESULT:\s*(-?\d+\.?\d*)")


def read_manifest(base):
    per = {}
    with open(os.path.join(base, "manifest_apo.txt")) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) == 2:
                per.setdefault(int(p[0]), []).append(p[1])
    return per


def box_energy(path):
    try:
        with open(path, "rb") as fh:
            m = RE_VINA.search(fh.read(HEAD))
    except OSError:
        return None
    return float(m.group(1)) if m else None


def first_model(path):
    """Lines of the first MODEL block (best pose), without MODEL/ENDMDL."""
    with open(path) as fh:
        lines = fh.readlines()
    if not any(l.startswith("MODEL") for l in lines):
        return lines
    out, inside = [], False
    for ln in lines:
        if ln.startswith("MODEL"):
            if inside:
                break
            inside = True
        elif ln.startswith("ENDMDL"):
            if inside:
                break
        elif inside:
            out.append(ln)
    return out


def collapse(base, lig, fids):
    raw = os.path.join(base, "outputs_vinagpu", "_raw", f"lig{lig}")
    win = os.path.join(base, "outputs_vinagpu", "winners")
    os.makedirs(win, exist_ok=True)

    rows, incomplete, energies = [], [], []
    with open(os.path.join(win, f"lig{lig}_winners.pdbqt.tmp"), "w") as pq:
        for n, tag in enumerate(fids, 1):
            best_e, best_r, found = None, None, 0
            for r in range(1, N_RES + 1):
                e = box_energy(os.path.join(raw, f"frame_{tag}_res{r:02d}_out.pdbqt"))
                if e is None:
                    continue
                found += 1
                if best_e is None or e < best_e:
                    best_e, best_r = e, r
            if found < N_RES:
                incomplete.append(tag)
            if best_e is None:
                continue
            body = first_model(os.path.join(raw, f"frame_{tag}_res{best_r:02d}_out.pdbqt"))
            fid = int(tag)
            pq.write(f"MODEL {fid + 1}\n")
            pq.writelines(body)
            pq.write("ENDMDL\n")
            rows.append((fid, best_r, best_e, found))
            energies.append(best_e)
            if n % 1000 == 0:
                print(f"  lig{lig}: {n}/{len(fids)} frames", flush=True)
    os.replace(os.path.join(win, f"lig{lig}_winners.pdbqt.tmp"),
               os.path.join(win, f"lig{lig}_winners.pdbqt"))

    with open(os.path.join(win, f"lig{lig}_winners.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_id", "res_box", "energy", "n_boxes_found"])
        for fid, r, e, k in rows:
            w.writerow([fid, r, f"{e:.3f}", k])
    with open(os.path.join(win, f"lig{lig}_incomplete.txt"), "w") as fh:
        for tag in incomplete:
            fh.write(f"{lig} {tag}\n")

    nb = [k for *_, k in rows]
    summary = (f"lig{lig}\t{len(fids)}\t{len(rows)}\t{len(fids) - len(incomplete)}\t"
               f"{sum(nb)}\t{len(fids) * N_RES}\t"
               f"{min(energies):.2f}\t{statistics.fmean(energies):.2f}" if energies
               else f"lig{lig}\t{len(fids)}\t0\t0\t0\t{len(fids) * N_RES}\tnan\tnan")
    with open(os.path.join(win, f"lig{lig}_summary.tsv"), "w") as fh:
        fh.write(summary + "\n")
    print(f"lig{lig}: {len(rows)}/{len(fids)} frames with a winner, "
          f"{len(incomplete)} incomplete, boxes {sum(nb)}/{len(fids) * N_RES}",
          flush=True)


def merge(base):
    win = os.path.join(base, "outputs_vinagpu", "winners")
    per = read_manifest(base)
    lines, refill, missing = [], [], []
    for lig in sorted(per):
        s = os.path.join(win, f"lig{lig}_summary.tsv")
        if not os.path.exists(s):
            missing.append(lig); continue
        lines.append(open(s).read().strip())
        refill.extend(open(os.path.join(win, f"lig{lig}_incomplete.txt")).read().splitlines())
    hdr = "ligand\tframes\twith_winner\tcomplete\tboxes_found\tboxes_expected\tbest\tmean"
    with open(os.path.join(win, "collapse_summary.tsv"), "w") as fh:
        fh.write(hdr + "\n" + "\n".join(lines) + "\n")
    with open(os.path.join(base, "manifest_apo_refill.txt"), "w") as fh:
        fh.write("\n".join(refill) + ("\n" if refill else ""))
    print(hdr); print("\n".join(lines))
    tot_f = sum(int(l.split("\t")[4]) for l in lines)
    tot_e = sum(int(l.split("\t")[5]) for l in lines)
    print(f"\nboxes found {tot_f:,} / {tot_e:,} ({100 * tot_f / max(tot_e, 1):.3f}%)")
    print(f"incomplete frames: {len(refill)}  -> {base}/manifest_apo_refill.txt")
    if missing:
        print(f"NOT collapsed yet: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--task", type=int, default=None, help="array index -> ligand")
    ap.add_argument("--lig", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args.base); return
    per = read_manifest(args.base)
    ligs = sorted(per)
    if args.lig is not None:
        lig = args.lig
    elif args.task is not None:
        if args.task >= len(ligs):
            print(f"task {args.task}: only {len(ligs)} ligands"); return
        lig = ligs[args.task]
    else:
        ap.error("give --task, --lig or --merge")
    collapse(args.base, lig, per[lig])


if __name__ == "__main__":
    main()
