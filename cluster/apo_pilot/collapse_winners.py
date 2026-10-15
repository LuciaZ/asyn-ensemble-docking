#!/usr/bin/env python3
# collapse_winners_v2.py -- box-major traversal, 10x fewer file opens, live progress
import os, re, sys, argparse

BASE   = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/apo_50lig_08122026"
RAW    = os.path.join(BASE, "outputs_vinagpu", "_raw")
OUTDIR = os.path.join(BASE, "outputs_vinagpu", "winners")
PILOT_LIGS = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]
FRAME_STRIDE = 4
N_FRAMES_TOTAL = 20020
_RE_ENERGY = re.compile(r'^REMARK\s+VINA\s+RESULT[:\s]+\s*(-?\d+\.?\d*)')

def best_model_in_file(path):
    best_e, best_lines = None, None
    cur, cur_e, in_model = [], None, False
    try:
        fh = open(path)
    except OSError:
        return None, None
    with fh:
        for line in fh:
            if line.startswith("MODEL"):
                in_model, cur, cur_e = True, [line], None
                continue
            if not in_model:
                continue
            cur.append(line)
            if cur_e is None:
                m = _RE_ENERGY.match(line)
                if m:
                    cur_e = float(m.group(1))
            if line.startswith("ENDMDL"):
                if cur_e is not None and (best_e is None or cur_e < best_e):
                    best_e, best_lines = cur_e, list(cur)
                in_model, cur, cur_e = False, [], None
    return best_e, best_lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=PILOT_LIGS)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    frame_ids = list(range(0, N_FRAMES_TOTAL, FRAME_STRIDE))
    print(f"stride-{FRAME_STRIDE}: {len(frame_ids)} frames; ligands: {args.ligs}", flush=True)

    # per-ligand accumulators: best pose so far for each frame
    best = {lig: {} for lig in args.ligs}   # lig -> {fid: (energy, res, lines)}

    for i, fid in enumerate(frame_ids):
        fid_str = f"{fid:05d}"
        for res in range(1, 21):
            boxdir = os.path.join(RAW, f"frame_{fid_str}_res{res:02d}")
            if not os.path.isdir(boxdir):
                continue
            for lig in args.ligs:
                path = os.path.join(boxdir, f"{lig}_charged_out.pdbqt")
                if not os.path.exists(path):
                    continue
                e, lines = best_model_in_file(path)
                if e is None:
                    continue
                prev = best[lig].get(fid)
                if prev is None or e < prev[0]:
                    best[lig][fid] = (e, res, lines)
        if (i+1) % 500 == 0:
            print(f"  ...{i+1}/{len(frame_ids)} frames scanned", flush=True)

    summary = []
    for lig in args.ligs:
        rows = sorted(best[lig].items())
        pdbqt_path = os.path.join(args.outdir, f"lig{lig}_winners.pdbqt")
        csv_path   = os.path.join(args.outdir, f"lig{lig}_winners.csv")
        with open(pdbqt_path, "w") as f:
            for fid, (e, res, lines) in rows:
                f.write(f"MODEL {fid+1}\n")
                for ln in lines:
                    if not ln.startswith("MODEL"):
                        f.write(ln)
        with open(csv_path, "w") as f:
            f.write("frame_id,res_box,energy\n")
            for fid, (e, res, lines) in rows:
                f.write(f"{fid},{res},{e:.3f}\n")
        if rows:
            es = [v[0] for _, v in rows]
            line = (f"lig{lig:>3}: winners={len(rows)} missing={len(frame_ids)-len(rows)}  "
                    f"E[min/mean/max]={min(es):.2f}/{sum(es)/len(es):.2f}/{max(es):.2f}")
        else:
            line = f"lig{lig:>3}: winners=0 (NO DATA)"
        print(line, flush=True)
        summary.append(line)
    with open(os.path.join(args.outdir, "collapse_summary.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nwinners -> {args.outdir}", flush=True)

if __name__ == "__main__":
    main()
