#!/usr/bin/env python3
"""Aggregate runs/*/timing.csv + gpu.csv + meta.env into a benchmark table.
Reports first-dock wall (cold, e.g. Vina-GPU kernel compile) separately from
warm median, so cold-start cost is visible rather than smeared into the mean."""
import csv, glob, os, statistics, sys

def read_meta(d):
    m = {}
    p = os.path.join(d, "meta.env")
    if os.path.exists(p):
        for ln in open(p):
            if "=" in ln:
                k, v = ln.strip().split("=", 1); m[k] = v
    return m

def gpu_util_window(d, t0, t1):
    p = os.path.join(d, "gpu.csv"); vals = []
    if not (os.path.exists(p) and t0 and t1): return None
    for ln in open(p):
        parts = ln.strip().split(",")
        if len(parts) < 2: continue
        try:
            t = float(parts[0]); u = float(parts[1])
        except ValueError:
            continue
        if t0 <= t <= t1: vals.append(u)
    return round(statistics.mean(vals), 1) if vals else None

def iqr(xs):
    if len(xs) < 2: return 0.0
    q = statistics.quantiles(xs, n=4)
    return round(q[2] - q[0], 3)

runs_dir, out_dir = sys.argv[1], sys.argv[2]
# group runs by (tool,deploy,nframes,residues), pooling reps
groups = {}
for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
    tcsv = os.path.join(d, "timing.csv")
    if not os.path.exists(tcsv): continue
    rows = list(csv.DictReader(open(tcsv)))
    if not rows: continue
    r0 = rows[0]
    key = (r0["tool"], r0["deploy"], int(r0["nframes"]), r0["residues"])
    meta = read_meta(d)
    t0 = float(meta.get("run_start", 0)) or None
    t1 = float(meta.get("run_end", 0)) or None
    walls = [float(x["wall_s"]) for x in rows if x["exit"] == "0"]
    fails = sum(1 for x in rows if x["exit"] != "0")
    maxrss = max((int(x["maxrss_kb"]) for x in rows if x["maxrss_kb"].isdigit()), default=0)
    g = groups.setdefault(key, {"first": [], "warm": [], "run_wall": [],
                                "gpu": [], "maxrss": 0, "fails": 0, "ndock": 0})
    if walls:
        g["first"].append(walls[0])          # cold dock of this run
        g["warm"].extend(walls[1:] or walls)  # warm docks (fallback if n=1)
    if t0 and t1: g["run_wall"].append(t1 - t0)
    u = gpu_util_window(d, t0, t1)
    if u is not None: g["gpu"].append(u)
    g["maxrss"] = max(g["maxrss"], maxrss)
    g["fails"] += fails
    g["ndock"] += len(rows)

hdr = ["tool","deploy","n_frames","residues","docks","first_dock_s",
       "warm_median_s","warm_IQR_s","per_frame_s","throughput_docks_hr",
       "gpu_util_%","maxRSS_MB","fails"]
lines = []
for key in sorted(groups):
    tool, deploy, nf, res = key
    g = groups[key]
    first = round(statistics.median(g["first"]), 3) if g["first"] else None
    warm = round(statistics.median(g["warm"]), 3) if g["warm"] else None
    nres = 20 if res == "all" else len(res.split("-"))
    per_frame = round(warm * nres, 3) if warm else None
    thr = round(3600.0 / warm) if warm else None
    gpu = round(statistics.mean(g["gpu"]), 1) if g["gpu"] else "-"
    lines.append([tool, deploy, nf, res, g["ndock"], first, warm,
                  iqr(g["warm"]), per_frame, thr, gpu,
                  round(g["maxrss"]/1024), g["fails"]])

# markdown
md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"]*len(hdr)) + "|"]
for r in lines:
    md.append("| " + " | ".join(str(x) for x in r) + " |")
foot = (f"\n_ligand fixed; box edge, exhaustiveness={{}}, thread, seed identical across cells. "
        f"first_dock_s = cold (incl. Vina-GPU kernel compile); warm_median excludes it._\n")
open(os.path.join(out_dir, "benchmark_table.md"), "w").write("\n".join(md) + foot)
with open(os.path.join(out_dir, "benchmark_table.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(lines)
print("\n".join(md))
