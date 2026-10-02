#!/usr/bin/env python3
import csv, glob, os, statistics, sys
runs = sys.argv[1]; out_dir = sys.argv[2]
agg = {}
for f in glob.glob(os.path.join(runs, "*", "timing.csv")):
    for r in csv.DictReader(open(f)):
        k = (r["tool"], r["ligand"])
        d = agg.setdefault(k, {"walls": [], "edge": r["edge"], "nheavy": r["n_heavy"], "fail": 0})
        if r["exit"] != "0":
            d["fail"] += 1
            continue
        d["walls"].append(float(r["wall_s"]))
rows = []
for (tool, lig), d in agg.items():
    if not d["walls"]: continue
    per_box = statistics.median(d["walls"])
    per_frame = sum(d["walls"]) / max(1, len(d["walls"])) * 20
    rows.append([tool, lig, d["edge"], d["nheavy"], len(d["walls"]),
                 round(per_box,3), round(per_frame,2), d["fail"]])
rows.sort(key=lambda x: (x[0], float(x[1].split('_')[0])))
hdr = ["tool","ligand","edge_A","n_atoms","n_box","per_box_s","per_frame_s","fails"]
md = ["| "+" | ".join(hdr)+" |", "|"+"|".join(["---"]*len(hdr))+"|"]
for r in rows: md.append("| "+" | ".join(str(x) for x in r)+" |")
for tool in sorted(set(r[0] for r in rows)):
    pf = [r[6] for r in rows if r[0]==tool]
    md.append(f"\n**{tool}**: {len(pf)} ligands, per-frame median {statistics.median(pf):.1f}s, "
              f"mean {statistics.mean(pf):.1f}s, range {min(pf):.1f}-{max(pf):.1f}s")
open(os.path.join(out_dir,"multilig_table.md"),"w").write("\n".join(md))
print("\n".join(md))
