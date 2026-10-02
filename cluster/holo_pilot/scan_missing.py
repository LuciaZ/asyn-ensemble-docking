#!/usr/bin/env python3
import os, glob
BASE = "/dartfs-hpc/rc/lab/R/RobustelliP/Zhihan/holo_50lig_pilot"
CFGROOT = BASE + "/configs"
DONEDIR = BASE + "/done"
PILOT = [38,28,5,34,24,3,17,53,9,40]
missing = []
per_lig = {}
for lig in PILOT:
    cfgs = glob.glob(os.path.join(CFGROOT, f"lig{lig}", "frame_*", "res*.txt"))
    miss_l = 0
    for cfg in cfgs:
        parts = cfg.split(os.sep)
        res_str = parts[-1].replace("res","").replace(".txt","")
        fid = parts[-2].replace("frame_","")
        marker = os.path.join(DONEDIR, f"lig{lig}_frame_{fid}_res{res_str}.done")
        if not os.path.exists(marker):
            missing.append(cfg); miss_l += 1
    per_lig[lig] = (len(cfgs), miss_l)
    print(f"lig{lig}: {len(cfgs)} configs, {miss_l} missing", flush=True)
with open(os.path.join(BASE, "missing_configs.txt"), "w") as f:
    f.write("\n".join(missing) + "\n")
print(f"\nTOTAL MISSING: {len(missing)}")
print(f"-> {BASE}/missing_configs.txt")
