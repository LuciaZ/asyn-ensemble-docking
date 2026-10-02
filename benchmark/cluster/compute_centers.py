#!/usr/bin/env python3
"""Print '<resSeq> <cx> <cy> <cz>' per residue in a receptor PDBQT (heavy atoms)."""
import sys
from collections import OrderedDict
def main(pdbqt):
    res = OrderedDict()
    with open(pdbqt) as fh:
        for ln in fh:
            if ln[:6] not in ("ATOM  ", "HETATM"): continue
            if ln.split()[-1].startswith("H"): continue
            try:
                rs = int(ln[22:26]); x=float(ln[30:38]); y=float(ln[38:46]); z=float(ln[46:54])
            except ValueError: continue
            res.setdefault(rs, []).append((x,y,z))
    for rs in sorted(res):
        xs=res[rs]; n=len(xs)
        print(f"{rs} {sum(p[0] for p in xs)/n:.3f} {sum(p[1] for p in xs)/n:.3f} {sum(p[2] for p in xs)/n:.3f}")
if __name__=="__main__": main(sys.argv[1])
