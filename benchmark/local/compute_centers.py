#!/usr/bin/env python3
"""Print '<resSeq> <cx> <cy> <cz>' for each residue in a receptor PDBQT,
one line per residue sorted by resSeq ascending. Center = geometric mean of
heavy atoms (matches the lab generate_gpf.py convention; hydrogens skipped)."""
import sys
from collections import OrderedDict

def main(pdbqt):
    res = OrderedDict()
    with open(pdbqt) as fh:
        for ln in fh:
            if ln[:6] not in ("ATOM  ", "HETATM"):
                continue
            adtype = ln.split()[-1]
            if adtype.startswith("H"):          # HD/HS/H -> hydrogen, skip
                continue
            try:
                resseq = int(ln[22:26])
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except ValueError:
                continue
            res.setdefault(resseq, []).append((x, y, z))
    for rs in sorted(res):
        xs = res[rs]
        n = len(xs)
        cx = sum(p[0] for p in xs) / n
        cy = sum(p[1] for p in xs) / n
        cz = sum(p[2] for p in xs) / n
        print(f"{rs} {cx:.3f} {cy:.3f} {cz:.3f}")

if __name__ == "__main__":
    main(sys.argv[1])
