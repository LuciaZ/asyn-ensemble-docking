#!/usr/bin/env python3
"""Print box edge (A) for a ligand PDBQT: edge = 60^(1/3) * Rg over heavy atoms.
Reproduces the config generator's convention (41->13.01, 47->16.58)."""
import sys, math

def main(pdbqt):
    xs = []
    with open(pdbqt) as fh:
        for ln in fh:
            if ln[:6] not in ("ATOM  ", "HETATM"):
                continue
            adtype = ln.split()[-1]
            if adtype.startswith("H"):          # skip hydrogens
                continue
            try:
                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
            except ValueError:
                continue
            xs.append((x, y, z))
    n = len(xs)
    cx = sum(p[0] for p in xs)/n; cy = sum(p[1] for p in xs)/n; cz = sum(p[2] for p in xs)/n
    rg = math.sqrt(sum((p[0]-cx)**2+(p[1]-cy)**2+(p[2]-cz)**2 for p in xs)/n)
    edge = (60.0 ** (1.0/3.0)) * rg
    print(f"{edge:.2f}")

if __name__ == "__main__":
    main(sys.argv[1])
