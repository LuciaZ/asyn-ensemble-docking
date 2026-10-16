#!/usr/bin/env python3
# ============================================================================
# md_bound_fraction.py -- MD bound fraction for every ligand
#
# run_all_bound.py already stored, for each ligand's MD trajectory, the
# per-frame minimum protein-ligand heavy-atom distance (bound_out/
# min_dist_nm.npy, nm). The bound fraction is the fraction of frames below the
# same 0.6 nm cutoff used to select the holo docking frames, so no trajectory
# needs to be read again.
#
# Error: consecutive MD frames are strongly correlated, so the standard error
# is sd * sqrt(g / N) with g = 1 + 2*tau_int (FFT autocorrelation, Sokal
# window c = 5), the same estimator as error_analysis.py and stride_test.py.
#
# Output: ~/data/VinaGPU/hpc_results/error_analysis/md_bound_fraction_all.csv
#   ligand, bf, bf_se, n_frames, g
# which the 1000-frame notebook picks up automatically.
#
#   python md_bound_fraction.py
#   python md_bound_fraction.py --check error_md_bound_fraction.csv
# ============================================================================
import os, re, csv, glob, argparse
import numpy as np

MD = os.path.expanduser(os.environ.get("MDDATA", "~/data/MDdata"))
OUT = os.path.expanduser(os.environ.get(
    "BF_OUT", "~/data/VinaGPU/hpc_results/error_analysis/md_bound_fraction_all.csv"))
CUTOFF_NM = 0.6


def g_factor(x, c=5.0):
    x = np.asarray(x, float)
    n = x.size
    x = x - x.mean()
    v = np.dot(x, x) / n
    if n < 4 or v <= 0:
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (v * n)
    tau = 0.5
    for m in range(1, n):
        tau += acf[m]
        if m >= c * tau:
            break
    return max(1.0, 2.0 * tau)


def ligand_ids():
    """{ligand: DESRES id} from the archive names, e.g. 23_DESRES-...-5447828-..."""
    m = {}
    for f in glob.glob(os.path.join(MD, "*.tar")) + glob.glob(os.path.join(MD, "*.tar.gz")):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)", os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)
    return m


def min_dist_file(desres_id):
    hits = sorted(glob.glob(os.path.join(
        MD, f"DESRES-Trajectory_biorxiv2021-{desres_id}-*", "*", "bound_out", "min_dist_nm.npy")))
    return hits[0] if hits else None


def read_reference(path):
    """{ligand: bf} from an older CSV, whatever its column names are."""
    out = {}
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        lc = next((c for c in cols if c.lower() in ("lig", "ligand", "ligand_id")), cols[0])
        num = [c for c in cols if c != lc]
        for r in rdr:
            try:
                lig = int(re.sub(r"\D", "", r[lc]))
            except ValueError:
                continue
            for c in num:                     # first numeric value in (0, 1]
                try:
                    v = float(r[c])
                except (ValueError, TypeError):
                    continue
                if 0 < v <= 1:
                    out[lig] = (c, v)
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=CUTOFF_NM, help="nm")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", default=None,
                    help="older bound-fraction CSV to compare against")
    args = ap.parse_args()

    ids = ligand_ids()
    rows = []
    for lig in sorted(ids):
        f = min_dist_file(ids[lig])
        if f is None:
            print(f"lig{lig:<3d} no min_dist_nm.npy -- skipped")
            continue
        d = np.load(f).astype(float)
        d = d[np.isfinite(d)]
        b = (d < args.cutoff).astype(float)
        g = g_factor(b)
        bf = b.mean()
        se = b.std(ddof=1) * np.sqrt(g / b.size)
        rows.append(dict(ligand=lig, bf=round(bf, 5), bf_se=round(se, 5),
                         n_frames=b.size, g=round(g, 1)))
        print(f"lig{lig:<3d} bf = {bf:.4f} +/- {se:.4f}   (N = {b.size:,}, g = {g:.0f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ligand", "bf", "bf_se", "n_frames", "g"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} ligands -> {args.out}")

    if args.check:
        ref = read_reference(args.check)
        new = {r["ligand"]: r for r in rows}
        both = sorted(set(ref) & set(new))
        print(f"\ncomparison with {args.check} ({len(both)} ligands in common):")
        worst = 0.0
        for lig in both:
            col, v = ref[lig]
            dlt = new[lig]["bf"] - v
            worst = max(worst, abs(dlt))
            print(f"  lig{lig:<3d} old {v:.4f} ({col})   new {new[lig]['bf']:.4f}   "
                  f"diff {dlt:+.4f}   ({abs(dlt) / max(new[lig]['bf_se'], 1e-9):.1f} SE)")
        if both:
            print(f"largest difference {worst:.4f}")


if __name__ == "__main__":
    main()
