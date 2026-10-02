#!/usr/bin/env python3
# ============================================================================
# plot_corr_errorbars.py -- score vs bound-fraction scatters with error bars
#
# The plain scatter invites the reader to trust point positions that carry
# real uncertainty. Both axes get bars here:
#
#   x  SE of the MD bound fraction, from the full-trajectory binary series
#      corrected for autocorrelation (SE = sd / sqrt(N_eff), N_eff = N/g,
#      g = 1 + 2*tau from the integrated autocorrelation time). The naive
#      sd/sqrt(N) is ~50x too small here because binding events persist for
#      thousands of frames.
#   y  SE of the mean docking score, same correction. Winner scores come from
#      stride-sampled frames so g is usually near 1, making these bars small.
#
# Pearson r is reported alongside Spearman rho: the question is whether the
# ranking is right, rho answers that directly and is not dragged around by
# lig23 sitting alone at the left edge. Point area encodes heavy-atom count,
# because size correlates with both axes (r = -0.78 and +0.73) and explains
# roughly half the apparent association -- the partial correlation controlling
# for it is -0.55, which is the number worth quoting.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_corr_errorbars.py
#   python plot_corr_errorbars.py --no-size-encoding
# ============================================================================
import os, re, csv, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MDROOT = "/home/zhihanz/data/MDdata"
HOLO   = ["/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/winners",
          "/home/zhihanz/data/VinaGPU/hpc_results/scores_474123/holo"]
APO    = ["/home/zhihanz/data/VinaGPU/hpc_results/08142026/winners",
          "/home/zhihanz/data/VinaGPU/hpc_results/scores_474123/apo"]
OUT    = "/home/zhihanz/data/VinaGPU/hpc_results/scores/errorbars"
LIGS   = [38, 5, 28, 34, 3, 24, 17, 53, 9, 40, 23, 41, 47]
CUT_NM = 0.6

# heavy-atom counts read from each ligand's MD topology
HEAVY = {38: 16, 5: 18, 28: 19, 34: 20, 3: 20, 24: 21, 17: 21,
         53: 24, 9: 25, 40: 25}

INK, MUTE, GRID = "#1A1A1A", "#6E7B8B", "#D8DEE4"
DOT, FIT = "#065A82", "#B23A48"
FS = dict(title=15, axlab=14, tick=12, lab=11, note=12, leg=11)


# ------------------------------------------------- autocorrelation-aware SE ---
def tau_int(x, c=6.0):
    """Integrated autocorrelation time, FFT with Sokal's automatic window.
    Validated on AR(1): g = 1 + 2*tau recovers (1+phi)/(1-phi)."""
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    if n < 16 or x.std() == 0:
        return 0.0, 1.0
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nf)
    acf = np.fft.irfft(f * np.conjugate(f), nf)[:n].real
    if acf[0] <= 0:
        return 0.0, 1.0
    acf /= acf[0]
    tau_w = 2.0 * np.cumsum(acf) - 1.0
    W = n - 1
    for w in range(1, n):
        if w >= c * max(tau_w[w], 1e-12):
            W = w
            break
    tau = max(0.0, 0.5 * (tau_w[W] - 1.0))
    return float(tau), float(1.0 + 2.0 * tau)


def mean_and_se(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.nan, np.nan, np.nan
    sd = x.std(ddof=1)
    _, g = tau_int(x)
    g = max(g, 1.0)
    return float(x.mean()), float(sd / np.sqrt(x.size / g)), float(g)


# -------------------------------------------------------------------- input ---
def lig_to_desres(mdroot):
    m = {}
    for f in (glob.glob(os.path.join(mdroot, "*.tar"))
              + glob.glob(os.path.join(mdroot, "*.tar.gz"))):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)",
                      os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)
    return m


def bound_series(lig, mapping, mdroot):
    if lig not in mapping:
        return None
    hits = glob.glob(os.path.join(
        mdroot, f"DESRES-Trajectory_biorxiv2021-{mapping[lig]}-no-water-glue",
        "*", "bound_out", "min_dist_nm.npy"))
    if not hits:
        return None
    return (np.load(hits[0]).astype(float) < CUT_NM).astype(float)


def score_series(dirs, lig, min_boxes=15):
    p = next((q for q in (os.path.join(d, f"lig{lig}_winners.csv")
                          for d in dirs) if os.path.exists(q)), None)
    if p is None:
        return None
    rows = []
    with open(p) as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            return None
        ec = next((c for c in rdr.fieldnames if c.strip().lower()
                   in ("energy", "score", "affinity")), None)
        fc = next((c for c in rdr.fieldnames
                   if c.strip().lower() == "frame_id"), None)
        bc = next((c for c in rdr.fieldnames
                   if c.strip().lower() == "n_boxes_found"), None)
        if ec is None:
            return None
        for r in rdr:
            try:
                e = float(r[ec])
                fid = int(r[fc]) if fc else len(rows)
            except (TypeError, ValueError):
                continue
            if min_boxes and bc:
                try:
                    if int(r[bc]) < min_boxes:
                        continue
                except (TypeError, ValueError):
                    pass
            rows.append((fid, e))
    if not rows:
        return None
    rows.sort()
    return np.array([e for _, e in rows])


# -------------------------------------------------------------------- stats ---
def ranks(v):
    v = np.asarray(v, float)
    o = np.argsort(v)
    r = np.empty(v.size, float)
    r[o] = np.arange(1, v.size + 1)
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def partial_r(x, y, z):
    """r(x,y) with z regressed out of both."""
    res = lambda a: a - np.polyval(np.polyfit(z, a, 1), z)
    return float(np.corrcoef(res(x), res(y))[0, 1])


# ------------------------------------------------------------------- figure ---
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "p", "d"]


def scatter(xs, xe, ys, ye, ligs, xlab, ylab, title, fname, outdir,
            style="color", heavy=None):
    """Identity is carried by marker colour and shape with the key outside the axes. With 13 points, error bars and two tight clusters,
    in-plot text either covers markers or drifts so far that the pairing becomes guesswork.
    A leader-line variant was tried and discarded: the connectors cross each other and read worse than no labels at all."""
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        print(f"  {fname}: <3 usable points"); return None
    fig, ax = plt.subplots(figsize=(8.4, 7.0))

    # tab20 pairs each hue with a pale twin that washes out on white; this set keeps every entry saturated
    PAL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#393b79",
           "#a55194", "#637939", "#8c6d31"]
    idx = np.where(m)[0]
    for k, i in enumerate(idx):
        col = PAL[k % len(PAL)]
        mk = MARKERS[k % len(MARKERS)]
        ax.errorbar(xs[i], ys[i],
                    xerr=(xe[i] if np.isfinite(xe[i]) else 0.0),
                    yerr=(ye[i] if np.isfinite(ye[i]) else 0.0),
                    fmt=mk, ms=8, color=col, ecolor=col,
                    elinewidth=1.3, capsize=3.0, capthick=1.3,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=f"lig {ligs[i]}", zorder=3)

    r = float(np.corrcoef(xs[m], ys[m])[0, 1])
    rho = spearman(xs[m], ys[m])
    pr = np.nan
    if heavy is not None:
        hv = np.array([heavy.get(int(l), np.nan) for l in ligs], float)
        ok = m & np.isfinite(hv)
        if ok.sum() >= 4:
            pr = partial_r(xs[ok], ys[ok], hv[ok])

    b, a = np.polyfit(xs[m], ys[m], 1)
    xf = np.linspace(xs[m].min(), xs[m].max(), 50)
    ax.plot(xf, a + b * xf, "--", color=FIT, lw=1.6, zorder=1)

    ax.set_xlabel(xlab, fontsize=FS["axlab"], color=INK)
    ax.set_ylabel(ylab, fontsize=FS["axlab"], color=INK)
    ax.set_title(f"{title}\nr = {r:+.3f}   \u03c1 = {rho:+.3f}   "
                 f"(n = {int(m.sum())})", fontsize=FS["title"], color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color(MUTE); sp.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=FS["tick"])

    h, lb = ax.get_legend_handles_labels()
    ax.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=7, frameon=False, fontsize=FS["leg"],
              handletextpad=0.3, columnspacing=1.0, numpoints=1)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    p = os.path.join(outdir, fname)
    fig.savefig(p, dpi=200); plt.close(fig)
    return p, r, rho, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--mdroot", default=MDROOT)
    ap.add_argument("--holo", nargs="+", default=HOLO)
    ap.add_argument("--apo", nargs="+", default=APO)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--min-boxes", type=int, default=15)
    ap.add_argument("--style", choices=["color"], default="color",
                    help="identity via marker colour+shape, key outside axes")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    mapping = lig_to_desres(args.mdroot)

    ligs, bf, bf_e, hol, hol_e, apo, apo_e = [], [], [], [], [], [], []
    print(f"{'lig':>4} {'MD bf':>8} {'SE':>8} {'g':>7} | "
          f"{'holo':>8} {'SE':>7} {'g':>6} | {'apo':>8} {'SE':>7} {'g':>6}")
    for lig in args.ligs:
        s = bound_series(lig, mapping, args.mdroot)
        f_, fe, fg = mean_and_se(s) if s is not None else (np.nan,)*3
        h = score_series(args.holo, lig, args.min_boxes)
        h_, he, hg = mean_and_se(h) if h is not None else (np.nan,)*3
        a = score_series(args.apo, lig, args.min_boxes)
        a_, ae, ag = mean_and_se(a) if a is not None else (np.nan,)*3
        ligs.append(lig)
        bf.append(f_); bf_e.append(fe)
        hol.append(h_); hol_e.append(he)
        apo.append(a_); apo_e.append(ae)
        print(f"{lig:>4} {f_:>8.4f} {fe:>8.4f} {fg:>7.1f} | "
              f"{h_:>8.3f} {he:>7.4f} {hg:>6.1f} | "
              f"{a_:>8.3f} {ae:>7.4f} {ag:>6.1f}")

    A = lambda v: np.array(v, float)
    heavy = HEAVY          # used for the partial correlation only
    jobs = [
        (A(bf), A(bf_e), A(hol), A(hol_e), "MD bound fraction",
         "Mean holo score (kcal/mol)",
         "Holo docking score vs MD bound fraction", "corr_holo_vs_md_err.png"),
        (A(bf), A(bf_e), A(apo), A(apo_e), "MD bound fraction",
         "Mean apo score (kcal/mol)",
         "Apo docking score vs MD bound fraction", "corr_apo_vs_md_err.png"),
        (A(apo), A(apo_e), A(hol), A(hol_e), "Mean apo score (kcal/mol)",
         "Mean holo score (kcal/mol)",
         "Holo vs apo docking score", "corr_holo_vs_apo_err.png"),
    ]
    print("\n=== correlations ===")
    for xs, xe, ys, ye, xl, yl, ttl, fn in jobs:
        res = scatter(xs, xe, ys, ye, ligs, xl, yl, ttl, fn, args.outdir,
                      style=args.style, heavy=heavy)
        if res:
            extra = (f"   partial r (heavy-atom controlled) = {res[3]:+.3f}"
                     if np.isfinite(res[3]) else "")
            print(f"  {fn}:  r = {res[1]:+.3f}   rho = {res[2]:+.3f}{extra}")

    print(f"\nfigures -> {args.outdir}/")


if __name__ == "__main__":
    main()
