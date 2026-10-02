#!/usr/bin/env python3
# ============================================================================
# error_analysis.py -- 2 sd, naive SE, and block-averaged SE
#
# Three quantities that are routinely conflated:
#
#   2 sd        spread of INDIVIDUAL observations. For a bound/unbound series
#               sd = sqrt(f(1-f)), which describes per-frame fluctuation, not
#               the uncertainty in f. Using it as an error bar overstates the
#               uncertainty enormously (f=0.5 would get +-1.0).
#   naive SE    sd/sqrt(N). Correct only for independent samples. MD frames
#               are not: binding and unbinding events persist for thousands
#               of frames, so this underestimates the error, often 10x or
#               more.
#   block SE    the defensible one. Split the series into blocks of length b,
#               take the standard error of the block means, and increase b
#               until the estimate stops growing. The plateau is the real
#               uncertainty; the ratio to the naive value is the statistical
#               inefficiency g = 1 + 2*tau, giving N_eff = N/g independent
#               samples and correlation time tau.
#
# Applied to
#   MD bound fraction  the full-trajectory binary series (min_dist < 6 A),
#                      which is where autocorrelation matters most
#   docking scores     winner scores in frame order. These come from
#                      stride-sampled frames (~50 apart for holo), so they
#                      are partly decorrelated already -- blocking quantifies
#                      how much.
#
# Outputs a table plus block-SE-vs-block-size plots (the standard diagnostic:
# a curve that has not plateaued means the trajectory is too short to quote
# an error).
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python error_analysis.py
#   python error_analysis.py --what scores
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
OUT    = "/home/zhihanz/data/VinaGPU/hpc_results/error_analysis"
LIGS   = [38, 5, 28, 34, 3, 24, 17, 53, 9, 40, 23, 41, 47]
CUT_NM = 0.6

INK, MUTE, GRID = "#1A1A1A", "#6E7B8B", "#D8DEE4"


# ------------------------------------------------------------------- blocks ---
def block_curve(x, n_points=28, min_blocks=20):
    """SE of the mean as a function of block length.
    Returns (block_lengths, se_values)."""
    x = np.asarray(x, float)
    N = x.size
    if N < min_blocks * 2:
        return np.array([]), np.array([])
    bmax = N // min_blocks
    bs = np.unique(np.geomspace(1, max(bmax, 2), n_points).astype(int))
    bs = bs[bs >= 1]
    out = []
    for b in bs:
        M = N // b
        if M < min_blocks:
            continue
        means = x[:M * b].reshape(M, b).mean(axis=1)
        # ddof=1: block means are a sample, not the population
        out.append((b, means.std(ddof=1) / np.sqrt(M)))
    if not out:
        return np.array([]), np.array([])
    a = np.array(out)
    return a[:, 0], a[:, 1]


def tau_int(x, c=6.0):
    """Integrated autocorrelation time by FFT with Sokal's automatic window.
    Preferred over reading the block-curve plateau: the plateau sits at the
    largest block lengths, where the number of blocks -- and so the precision
    of the estimate -- is smallest. Validated against AR(1) series, where
    g = 1 + 2*tau should equal (1+phi)/(1-phi)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    if n < 16 or x.std() == 0:
        return 0.0, 1.0
    nf = 1 << (int(np.ceil(np.log2(2 * n))))
    f = np.fft.rfft(x, nf)
    acf = np.fft.irfft(f * np.conjugate(f), nf)[:n].real
    if acf[0] <= 0:
        return 0.0, 1.0
    acf /= acf[0]
    # Sokal: grow the window W until W >= c * tau(W)
    csum = np.cumsum(acf)
    tau_w = 2.0 * csum - 1.0
    W = n - 1
    for w in range(1, n):
        if w >= c * max(tau_w[w], 1e-12):
            W = w
            break
    tau = max(0.0, 0.5 * (tau_w[W] - 1.0))
    return float(tau), float(1.0 + 2.0 * tau)


def plateau_se(bs, ses, frac=0.35):
    """Plateau estimate: mean of the largest-block tail, plus a flag for
    whether it has actually levelled off. A still-rising curve means the
    series is too short for the quoted error to be trusted."""
    if bs.size < 4:
        return np.nan, np.nan, False
    k = max(3, int(len(bs) * frac))
    tail = ses[-k:]
    se = float(tail.mean())
    spread = float(tail.std(ddof=1)) if k > 1 else 0.0
    # levelled off if the tail varies by <15% and is not monotonically rising
    rising = tail[-1] > tail[0] * 1.15
    return se, spread, (not rising and spread < 0.15 * max(se, 1e-12))


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = x.size
    if N < 4:
        return None
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se_naive = sd / np.sqrt(N)
    bs, ses = block_curve(x)
    se_plateau, spread, flat = plateau_se(bs, ses)
    tau, g = tau_int(x)
    g = max(g, 1.0)                      # g < 1 is unphysical
    n_eff = N / g
    se_block = sd / np.sqrt(max(n_eff, 1.0))
    return dict(N=N, mean=mean, sd=sd, two_sd=2 * sd, se_naive=se_naive,
                se_block=se_block, se_plateau=se_plateau, se_spread=spread,
                plateaued=flat, g=g, n_eff=n_eff, tau=tau, bs=bs, ses=ses)


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
    """Per-frame 0/1 bound indicator over the whole trajectory."""
    if lig not in mapping:
        return None
    hits = glob.glob(os.path.join(
        mdroot, f"DESRES-Trajectory_biorxiv2021-{mapping[lig]}-no-water-glue",
        "*", "bound_out", "min_dist_nm.npy"))
    if not hits:
        return None
    d = np.load(hits[0])
    return (np.asarray(d, float) < CUT_NM).astype(float)


def score_series(dirs, lig, min_boxes=15):
    """Winner scores in frame order."""
    p = next((q for q in (os.path.join(d, f"lig{lig}_winners.csv")
                          for d in dirs) if os.path.exists(q)), None)
    if p is None:
        return None
    rows = []
    with open(p) as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return None
        ec = next((c for c in rdr.fieldnames
                   if c.strip().lower() in ("energy", "score", "affinity")), None)
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
    rows.sort()                       # frame order matters for blocking
    return np.array([e for _, e in rows])


# ------------------------------------------------------------------- output ---
def fig_blocks(results, title, ylab, outdir, fname):
    items = [(l, r) for l, r in results.items() if r and r["bs"].size]
    if not items:
        return None
    cmap = plt.get_cmap("viridis")
    n = len(items)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for k, (lig, r) in enumerate(items):
        col = cmap(0.08 + 0.84 * k / max(n - 1, 1))
        ax.plot(r["bs"], r["ses"], "-o", ms=3.5, lw=1.8, color=col,
                label=f"lig {lig}")
        if np.isfinite(r["se_block"]):
            ax.axhline(r["se_block"], color=col, lw=0.6, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Block length (frames)", fontsize=13, color=INK)
    ax.set_ylabel(ylab, fontsize=13, color=INK)
    ax.set_title(f"{title}\nplateau = converged error; still rising = "
                 f"trajectory too short", fontsize=13, color=INK)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=11)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, fname)
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def report(results, label, unit, outdir):
    print(f"\n=== {label} ===")
    print(f"{'lig':>4} {'N':>8} {'mean':>9} {'sd':>8} {'2sd':>8} "
          f"{'SE naive':>9} {'SE corr':>9} {'SE plat':>9} {'g':>7} "
          f"{'N_eff':>8} {'tau':>8}")
    for lig, r in results.items():
        if not r:
            print(f"{lig:>4}   (no data)"); continue
        print(f"{lig:>4} {r['N']:>8d} {r['mean']:>9.4f} {r['sd']:>8.4f} "
              f"{r['two_sd']:>8.4f} {r['se_naive']:>9.5f} "
              f"{r['se_block']:>9.5f} {r['se_plateau']:>9.5f} "
              f"{r['g']:>7.1f} {r['n_eff']:>8.1f} {r['tau']:>8.1f}")
    print("  SE corr = sd/sqrt(N_eff) with N_eff from the integrated "
          "autocorrelation time; SE plat = block-curve plateau (cross-check)")
    nf = [l for l, r in results.items() if r and not r["plateaued"]]
    if nf:
        print(f"  NOTE: block curve had not plateaued for ligand(s) {nf} -- "
              f"compare SE corr against SE plat before quoting an error.")

    p = os.path.join(outdir, f"error_{label.replace(' ', '_').lower()}.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ligand", "N", f"mean_{unit}", "sd", "two_sd",
                    "se_naive", "se_corr", "se_block_plateau",
                    "se_plateau_spread", "plateaued",
                    "stat_inefficiency_g", "n_eff",
                    "corr_time_tau_frames"])
        for lig, r in results.items():
            if not r:
                continue
            w.writerow([lig, r["N"], f"{r['mean']:.6f}", f"{r['sd']:.6f}",
                        f"{r['two_sd']:.6f}", f"{r['se_naive']:.6f}",
                        f"{r['se_block']:.6f}", f"{r['se_plateau']:.6f}",
                        f"{r['se_spread']:.6f}",
                        int(r["plateaued"]), f"{r['g']:.3f}",
                        f"{r['n_eff']:.1f}", f"{r['tau']:.1f}"])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--mdroot", default=MDROOT)
    ap.add_argument("--holo", nargs="+", default=HOLO)
    ap.add_argument("--apo", nargs="+", default=APO)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--min-boxes", type=int, default=15)
    ap.add_argument("--what", choices=["bf", "scores", "all"], default="all")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    mapping = lig_to_desres(args.mdroot)

    if args.what in ("bf", "all"):
        res = {}
        for lig in args.ligs:
            s = bound_series(lig, mapping, args.mdroot)
            res[lig] = stats(s) if s is not None else None
            if s is not None:
                print(f"lig{lig}: {s.size} frames, f = {s.mean():.4f}",
                      flush=True)
        report(res, "MD bound fraction", "f", args.outdir)
        fig_blocks(res, "MD bound fraction: block-averaged standard error",
                   "SE of bound fraction", args.outdir, "blocks_bound_fraction.png")

    if args.what in ("scores", "all"):
        for label, dirs in (("Holo docking score", args.holo),
                            ("Apo docking score", args.apo)):
            res = {}
            for lig in args.ligs:
                e = score_series(dirs, lig, args.min_boxes)
                res[lig] = stats(e) if e is not None else None
            report(res, label, "kcal/mol", args.outdir)
            fig_blocks(res, f"{label}: block-averaged standard error",
                       "SE of mean score (kcal/mol)", args.outdir,
                       f"blocks_{label.split()[0].lower()}_score.png")

    print(f"\noutputs -> {args.outdir}/")


if __name__ == "__main__":
    main()
