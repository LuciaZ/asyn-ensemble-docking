#!/usr/bin/env python3
# ============================================================================
# analyze_docking_scores.py -- score distributions, normalisation, correlations
#
# Normalisation follows the paper (eq. 1): min-max with xmin/xmax the worst
# and best scores across ALL docking calculations of a given method, POOLED
# over ligands. Pooling preserves between-ligand differences (Fig 6B shows
# lig47's density right of lig23's); per-ligand normalisation would force
# every ligand's best pose to 1.0 and destroy that signal.
#
# COMPARABILITY: the paper's bounds come from Ligand 47, Fasudil, Ligand 23.
# Use --pool-ligs 47 41 23 so the published trio sets the scale while extra
# ligands are plotted against it, or --bounds-holo/--bounds-apo to hard-set
# published values. Bounds used are printed and written into score_table.csv.
#
# MD BOUND FRACTION is whole-trajectory occupancy (bound_frame_idx /
# min_dist_nm lengths) -- the 53-80% values in bound_all.log -- NOT the
# per-residue `bf` in the fingerprint npz, which is ~1 by construction.
# ============================================================================
import os, re, glob, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOLO_WIN = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/winners"
APO_WIN  = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/winners"
MDROOT   = "/home/zhihanz/data/MDdata"
OUT      = "/home/zhihanz/data/VinaGPU/hpc_results/scores"
LIGS     = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

ENSEMBLES = [("holo", HOLO_WIN), ("apo", APO_WIN)]
CMAP = plt.get_cmap("tab20")          # 13 ligands exceed tab10's 10 colours,
                                      # where k % 10 made lig5 and lig41 share
                                      # orange and made the legend unreadable
PUBLISHED = {23, 41, 47}              # drawn dashed: paper protocol, not pilot


def read_scores(win_dirs, lig, min_boxes=0):
    """min_boxes drops frames whose best-of-20 rested on too few boxes.
    A frame searched at 1 box is a best-of-1; those degenerate values sit at
    the weak tail and would otherwise set xmin (lig23 apo worst: -2.50 with
    them, -3.90 without)."""
    if isinstance(win_dirs, str):
        win_dirs = [win_dirs]
    p = next((q for q in (os.path.join(d, f"lig{lig}_winners.csv")
                          for d in win_dirs) if os.path.exists(q)), None)
    if p is None:
        return None
    with open(p) as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            return None
        col = next((c for c in rdr.fieldnames
                    if c.strip().lower() in ("energy", "score", "affinity")), None)
        if col is None:
            print(f"    lig{lig}: no energy column (cols={rdr.fieldnames})")
            return None
        bcol = next((c for c in rdr.fieldnames
                     if c.strip().lower() == "n_boxes_found"), None)
        vals, dropped = [], 0
        for row in rdr:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if min_boxes and bcol:
                try:
                    if int(row[bcol]) < min_boxes:
                        dropped += 1
                        continue
                except (TypeError, ValueError):
                    pass
            vals.append(v)
        if dropped:
            print(f"    lig{lig}: dropped {dropped} frames with <{min_boxes} boxes")
    return np.array(vals) if vals else None


def lig_to_desres(mdroot):
    m = {}
    for f in (glob.glob(os.path.join(mdroot, "*.tar"))
              + glob.glob(os.path.join(mdroot, "*.tar.gz"))):
        mo = re.match(r"(\d+)_DESRES-Trajectory_biorxiv2021-(\d+)",
                      os.path.basename(f))
        if mo:
            m[int(mo.group(1))] = mo.group(2)
    return m


def md_bound_fraction(lig, mapping, mdroot):
    if lig not in mapping:
        return np.nan
    hits = glob.glob(os.path.join(
        mdroot, f"DESRES-Trajectory_biorxiv2021-{mapping[lig]}-no-water-glue",
        "*", "bound_out"))
    if not hits:
        return np.nan
    fb = os.path.join(hits[0], "bound_frame_idx.npy")
    fm = os.path.join(hits[0], "min_dist_nm.npy")
    if not (os.path.exists(fb) and os.path.exists(fm)):
        return np.nan
    n_bound = int(np.load(fb, mmap_mode="r").shape[0])
    n_total = int(np.load(fm, mmap_mode="r").shape[0])
    return n_bound / n_total if n_total else np.nan


def pooled_bounds(scores_by_lig, only=None):
    """(worst, best) over poses of the ligands that define the scale."""
    use = [v for k, v in scores_by_lig.items()
           if v is not None and (only is None or k in set(only))]
    if not use:
        return None
    allv = np.concatenate(use)
    return (float(allv.max()), float(allv.min()))


def normalise(e, bounds):
    e = np.asarray(e, float)
    worst, best = bounds
    span = best - worst
    return (e - worst) / span if abs(span) > 1e-12 else np.zeros_like(e)


def kde(x, grid, bw=None):
    x = np.asarray(x, float); n = x.size
    if n < 2 or x.std() < 1e-12:
        return np.zeros_like(grid)
    h = max(bw or 1.06 * x.std() * n ** (-1 / 5), 1e-6)
    z = (grid[:, None] - x[None, :]) / h
    return np.exp(-0.5 * z ** 2).sum(1) / (n * h * np.sqrt(2 * np.pi))


def fig_kde_overlay(scores, ligs, label, bounds, outdir, scope):
    lo, hi = 0.0, 1.0
    for lig in ligs:
        e = scores.get(lig)
        if e is None:
            continue
        b = bounds if scope == "pooled" else (float(e.max()), float(e.min()))
        n = normalise(e, b)
        lo, hi = min(lo, float(n.min())), max(hi, float(n.max()))
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, lig in enumerate(ligs):
        e = scores.get(lig)
        if e is None:
            continue
        b = bounds if scope == "pooled" else (float(e.max()), float(e.min()))
        y = kde(normalise(e, b), grid)
        c = CMAP(k % 20)
        ls = "--" if lig in PUBLISHED else "-"
        ax.plot(grid, y, ls, lw=2.0, color=c,
                label=f"lig {lig}" + (" *" if lig in PUBLISHED else ""))
        ax.fill_between(grid, y, color=c, alpha=0.06)
    ax.set_xlabel("Normalized docking score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"AutoDock Vina ({label.capitalize()}) - "
                 f"{len(ligs)} ligands, {scope} min-max\n"
                 f"worst {bounds[0]:.2f} / best {bounds[1]:.2f} kcal/mol",
                 fontsize=12)
    ax.legend(fontsize=8, frameon=False, ncol=2,
              title="* = published trio (sets the scale)",
              title_fontsize=8)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, f"scores_hist_{label}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_raw_grid(all_scores, ligs, outdir):
    n = len(ligs); ncol = 5; nrow = int(np.ceil(n / 5))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, lig in zip(axes, ligs):
        for (label, _), colour in zip(ENSEMBLES, ("#065A82", "#C0504D")):
            e = all_scores[label].get(lig)
            if e is None:
                continue
            ax.hist(e, bins=40, density=True, histtype="step", lw=1.6,
                    color=colour, label=label)
        ax.set_title(f"lig {lig}", fontsize=11)
        ax.tick_params(labelsize=8)
    for ax in axes[len(ligs):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, frameon=False)
    fig.supxlabel("Docking score (kcal/mol)", fontsize=11)
    fig.supylabel("Density", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "scores_hist_grid.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_affinity(rows, outdir):
    ligs = [r["lig"] for r in rows]
    metrics = [("MD Bound Fraction",
                [r["md_bf"] for r in rows], [0.0] * len(rows)),
               ("Vina Holo <norm score>",
                [r["holo_normmean"] for r in rows],
                [r.get("holo_normsd") or 0.0 for r in rows]),
               ("Vina Apo <norm score>",
                [r["apo_normmean"] for r in rows],
                [r.get("apo_normsd") or 0.0 for r in rows])]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ligs)); w = 0.26
    for k, (name, vals, errs) in enumerate(metrics):
        v = np.array([np.nan if q is None else q for q in vals], float)
        e = np.array([0.0 if q is None else q for q in errs], float)
        if np.all(np.isnan(v)):
            continue
        sc = np.nanmax(v)
        if sc and abs(sc) > 1e-12:
            v, e = v / sc, e / sc
        ax.bar(x + (k - 1) * w, v, w, yerr=e, capsize=2, label=name,
               color=("#444444", "#065A82", "#C0504D")[k],
               error_kw=dict(lw=0.9, ecolor="#333333"))
    ax.set_xticks(x); ax.set_xticklabels([f"lig {l}" for l in ligs],
                                         rotation=45, fontsize=9)
    ax.set_ylabel("Predicted ligand affinity\n(scaled to strongest = 1)",
                  fontsize=11)
    ax.set_title("Per-ligand affinity proxies", fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "predicted_affinity.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_scatter(xs, ys, ligs, xlab, ylab, title, fname, outdir):
    m = ~(np.isnan(xs) | np.isnan(ys))
    if m.sum() < 3:
        print(f"  {fname}: <3 usable points, skipped"); return None
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(xs[m], ys[m], s=70, color="#065A82", zorder=3)
    for xi, yi, l in zip(xs, ys, ligs):
        if not (np.isnan(xi) or np.isnan(yi)):
            ax.annotate(str(l), (xi, yi), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color="#333333")
    r = np.corrcoef(xs[m], ys[m])[0, 1]
    b, a = np.polyfit(xs[m], ys[m], 1)
    xf = np.linspace(xs[m].min(), xs[m].max(), 50)
    ax.plot(xf, a + b * xf, "--", color="#C0504D", lw=1.4,
            label=f"r = {r:.3f}  (n={m.sum()})")
    ax.set_xlabel(xlab, fontsize=11); ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, fname)
    fig.savefig(p, dpi=150); plt.close(fig)
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--holo", nargs="+", default=[HOLO_WIN])
    ap.add_argument("--apo", nargs="+", default=[APO_WIN])
    ap.add_argument("--mdroot", default=MDROOT)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--scope", choices=["pooled", "perlig"], default="pooled")
    ap.add_argument("--pool-ligs", nargs="*", type=int, default=None,
                    help="ligands that DEFINE the min-max bounds")
    ap.add_argument("--bounds-holo", nargs=2, type=float, default=None,
                    metavar=("WORST", "BEST"))
    ap.add_argument("--bounds-apo", nargs=2, type=float, default=None,
                    metavar=("WORST", "BEST"))
    ap.add_argument("--min-boxes", type=int, default=15,
                    help="skip frames whose winner came from fewer than this "
                         "many residue boxes (0 = keep all)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dirs = {"holo": args.holo, "apo": args.apo}
    all_scores = {lbl: {} for lbl, _ in ENSEMBLES}
    for lbl, _ in ENSEMBLES:
        for lig in args.ligs:
            e = read_scores(dirs[lbl], lig, args.min_boxes)
            if e is not None:
                all_scores[lbl][lig] = e
        print(f"{lbl}: loaded {len(all_scores[lbl])}/{len(args.ligs)} ligands "
              f"from {', '.join(dirs[lbl])}")

    override = {"holo": args.bounds_holo, "apo": args.bounds_apo}
    bounds, provenance = {}, {}
    for lbl, _ in ENSEMBLES:
        if override[lbl] is not None:
            bounds[lbl] = (float(override[lbl][0]), float(override[lbl][1]))
            provenance[lbl] = "user-supplied"
        else:
            b = pooled_bounds(all_scores[lbl], args.pool_ligs)
            if b is None:
                print(f"{lbl}: no scores for pooling ligands {args.pool_ligs}")
                bounds[lbl] = None; provenance[lbl] = "unavailable"; continue
            bounds[lbl] = b
            used = sorted(set(all_scores[lbl]) & set(args.pool_ligs)) \
                if args.pool_ligs else sorted(all_scores[lbl])
            provenance[lbl] = f"pooled over ligands {used}"
    for lbl, _ in ENSEMBLES:
        if bounds[lbl] is None:
            continue
        w_, b_ = bounds[lbl]
        print(f"{lbl}: bounds worst {w_:.2f} / best {b_:.2f} kcal/mol "
              f"[{provenance[lbl]}]")
        if args.pool_ligs:
            outside = [l for l, v in all_scores[lbl].items()
                       if v is not None and (v.min() < b_ or v.max() > w_)]
            if outside:
                print(f"    NOTE: ligand(s) {sorted(outside)} fall outside "
                      f"these bounds -> normalised values <0 or >1")

    mapping = lig_to_desres(args.mdroot)
    rows = []
    for lig in args.ligs:
        r = {"lig": lig, "md_bf": md_bound_fraction(lig, mapping, args.mdroot)}
        for lbl, _ in ENSEMBLES:
            e = all_scores[lbl].get(lig)
            if e is None:
                r.update({f"{lbl}_n": 0, f"{lbl}_mean": None, f"{lbl}_sd": None,
                          f"{lbl}_best": None, f"{lbl}_worst": None,
                          f"{lbl}_median": None, f"{lbl}_normmean": None,
                          f"{lbl}_normsd": None})
                continue
            if args.scope == "pooled":
                b = bounds[lbl]
                if b is None:
                    b = (float(e.max()), float(e.min()))
            else:
                b = (float(e.max()), float(e.min()))
            nrm = normalise(e, b)
            r.update({f"{lbl}_normsd": float(nrm.std()),
                      f"{lbl}_n": e.size, f"{lbl}_mean": float(e.mean()),
                      f"{lbl}_sd": float(e.std()), f"{lbl}_best": float(e.min()),
                      f"{lbl}_worst": float(e.max()),
                      f"{lbl}_median": float(np.median(e)),
                      f"{lbl}_normmean": float(nrm.mean())})
        rows.append(r)

    print(f"\n=== docking scores (kcal/mol), {args.scope} min-max (paper eq. 1) ===")
    print(f"{'lig':>4} {'MD bf':>7} | {'holo mean':>10} {'best':>7} "
          f"{'norm':>6} | {'apo mean':>9} {'best':>7} {'norm':>6}")
    for r in rows:
        hm = f"{r['holo_mean']:.2f}" if r['holo_mean'] is not None else "-"
        hb = f"{r['holo_best']:.2f}" if r['holo_best'] is not None else "-"
        hn = f"{r['holo_normmean']:.3f}" if r['holo_normmean'] is not None else "-"
        am = f"{r['apo_mean']:.2f}" if r['apo_mean'] is not None else "-"
        ab = f"{r['apo_best']:.2f}" if r['apo_best'] is not None else "-"
        an = f"{r['apo_normmean']:.3f}" if r['apo_normmean'] is not None else "-"
        bf = f"{r['md_bf']:.3f}" if not np.isnan(r['md_bf']) else "-"
        print(f"{r['lig']:>4} {bf:>7} | {hm:>10} {hb:>7} {hn:>6} | "
              f"{am:>9} {ab:>7} {an:>6}")

    keys = ["lig", "md_bf"] + [f"{l}_{s}" for l, _ in ENSEMBLES
                               for s in ("n", "mean", "sd", "median",
                                         "best", "worst", "normmean", "normsd")]
    with open(os.path.join(args.outdir, "bounds_used.txt"), "w") as f:
        for lbl, _ in ENSEMBLES:
            if bounds[lbl]:
                f.write(f"{lbl} min-max bounds: worst={bounds[lbl][0]:.3f} "
                        f"best={bounds[lbl][1]:.3f} ({provenance[lbl]})\n")
    with open(os.path.join(args.outdir, "score_table.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    for lbl, _ in ENSEMBLES:
        if all_scores[lbl]:
            fig_kde_overlay(all_scores[lbl],
                            [l for l in args.ligs if l in all_scores[lbl]],
                            lbl, bounds[lbl] or (0.0, 1.0),
                            args.outdir, args.scope)
            print(f"  scores_hist_{lbl}.png")
    if any(all_scores[l] for l, _ in ENSEMBLES):
        fig_raw_grid(all_scores, args.ligs, args.outdir)
        print("  scores_hist_grid.png")
    fig_affinity(rows, args.outdir)
    print("  predicted_affinity.png")

    ligs = np.array([r["lig"] for r in rows])
    bf   = np.array([r["md_bf"] for r in rows], float)
    hol  = np.array([r["holo_mean"] if r["holo_mean"] is not None else np.nan
                     for r in rows], float)
    apo  = np.array([r["apo_mean"] if r["apo_mean"] is not None else np.nan
                     for r in rows], float)

    print("\n=== correlations ===")
    for xs, ys, xl, yl, ttl, fn in (
        (bf, hol, "MD bound fraction", "Mean holo score (kcal/mol)",
         "Holo docking score vs MD bound fraction", "corr_holo_vs_md.png"),
        (bf, apo, "MD bound fraction", "Mean apo score (kcal/mol)",
         "Apo docking score vs MD bound fraction", "corr_apo_vs_md.png"),
        (apo, hol, "Mean apo score (kcal/mol)", "Mean holo score (kcal/mol)",
         "Holo vs apo docking score", "corr_holo_vs_apo.png")):
        res = fig_scatter(xs, ys, ligs, xl, yl, ttl, fn, args.outdir)
        if res:
            print(f"  {fn}:  r = {res[1]:.3f}")

    print(f"\noutputs -> {args.outdir}/")


if __name__ == "__main__":
    main()
