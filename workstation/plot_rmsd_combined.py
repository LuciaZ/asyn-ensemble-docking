#!/usr/bin/env python3
# ============================================================================
# plot_rmsd_combined.py -- all ligands on one figure instead of 10 separate
#                          two-panel plots
#
# holo_rmsd.py writes one figure per ligand, which hides the thing that
# matters most: the best-matched floor varies from ~1 A (lig3, lig40) to
# ~4.2 A (lig9), and that spread is only visible when the ligands share an
# axis. Four layouts, all reading from the existing npz files:
#
#   ridge    ridgeline, ligands ordered by median RMSD, frame-matched and
#            best-matched side by side. The ordering IS the ranking.
#   overlay  one axes per metric, all ligands as thin lines. Compact, good
#            for spotting which distributions are bimodal.
#   facet    small multiples: per ligand, both metrics in one panel.
#   bars     success-rate summary: %<3 A and %<5 A as grouped bars, which is
#            the Table 2 content in visual form.
#
# The 3 A and 5 A thresholds are drawn as reference lines throughout, since
# those are the numbers the table reports.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_rmsd_combined.py
#   python plot_rmsd_combined.py --style all
#   python plot_rmsd_combined.py --style ridge --order best
# ============================================================================
import os, re, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RMSD_DIR = "/home/zhihanz/data/VinaGPU/hpc_results/holo_rmsd"
OUT      = os.path.join(RMSD_DIR, "combined")

INK, MUTE, GRID = "#1A1A1A", "#6E7B8B", "#D8DEE4"
METRICS = [("frame_matched", "Frame-matched RMSD", 50.0),
           ("best_matched",  "Best-matched RMSD",   8.0)]
FS = dict(title=36, axlab=33, tick=27, lig=28.5, note=25.5)


def load(rmsd_dir):
    data = {}
    for f in sorted(glob.glob(os.path.join(rmsd_dir, "lig*_rmsd.npz"))):
        m = re.search(r"lig(\d+)_rmsd\.npz$", os.path.basename(f))
        if not m:
            continue
        z = np.load(f)
        d = {}
        for key, _, _ in METRICS:
            if key in z.files:
                a = np.asarray(z[key], float)
                d[key] = a[np.isfinite(a)]
        if d:
            data[int(m.group(1))] = d
    return data


def kde(x, grid):
    x = np.asarray(x, float)
    n = x.size
    if n < 2 or x.std() < 1e-9:
        return np.zeros_like(grid)
    h = max(1.06 * x.std() * n ** (-1 / 5), 1e-6)
    out = np.zeros_like(grid)
    for s in range(0, n, 20000):
        c = x[s:s + 20000]
        z = (grid[:, None] - c[None, :]) / h
        out += np.exp(-0.5 * z ** 2).sum(1)
    return out / (n * h * np.sqrt(2 * np.pi))


def tidy(ax, hide=("top", "right")):
    for s in hide:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTE); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=FS["tick"], width=0.8)


def sort_ligs(data, how):
    if how == "best":
        return sorted(data, key=lambda l: np.median(
            data[l].get("best_matched", [np.inf])))
    if how == "frame":
        return sorted(data, key=lambda l: np.median(
            data[l].get("frame_matched", [np.inf])))
    return sorted(data)                                   # by id


# ------------------------------------------------------------------- ridge ---
def fig_ridge(data, order, outdir):
    n = len(order)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(23.0, 1.55 * n + 5.5))

    for ax, (key, ttl, xmax) in zip(axes, METRICS):
        top = max((np.percentile(data[l][key], 99.5)
                   for l in order if key in data[l]), default=xmax)
        grid = np.linspace(0, min(xmax, top * 1.08), 500)
        curves = {l: kde(data[l][key], grid) for l in order if key in data[l]}
        if not curves:
            ax.axis("off"); continue
        peak = max(c.max() for c in curves.values())
        # Unfilled with fully separated rows: pitch = 1.22x peak height, so no curve crosses its neighbour's baseline
        off = peak * 1.22

        for k, lig in enumerate(order):
            if lig not in curves:
                continue
            y0 = (n - 1 - k) * off
            y = curves[lig]
            col = cmap(0.08 + 0.84 * k / max(n - 1, 1))
            ax.axhline(y0, color=GRID, lw=1.0, zorder=1)
            ax.plot(grid, y0 + y, color=col, lw=3.2, zorder=3)
            med = float(np.median(data[lig][key]))
            if grid[0] <= med <= grid[-1]:
                ax.plot([med, med], [y0, y0 + np.interp(med, grid, y)],
                        color=col, lw=1.8, ls=":", zorder=2)
            pct = 100.0 * float((data[lig][key] < 3).mean())
            ax.text(grid[-1] * 1.02, y0 + peak * 0.06, f"{pct:.1f}%",
                    ha="left", va="bottom", fontsize=FS["note"], color=MUTE)
            if ax is axes[0]:
                ax.text(-grid[-1] * 0.03, y0 + peak * 0.06, f"lig {lig}",
                        ha="right", va="bottom", fontsize=FS["lig"], color=INK)

        # on the 50 A frame-matched axis the 3 and 5 labels sit ~2 px apart,
        # so stagger their heights instead of centring both
        for thr, lab, ha, dx in ((3.0, "3 \u00c5", "right", -grid[-1]*0.008),
                                 (5.0, "5 \u00c5", "left",   grid[-1]*0.008)):
            if grid[0] <= thr <= grid[-1]:
                ax.axvline(thr, color="#B23A48", lw=2.0, ls="--", zorder=1)
                ax.text(thr + dx, (n - 1) * off + peak * 1.10, lab,
                        ha=ha, va="bottom", fontsize=FS["note"],
                        color="#B23A48")

        ax.set_xlim(-grid[-1] * 0.12, grid[-1] * 1.16)
        ax.set_ylim(-off * 0.35, (n - 1) * off + peak * 1.42)
        ax.set_yticks([])
        ax.set_xlabel(f"{ttl} (\u00c5)", fontsize=FS["axlab"], color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTE); ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=FS["tick"])

    fig.suptitle("Holo docking vs MD reference poses \u2014 "
                 "all ligands (right-hand number: % frames < 3 \u00c5)",
                 fontsize=FS["title"], color=INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, "rmsd_ridge_all.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# ----------------------------------------------------------------- overlay ---
def fig_overlay(data, order, outdir):
    cmap = plt.get_cmap("viridis")
    n = len(order)
    fig, axes = plt.subplots(1, 2, figsize=(23.0, 9.2))
    for ax, (key, ttl, xmax) in zip(axes, METRICS):
        top = max((np.percentile(data[l][key], 99.5)
                   for l in order if key in data[l]), default=xmax)
        grid = np.linspace(0, min(xmax, top * 1.08), 500)
        hi = 0
        for k, lig in enumerate(order):
            if key not in data[lig]:
                continue
            y = kde(data[lig][key], grid)
            hi = max(hi, y.max())
            col = cmap(0.08 + 0.84 * k / max(n - 1, 1))
            ax.plot(grid, y, color=col, lw=2.8)
            j = int(np.argmax(y))
            ax.annotate(f"{lig}", (grid[j], y[j]),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=FS["note"], color=col)
        for thr in (3.0, 5.0):
            if grid[0] <= thr <= grid[-1]:
                ax.axvline(thr, color="#B23A48", lw=2.0, ls="--", zorder=1)
        ax.set_ylim(0, hi * 1.18)
        ax.set_xlabel(f"{ttl} (\u00c5)", fontsize=FS["axlab"], color=INK)
        ax.set_ylabel("Density", fontsize=FS["axlab"], color=INK)
        ax.set_title(ttl, fontsize=FS["axlab"], color=INK)
        tidy(ax)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Curves labelled at their mode; dashed lines at 3 and 5 \u00c5",
                 fontsize=FS["title"], color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(outdir, "rmsd_overlay_all.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# ------------------------------------------------------------------- facet ---
def fig_facet(data, order, outdir):
    n = len(order)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k, lig in enumerate(order):
        ax = axes[k]
        ax2 = ax.twiny()
        for (key, _, xmax), col, axis in (
                (METRICS[0], "#C0504D", ax), (METRICS[1], "#065A82", ax2)):
            if key not in data[lig]:
                continue
            x = data[lig][key]
            g = np.linspace(0, min(xmax, np.percentile(x, 99.5) * 1.1), 300)
            y = kde(x, g)
            axis.plot(g, y, color=col, lw=3.0)
            axis.fill_between(g, y, color=col, alpha=0.22)
            axis.set_xlim(0, g[-1])
            axis.tick_params(axis="x", colors=col, labelsize=22)
        ax.set_title(f"lig {lig}", fontsize=FS["lig"], color=INK, pad=26)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        p3 = 100 * float((data[lig][METRICS[1][0]] < 3).mean())
        ax.text(0.97, 0.90, f"{p3:.1f}% < 3\u00c5", transform=ax.transAxes,
                ha="right", fontsize=FS["note"], color="#065A82")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Frame-matched (red, bottom axis) vs "
                 "best-matched (blue, top axis), \u00c5",
                 fontsize=FS["title"], color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(outdir, "rmsd_facet_all.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# -------------------------------------------------------------------- bars ---
def fig_bars(data, order, outdir):
    x = np.arange(len(order))
    w = 0.2
    fig, ax = plt.subplots(figsize=(1.65 * len(order) + 6.0, 9.6))
    series = [("frame_matched", 3, "#C0504D", "Frame-matched < 3 \u00c5"),
              ("frame_matched", 5, "#E39B9B", "Frame-matched < 5 \u00c5"),
              ("best_matched", 3, "#065A82", "Best-matched < 3 \u00c5"),
              ("best_matched", 5, "#8FBBD1", "Best-matched < 5 \u00c5")]
    for k, (key, thr, col, lab) in enumerate(series):
        v = [100 * float((data[l][key] < thr).mean()) if key in data[l] else np.nan
             for l in order]
        ax.bar(x + (k - 1.5) * w, v, w, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"lig {l}" for l in order],
                                         rotation=45, fontsize=FS["tick"])
    ax.set_ylabel("Frames below threshold (%)", fontsize=FS["axlab"], color=INK)
    ax.set_title("Pose-recovery success rates (Table 2 content)",
                 fontsize=FS["title"], color=INK)
    ax.legend(fontsize=FS["note"], frameon=False, ncol=2)
    tidy(ax)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, "rmsd_bars_all.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmsd-dir", default=RMSD_DIR)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--style", choices=["ridge", "overlay", "facet", "bars",
                                        "all"], default="all")
    ap.add_argument("--order", choices=["best", "frame", "id"], default="best")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = load(args.rmsd_dir)
    if not data:
        print(f"no lig*_rmsd.npz in {args.rmsd_dir}"); return
    order = sort_ligs(data, args.order)
    print(f"{len(data)} ligands, ordered by {args.order}: {order}")

    made = []
    if args.style in ("ridge", "all"):
        made.append(fig_ridge(data, order, args.outdir))
    if args.style in ("overlay", "all"):
        made.append(fig_overlay(data, order, args.outdir))
    if args.style in ("facet", "all"):
        made.append(fig_facet(data, order, args.outdir))
    if args.style in ("bars", "all"):
        made.append(fig_bars(data, order, args.outdir))
    for p in made:
        print(f"  {os.path.basename(p)}")

    print("\n=== medians and success rates ===")
    print(f"{'lig':>5} {'frame med':>10} {'<3A':>7} {'<5A':>7} "
          f"{'best med':>9} {'<3A':>7} {'<5A':>7}")
    for l in order:
        fm = data[l].get("frame_matched"); bm = data[l].get("best_matched")
        f = (f"{np.median(fm):10.2f} {100*(fm<3).mean():6.2f}% "
             f"{100*(fm<5).mean():6.2f}%") if fm is not None else " " * 26
        b = (f"{np.median(bm):9.2f} {100*(bm<3).mean():6.2f}% "
             f"{100*(bm<5).mean():6.2f}%") if bm is not None else ""
        print(f"{l:>5} {f} {b}")
    print(f"\nfigures -> {args.outdir}/")


if __name__ == "__main__":
    main()
