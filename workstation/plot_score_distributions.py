#!/usr/bin/env python3
# ============================================================================
# plot_score_distributions.py -- publication-style normalised score figures
#
# Thirteen filled KDEs overlaid in one axes occlude each other; the paper's
# Fig 6B is legible because each panel carries only three curves. Three
# layouts that stay readable at this ligand count:
#
#   ridge   (default) ridgeline / joyplot. Rows ordered by mean normalised
#           score, so the ligand ranking IS the vertical axis. Colour runs
#           along that same order, making it redundant with position rather
#           than load-bearing. Best single figure for 10+ distributions.
#   facet   small multiples, one panel per ligand, each highlighted against
#           all others drawn as thin grey context lines. Best when a reader
#           needs to inspect individual ligands.
#   overlay the original idea done properly: no fills, thin lines, curves
#           labelled at their peaks instead of a legend block.
#
# Normalisation is the paper's eq. 1 pooled min-max (--pool-ligs sets which
# ligands define xmin/xmax); the published trio is marked with * and drawn
# dashed throughout, since its docking protocol differs from the pilot's
# (Rg-scaled boxes vs the uniform 18.75 A box), so its position on the axis
# is not purely a ligand property.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_score_distributions.py --pool-ligs 47 41 23
#   python plot_score_distributions.py --style facet --pool-ligs 47 41 23
#   python plot_score_distributions.py --style all --pool-ligs 47 41 23
# ============================================================================
import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HOLO = ["/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/winners",
        "/home/zhihanz/data/VinaGPU/hpc_results/scores_474123/holo"]
APO  = ["/home/zhihanz/data/VinaGPU/hpc_results/08142026/winners",
        "/home/zhihanz/data/VinaGPU/hpc_results/scores_474123/apo"]
OUT  = "/home/zhihanz/data/VinaGPU/hpc_results/scores/paper_style"
LIGS = [38, 5, 28, 34, 3, 24, 17, 53, 9, 40, 23, 41, 47]
PUBLISHED = {23, 41, 47}

INK, MUTE, GRID = "#1A1A1A", "#6E7B8B", "#D8DEE4"
ACCENT = "#B23A48"          # published trio


def read_scores(dirs, lig, min_boxes=15):
    p = next((q for q in (os.path.join(d, f"lig{lig}_winners.csv")
                          for d in dirs) if os.path.exists(q)), None)
    if p is None:
        return None
    with open(p) as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return None
        col = next((c for c in rdr.fieldnames
                    if c.strip().lower() in ("energy", "score", "affinity")), None)
        bcol = next((c for c in rdr.fieldnames
                     if c.strip().lower() == "n_boxes_found"), None)
        if col is None:
            return None
        vals = []
        for row in rdr:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                continue
            if min_boxes and bcol:
                try:
                    if int(row[bcol]) < min_boxes:
                        continue
                except (TypeError, ValueError):
                    pass
            vals.append(v)
    return np.array(vals) if vals else None


def normalise(e, worst, best):
    span = best - worst
    return (e - worst) / span if abs(span) > 1e-12 else np.zeros_like(e)


def kde(x, grid):
    x = np.asarray(x, float)
    n = x.size
    if n < 2 or x.std() < 1e-12:
        return np.zeros_like(grid)
    h = max(1.06 * x.std() * n ** (-1 / 5), 1e-6)
    out = np.zeros_like(grid)
    for s in range(0, n, 20000):
        c = x[s:s + 20000]
        z = (grid[:, None] - c[None, :]) / h
        out += np.exp(-0.5 * z ** 2).sum(1)
    return out / (n * h * np.sqrt(2 * np.pi))


def tidy(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTE)
        s.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=13.5, width=0.8)


# ------------------------------------------------------------------- ridge ---
def fig_ridge(data, order, bounds, label, outdir):
    n = len(order)
    grid = np.linspace(-0.08, 1.08, 500)
    curves = {l: kde(data[l], grid) for l in order}
    peak = max(c.max() for c in curves.values())
    offset = peak * 0.60      # tighter overlap stacks fills into dark bands

    fig, ax = plt.subplots(figsize=(9.6, 0.70 * n + 2.6))
    cmap = plt.get_cmap("viridis")
    for k, lig in enumerate(order):            # weakest at top, strongest last
        y0 = (n - 1 - k) * offset
        y = curves[lig]
        col = cmap(0.08 + 0.84 * k / max(n - 1, 1))
        ax.fill_between(grid, y0, y0 + y, color=col, alpha=0.92,
                        zorder=n - k, lw=0)
        ax.plot(grid, y0 + y, color="white", lw=1.4, zorder=n - k + 0.1)
        ax.plot(grid, y0 + y, color=col, lw=0.9, zorder=n - k + 0.2)
        med = float(np.median(data[lig]))   # already normalised
        ax.plot([med, med], [y0, y0 + np.interp(med, grid, y)],
                color="white", lw=1.0, zorder=n - k + 0.3)
        ax.text(-0.11, y0 + peak * 0.06, f"lig {lig}",
                ha="right", va="bottom", fontsize=14.2, color=INK)
        ax.text(1.10, y0 + peak * 0.06, f"{med:.2f}",
                ha="left", va="bottom", fontsize=12.8, color=MUTE)

    ax.set_xlim(-0.30, 1.22)
    ax.set_ylim(-offset * 0.35, (n - 1) * offset + peak * 1.15)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Normalized docking score", fontsize=16.5, color=INK)
    ax.text(1.10, (n - 1) * offset + peak * 1.02, "median",
            ha="left", va="bottom", fontsize=12.8, color=MUTE, style="italic")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTE); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=13.5, width=0.8)
    for v in np.arange(0.2, 1.0, 0.2):
        ax.axvline(v, color=GRID, lw=0.6, zorder=0)

    ax.set_title(f"AutoDock Vina ({label}) — normalized docking scores\n"
                 f"pooled min-max: worst {bounds[0]:.2f} / "
                 f"best {bounds[1]:.2f} kcal/mol",
                 fontsize=18, color=INK, pad=12)
    fig.tight_layout()
    p = os.path.join(outdir, f"scores_ridge_{label}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- facet ---
def fig_facet(data, order, bounds, label, outdir):
    n = len(order)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    grid = np.linspace(-0.08, 1.08, 400)
    curves = {l: kde(data[l], grid) for l in order}
    peak = max(c.max() for c in curves.values())

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    for k, lig in enumerate(order):
        ax = axes[k]
        for other in order:                    # context
            if other != lig:
                ax.plot(grid, curves[other], color=GRID, lw=0.7, zorder=1)
        col = cmap(0.08 + 0.84 * k / max(n - 1, 1))
        ax.fill_between(grid, curves[lig], color=col, alpha=0.30, zorder=2, lw=0)
        ax.plot(grid, curves[lig], color=col, lw=1.8, zorder=3)
        med = float(np.median(data[lig]))   # already normalised
        ax.axvline(med, color=col, lw=1.0, ls=":", zorder=4)
        ax.set_title(f"lig {lig}", fontsize=15.8, color=INK)
        ax.text(0.97, 0.88, f"{med:.2f}", transform=ax.transAxes,
                ha="right", fontsize=12.8, color=MUTE)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, peak * 1.12)
        ax.set_yticks([])
        tidy(ax, ("top", "right", "left"))
    for ax in axes[n:]:
        ax.axis("off")
    fig.supxlabel("Normalized docking score", fontsize=16.5, color=INK)
    fig.suptitle(f"AutoDock Vina ({label}) — per-ligand score distributions "
                 f"(grey = all other ligands)", fontsize=18.8, color=INK)
    fig.tight_layout()
    p = os.path.join(outdir, f"scores_facet_{label}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------- overlay ---
def fig_overlay(data, order, bounds, label, outdir):
    grid = np.linspace(-0.08, 1.08, 500)
    curves = {l: kde(data[l], grid) for l in order}
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    cmap = plt.get_cmap("viridis")
    n = len(order)
    for k, lig in enumerate(order):
        col = cmap(0.08 + 0.84 * k / max(n - 1, 1))
        y = curves[lig]
        ax.plot(grid, y, color=col, lw=1.4, zorder=2)
        j = int(np.argmax(y))
        ax.annotate(f"{lig}", (grid[j], y[j]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=12.8, color=col)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, max(c.max() for c in curves.values()) * 1.16)
    ax.set_xlabel("Normalized docking score", fontsize=16.5, color=INK)
    ax.set_ylabel("Density", fontsize=16.5, color=INK)
    ax.set_title(f"AutoDock Vina ({label}) — normalized docking scores\n"
                 f"curves labelled at their mode",
                 fontsize=18, color=INK)
    tidy(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(outdir, f"scores_overlay_{label}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--holo", nargs="+", default=HOLO)
    ap.add_argument("--apo", nargs="+", default=APO)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--pool-ligs", nargs="*", type=int, default=None)
    ap.add_argument("--min-boxes", type=int, default=15)
    ap.add_argument("--style", choices=["ridge", "facet", "overlay", "all"],
                    default="ridge")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for label, dirs in (("Holo", args.holo), ("Apo", args.apo)):
        raw = {}
        for lig in args.ligs:
            e = read_scores(dirs, lig, args.min_boxes)
            if e is not None:
                raw[lig] = e
        if not raw:
            print(f"{label}: no scores found"); continue

        pool = [l for l in (args.pool_ligs or raw) if l in raw]
        allv = np.concatenate([raw[l] for l in pool])
        bounds = (float(allv.max()), float(allv.min()))     # worst, best
        print(f"{label}: {len(raw)} ligands; bounds from {sorted(pool)} "
              f"-> worst {bounds[0]:.2f} / best {bounds[1]:.2f}")

        norm = {l: normalise(v, *bounds) for l, v in raw.items()}
        order = sorted(norm, key=lambda l: float(np.mean(norm[l])))

        made = []
        if args.style in ("ridge", "all"):
            made.append(fig_ridge(norm, order, bounds, label, args.outdir))
        if args.style in ("facet", "all"):
            made.append(fig_facet(norm, order, bounds, label, args.outdir))
        if args.style in ("overlay", "all"):
            made.append(fig_overlay(norm, order, bounds, label, args.outdir))
        for p in made:
            print(f"  {os.path.basename(p)}")

    print(f"\nfigures -> {args.outdir}/")


if __name__ == "__main__":
    main()
