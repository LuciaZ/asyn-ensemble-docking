#!/usr/bin/env python3
# ============================================================================
# plot_dual_contacts.py -- dedicated dual-residue contact figures
#
# The three-panel MD/holo/apo figure shows these maps small; this renders them
# large enough to read individual residue pairs, and adds the two views that
# figure can't fit:
#
#   lig{N}_dual.png       row 1: MD | Vina Holo | Vina Apo  (shared colorbar)
#                         row 2: Holo-MD | Apo-MD | Holo-Apo (diverging)
#   dual_summary.png      per-ligand totals, and the off-diagonal fraction
#                         (how much of the signal is genuine two-residue
#                         bridging rather than single-residue contact)
#
# Why the off-diagonal fraction matters: in the paper's MD maps the signal
# hugs the diagonal (the ligand touches one residue at a time), whereas the
# docked maps light up off-diagonal blocks around Y133/D135 (the pose bridges
# two residues at once). Quantifying that separation turns a visual impression
# into a number.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_dual_contacts.py
#   python plot_dual_contacts.py --ligs 38 53
#   python plot_dual_contacts.py --band 2      # |i-j|<=2 counts as diagonal
# ============================================================================
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MD   = "/home/zhihanz/data/VinaGPU/hpc_results/md_reference"
HOLO = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/analysis_v2"
APO  = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/analysis"
OUT  = "/home/zhihanz/data/VinaGPU/hpc_results/dual_contacts"
LIGS = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

RES = ['D121', 'N122', 'E123', 'A124', 'Y125', 'E126', 'M127', 'P128',
       'S129', 'E130', 'E131', 'G132', 'Y133', 'Q134', 'D135', 'Y136',
       'E137', 'P138', 'E139', 'A140']
TICK = ['D121', '', 'E123', '', 'Y125', '', 'M127', '', 'S129', '',
        'E131', '', 'Y133', '', 'D135', 'Y136', 'E137', '', 'E139', '']
TYR_IDX = [4, 12, 15]

SOURCES = [("MD", MD, "lig{}_md_fingerprint.npz"),
           ("Vina Holo", HOLO, "lig{}_fingerprint.npz"),
           ("Vina Apo", APO, "lig{}_fingerprint.npz")]


def load_dual(lig):
    out = {}
    for label, root, tmpl in SOURCES:
        p = os.path.join(root, tmpl.format(lig))
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            d = np.asarray(z["dual"], dtype=float)[:20, :20]
            out[label] = 0.5 * (d + d.T)      # symmetrise; pair order is arbitrary
        else:
            out[label] = None
    return out


def offdiag_fraction(d, band):
    """Share of total signal further than `band` residues from the diagonal."""
    if d is None or d.sum() <= 0:
        return float("nan")
    i, j = np.indices(d.shape)
    return float(d[np.abs(i - j) > band].sum() / d.sum())


def peak_pair(d):
    if d is None:
        return "-"
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return f"{RES[i]}-{RES[j]}"


def annotate(ax, tyr=True):
    ax.set_xticks(range(20)); ax.set_yticks(range(20))
    ax.set_xticklabels(TICK, rotation=60, fontsize=8, ha="right")
    ax.set_yticklabels(TICK, fontsize=8)
    ax.tick_params(labelsize=8)
    if tyr:
        for t in TYR_IDX:
            ax.axhline(t, color="white", lw=0.4, alpha=0.35)
            ax.axvline(t, color="white", lw=0.4, alpha=0.35)


def figure_for(lig, duals, outdir, band):
    have = [k for k, v in duals.items() if v is not None]
    if not have:
        return None
    vmax = max(0.75, float(np.ceil(max(duals[k].max() for k in have) * 20) / 20))

    diffs = []
    if duals["Vina Holo"] is not None and duals["MD"] is not None:
        diffs.append(("Vina Holo \u2212 MD", duals["Vina Holo"] - duals["MD"]))
    if duals["Vina Apo"] is not None and duals["MD"] is not None:
        diffs.append(("Vina Apo \u2212 MD", duals["Vina Apo"] - duals["MD"]))
    if duals["Vina Holo"] is not None and duals["Vina Apo"] is not None:
        diffs.append(("Vina Holo \u2212 Vina Apo",
                      duals["Vina Holo"] - duals["Vina Apo"]))
    dlim = (max(abs(m).max() for _, m in diffs) if diffs else 1.0) or 1.0

    nrow = 2 if diffs else 1
    fig = plt.figure(figsize=(17, 6.2 * nrow))
    gs = fig.add_gridspec(nrow, 4, width_ratios=[1, 1, 1, 0.05],
                          hspace=0.30, wspace=0.22)

    im = None
    for c, (label, _, _) in enumerate(SOURCES):
        ax = fig.add_subplot(gs[0, c])
        d = duals[label]
        if d is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=13, fontweight="bold"); continue
        im = ax.imshow(d, origin="lower", cmap="jet", vmin=0, vmax=vmax)
        annotate(ax)
        ax.set_title(f"{label}\n\u03a3={d.sum():.1f}  peak {peak_pair(d)}  "
                     f"off-diag {100*offdiag_fraction(d, band):.0f}%",
                     fontsize=12, fontweight="bold")
    if im is not None:
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
        cb.set_label("Dual-Residue Contact Probability", fontsize=10)
        cb.ax.tick_params(labelsize=8)

    if diffs:
        imd = None
        for c, (label, m) in enumerate(diffs):
            ax = fig.add_subplot(gs[1, c])
            imd = ax.imshow(m, origin="lower", cmap="RdBu_r",
                            vmin=-dlim, vmax=dlim)
            annotate(ax, tyr=False)
            for t in TYR_IDX:
                ax.axhline(t, color="k", lw=0.4, alpha=0.25)
                ax.axvline(t, color="k", lw=0.4, alpha=0.25)
            ax.set_title(f"{label}   (\u03a3\u0394={m.sum():+.1f})", fontsize=12)
        if imd is not None:
            cb = fig.colorbar(imd, cax=fig.add_subplot(gs[1, 3]))
            cb.set_label("Difference (red = first higher)", fontsize=10)
            cb.ax.tick_params(labelsize=8)

    title = f"Ligand {lig}" if lig != "_mean" else f"Mean over ligands"
    fig.suptitle(f"{title}: dual-residue contacts", fontsize=15, y=0.995)
    name = f"lig{lig}_dual.png" if lig != "_mean" else "mean_dual.png"
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def summary_figure(rows, outdir, band):
    """rows: list of (lig, {label: (total, offdiag_frac)})"""
    if not rows:
        return None
    ligs = [str(r[0]) for r in rows]
    x = np.arange(len(ligs))
    w = 0.26
    colours = {"MD": "#444444", "Vina Holo": "#065A82", "Vina Apo": "#1C7293"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for k, (label, _, _) in enumerate(SOURCES):
        tot = [r[1].get(label, (np.nan, np.nan))[0] for r in rows]
        off = [100 * r[1].get(label, (np.nan, np.nan))[1] for r in rows]
        axes[0].bar(x + (k - 1) * w, tot, w, label=label,
                    color=colours[label])
        axes[1].bar(x + (k - 1) * w, off, w, label=label,
                    color=colours[label])
    axes[0].set_ylabel("Total dual-residue signal (\u03a3)", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].tick_params(labelsize=9)
    axes[1].set_ylabel(f"Off-diagonal share (%)  |i\u2212j| > {band}", fontsize=11)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"lig {l}" for l in ligs],
                                                   rotation=45, fontsize=9)
    axes[1].tick_params(labelsize=9)
    axes[0].set_title("Dual-residue contacts across ligands", fontsize=13)
    fig.tight_layout()
    p = os.path.join(outdir, "dual_summary.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--band", type=int, default=1,
                    help="|i-j| <= band counts as diagonal (default 1)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows, stacks = [], {lbl: [] for lbl, _, _ in SOURCES}
    for lig in args.ligs:
        duals = load_dual(lig)
        miss = [k for k, v in duals.items() if v is None]
        if miss:
            print(f"lig{lig}: missing {miss}")
        p = figure_for(lig, duals, args.outdir, args.band)
        if p:
            print(f"lig{lig}: {os.path.basename(p)}", flush=True)
        cells = {}
        for label in duals:
            if duals[label] is None:
                continue
            cells[label] = (duals[label].sum(),
                            offdiag_fraction(duals[label], args.band))
            stacks[label].append(duals[label])
        rows.append((lig, cells))

    mean_duals = {k: (np.array(v).mean(0) if v else None)
                  for k, v in stacks.items()}
    if any(v is not None for v in mean_duals.values()):
        figure_for("_mean", mean_duals, args.outdir, args.band)
        print("mean_dual.png", flush=True)
    summary_figure(rows, args.outdir, args.band)
    print("dual_summary.png", flush=True)

    print(f"\n=== dual-residue contacts (off-diag = |i-j| > {args.band}) ===")
    print(f"{'lig':>4} | " + " | ".join(
        f"{lbl:>10} \u03a3 / off%" for lbl, _, _ in SOURCES))
    for lig, cells in rows:
        line = f"{lig:>4} | "
        line += " | ".join(
            (f"{cells[lbl][0]:>10.1f} / {100*cells[lbl][1]:>3.0f}%"
             if lbl in cells else f"{'-':>10} /   -")
            for lbl, _, _ in SOURCES)
        print(line)
    print("-" * 70)
    line = f"{'AVG':>4} | "
    for lbl, _, _ in SOURCES:
        vals = [c[lbl] for _, c in rows if lbl in c]
        line += (f"{np.mean([v[0] for v in vals]):>10.1f} / "
                 f"{100*np.mean([v[1] for v in vals]):>3.0f}% | " if vals
                 else f"{'-':>10} /   - | ")
    print(line.rstrip(" |"))
    print("\npeak pairs:")
    for lig in args.ligs:
        duals = load_dual(lig)
        pk = "  ".join(f"{lbl}: {peak_pair(duals[lbl]):<12}"
                       for lbl, _, _ in SOURCES if duals[lbl] is not None)
        print(f"  lig{lig:>3}  {pk}")

    print(f"\nfigures -> {args.outdir}/")


if __name__ == "__main__":
    main()
