#!/usr/bin/env python3
# ============================================================================
# plot_interactions_by_type.py -- one panel per interaction type per ensemble
#
# Replaces the single overlay that carried all four interaction types for both
# ensembles at once. In that figure hydrophobic contacts (peaking near 0.5)
# compressed charge contacts (peaking near 0.07) into the baseline, and the
# apo/holo solid-dashed pairing plus four colours meant eight lines competing
# in one axes.
#
# Layout: 2 rows (apo, holo) x 4 columns (hydrophobic, aromatic, H-bond,
# charge), ten ligand curves per panel. The y-axis is shared DOWN each column,
# so apo and holo are directly comparable for a given interaction type, while
# each type still gets a scale suited to its own magnitude.
#
# Ligand colour runs along heavy-atom count (viridis), since contact counts
# scale with ligand size -- ordering the colours by size makes that trend
# readable instead of scrambling it across a categorical palette.
#
# The three tyrosines (Y125, Y133, Y136) are marked: aromatic contacts are
# non-zero only at those positions, and Y136 is easy to miss because the
# every-other-residue axis labelling hides it.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_interactions_by_type.py
#   python plot_interactions_by_type.py --separate    # 8 individual files
#   python plot_interactions_by_type.py --free-y      # per-panel y scaling
# ============================================================================
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

APO  = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/analysis"
HOLO = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/analysis_v2"
OUT  = "/home/zhihanz/data/VinaGPU/hpc_results/interactions_by_type"

# ascending heavy-atom count, read from each ligand's MD topology
LIGS  = [38, 5, 28, 34, 3, 24, 17, 53, 9, 40]
HEAVY = {38: 16, 5: 18, 28: 19, 34: 20, 3: 20, 24: 21, 17: 21,
         53: 24, 9: 25, 40: 25}

TYPES = [("hphob",  "Hydrophobic contacts"),
         ("aro",    "Aromatic stacking"),
         ("hbond",  "Hydrogen bonding"),
         ("charge", "Charge-charge contacts")]
ENSEMBLES = [("Apo", APO, "lig{}_fingerprint.npz"),
             ("Holo", HOLO, "lig{}_fingerprint.npz")]

RES_LABEL = ['D121', '', 'E123', '', 'Y125', '', 'M127', '', 'S129', '',
             'E131', '', 'Y133', '', 'D135', 'Y136', 'E137', '', 'E139', '']
TYR_IDX = [4, 12, 15]

INK, MUTE = "#1A1A1A", "#6E7B8B"
FS = dict(title=17, panel=15, axlab=14, tick=12, leg=12)


def per_residue(z, key):
    a = np.asarray(z[key], dtype=float)
    return a.mean(axis=0) if a.ndim == 2 else a


def load(root, tmpl, ligs):
    out = {}
    for lig in ligs:
        p = os.path.join(root, tmpl.format(lig))
        if os.path.exists(p):
            out[lig] = np.load(p, allow_pickle=True)
    return out


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTE); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=FS["tick"])


def draw(ax, data, ligs, key, colours, ymax=None, xlabels=True, ylabel=False):
    x = np.arange(20)
    for lig in ligs:
        if lig not in data or key not in data[lig].files:
            continue
        ax.plot(x, per_residue(data[lig], key), "-", lw=1.6,
                color=colours[lig], label=f"lig {lig}")
    for t in TYR_IDX:
        ax.axvline(t, color="#CCCCCC", lw=0.7, ls=":", zorder=0)
    if ymax:
        ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(x)
    if xlabels:
        ax.set_xticklabels(RES_LABEL, rotation=60, fontsize=FS["tick"] - 1,
                           ha="right")
        ax.set_xlabel("Residue", fontsize=FS["axlab"], color=INK)
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel("Contact fraction", fontsize=FS["axlab"], color=INK)
    tidy(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--apo", default=APO)
    ap.add_argument("--holo", default=HOLO)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--separate", action="store_true",
                    help="write 8 individual files instead of one grid")
    ap.add_argument("--free-y", action="store_true",
                    help="scale each panel independently (default: shared "
                         "down each column so apo and holo are comparable)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ligs = sorted(args.ligs, key=lambda l: (HEAVY.get(l, 999), l))
    cmap = plt.get_cmap("viridis")
    colours = {l: cmap(0.06 + 0.86 * k / max(len(ligs) - 1, 1))
               for k, l in enumerate(ligs)}

    loaded = {}
    for label, root, tmpl in ENSEMBLES:
        root = args.apo if label == "Apo" else args.holo
        loaded[label] = load(root, tmpl, ligs)
        missing = [l for l in ligs if l not in loaded[label]]
        print(f"{label}: {len(loaded[label])}/{len(ligs)} ligands"
              + (f"; missing {missing}" if missing else ""))
    if not any(loaded.values()):
        print("no fingerprint npz found"); return

    # shared y per interaction type so apo/holo can be compared
    ymax = {}
    for key, _ in TYPES:
        vals = [per_residue(z, key).max()
                for lab in loaded for z in loaded[lab].values()
                if key in z.files]
        ymax[key] = (None if args.free_y or not vals
                     else float(np.ceil(max(vals) * 20) / 20))

    made = []
    if args.separate:
        for lab in loaded:
            for key, name in TYPES:
                fig, ax = plt.subplots(figsize=(8.0, 5.2))
                draw(ax, loaded[lab], ligs, key, colours, ymax[key],
                     xlabels=True, ylabel=True)
                ax.set_title(f"{name} \u2014 {lab} ensemble",
                             fontsize=FS["panel"], color=INK)
                ax.legend(fontsize=FS["leg"], frameon=False, ncol=2)
                fig.tight_layout()
                p = os.path.join(args.outdir,
                                 f"{key}_{lab.lower()}.png")
                fig.savefig(p, dpi=200); plt.close(fig)
                made.append(p)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.6))
        for r, (lab, _, _) in enumerate(ENSEMBLES):
            for c, (key, name) in enumerate(TYPES):
                ax = axes[r, c]
                draw(ax, loaded.get(lab, {}), ligs, key, colours, ymax[key],
                     xlabels=(r == 1), ylabel=(c == 0))
                ax.set_title(f"{name}\n{lab} ensemble",
                             fontsize=FS["panel"], color=INK)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(ligs),
                       frameon=False, fontsize=FS["leg"],
                       bbox_to_anchor=(0.5, -0.005))
        fig.suptitle("Per-residue interaction fingerprints \u2014 "
                     "10 ligands, apo (top) vs holo (bottom); "
                     "y-axis shared within each column",
                     fontsize=FS["title"], color=INK, y=0.995)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        p = os.path.join(args.outdir, "interactions_by_type_grid.png")
        fig.savefig(p, dpi=200); plt.close(fig)
        made.append(p)

    for p in made:
        print(f"  {os.path.basename(p)}")

    print("\n=== summed contact fraction per type ===")
    print(f"{'lig':>4} {'heavy':>6} | " + " | ".join(
        f"{n.split()[0][:6]:>13}" for _, n in TYPES))
    print(f"{'':>4} {'':>6} | " + " | ".join(f"{'apo':>6} {'holo':>6}"
                                             for _ in TYPES))
    for lig in ligs:
        row = f"{lig:>4} {HEAVY.get(lig, 0):>6} | "
        cells = []
        for key, _ in TYPES:
            vs = []
            for lab in ("Apo", "Holo"):
                z = loaded.get(lab, {}).get(lig)
                vs.append(per_residue(z, key).sum()
                          if z is not None and key in z.files else np.nan)
            cells.append(f"{vs[0]:>6.3f} {vs[1]:>6.3f}")
        print(row + " | ".join(cells))

    print(f"\nfigures -> {args.outdir}/")


if __name__ == "__main__":
    main()
