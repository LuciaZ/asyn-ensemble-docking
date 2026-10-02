#!/usr/bin/env python3
# ============================================================================
# plot_md_vs_docked.py -- JCIM Fig 8 style three-panel comparison
#
# Row A: per-residue interaction fingerprints, three ensembles side by side
#        (MD | AutoDock Vina Holo | AutoDock Vina Apo), shared y-axis so the
#        systematic MD/docked offset is visible rather than hidden by
#        per-panel rescaling.
# Row B: dual-residue contact matrices for the same three, shared colorbar.
#
# Also prints the MD-vs-docked Pearson r and RMSE per interaction type -- the
# numbers behind the paper's correlation table (Supp. Table 3 layout).
#
# The MD panel sits 2-3x lower than the docked panels in absolute magnitude.
# That is real, not a bug: MD samples the ligand's actual loosely-bound
# geometry, whereas docking reports an optimised pose inside a box. The paper
# shows the same offset (MD hydrophobic peak ~0.28 vs docked ~0.5).
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_md_vs_docked.py
#   python plot_md_vs_docked.py --ligs 38 53
#   python plot_md_vs_docked.py --free-y        # per-panel y scaling instead
# ============================================================================
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MD   = "/home/zhihanz/data/VinaGPU/hpc_results/md_reference"
HOLO = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/analysis_v2"
APO  = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/analysis"
OUT  = "/home/zhihanz/data/VinaGPU/hpc_results/md_vs_docked"
LIGS = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

# residue 1..20 == resSeq 121..140; three tyrosines at slots 4, 12, 15
RES_LABEL = ['D121', '', 'E123', '', 'Y125', '', 'M127', '', 'S129', '',
             'E131', '', 'Y133', '', 'D135', 'Y136', 'E137', '', 'E139', '']
TYR_IDX = [4, 12, 15]

# paper colours
CTYPES = [("hphob",  "Hydrophobic Contacts",   "green"),
          ("aro",    "Aromatic Stacking",      "black"),
          ("charge", "Charge-Charge Contacts", "blue"),
          ("hbond",  "Hydrogen Bonding",       "red")]

PANELS = [("MD Ensemble",                MD,   "lig{}_md_fingerprint.npz"),
          ("AutoDock Vina Holo Ensemble", HOLO, "lig{}_fingerprint.npz"),
          ("AutoDock Vina Apo Ensemble",  APO,  "lig{}_fingerprint.npz")]


def per_residue(z, key):
    """(n_frames, 20) -> (20,) mean; already-(20,) passes through."""
    a = np.asarray(z[key], dtype=float)
    return a.mean(axis=0) if a.ndim == 2 else a


def load_panels(lig):
    """[(label, npz or None), ...] in panel order."""
    out = []
    for label, root, tmpl in PANELS:
        p = os.path.join(root, tmpl.format(lig))
        out.append((label, np.load(p, allow_pickle=True)
                    if os.path.exists(p) else None))
    return out


def figure_for(lig, panels, outdir, shared_y=True):
    prof = {}                                     # label -> {ctype: (20,)}
    duals = {}
    for label, z in panels:
        if z is None:
            prof[label] = None; duals[label] = None; continue
        prof[label] = {k: per_residue(z, k) for k, _, _ in CTYPES}
        d = np.asarray(z["dual"], dtype=float)
        duals[label] = d[:20, :20]

    have = [l for l in prof if prof[l] is not None]
    if not have:
        return None

    ymax = 0.6
    if shared_y:
        peak = max(v.max() for l in have for v in prof[l].values())
        ymax = max(0.6, float(np.ceil(peak * 10) / 10))
    dmax = max((duals[l].max() for l in have if duals[l] is not None),
               default=0.75)
    dmax = max(0.75, float(np.ceil(dmax * 20) / 20))

    x = np.arange(20)
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06],
                          height_ratios=[1, 1.25], hspace=0.35, wspace=0.25)

    # ---- row A: per-residue fingerprints -------------------------------
    for c, (label, _, _) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, c])
        ax.set_title(label, fontsize=13, fontweight="bold")
        if prof[label] is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=MUTE_GREY)
            ax.set_xticks([]); ax.set_yticks([]); continue
        for key, legend, colour in CTYPES:
            ax.plot(x, prof[label][key], "-", color=colour, lw=1.8,
                    label=legend if c == 0 else None)
        for t in TYR_IDX:
            ax.axvline(t, color="grey", lw=0.6, ls=":", zorder=0)
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(RES_LABEL, rotation=60, fontsize=8, ha="right")
        ax.set_xlabel("Residue", fontsize=11)
        if c == 0:
            ax.set_ylabel("Probability", fontsize=12)
            ax.legend(fontsize=9, frameon=False, loc="upper left")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=9)

    # ---- row B: dual-residue contact maps ------------------------------
    im = None
    for c, (label, _, _) in enumerate(PANELS):
        ax = fig.add_subplot(gs[1, c])
        if duals[label] is None:
            ax.set_xticks([]); ax.set_yticks([]); continue
        im = ax.imshow(duals[label], origin="lower", cmap="jet",
                       vmin=0, vmax=dmax, aspect="equal")
        ax.set_xticks(x); ax.set_yticks(x)
        ax.set_xticklabels(RES_LABEL, rotation=45, fontsize=8, ha="right")
        ax.set_yticklabels(RES_LABEL if c == 0 else [], fontsize=8)
        ax.tick_params(labelsize=8)
    if im is not None:
        cax = fig.add_subplot(gs[1, 3])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Dual-Residue Contact Probability", fontsize=10)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Ligand {lig}: MD vs docked ensembles", fontsize=15, y=0.98)
    p = os.path.join(outdir, f"lig{lig}_md_vs_docked.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


MUTE_GREY = "#888888"


def stats_row(lig, panels):
    """MD-vs-docked Pearson r and RMSE per interaction type + dual."""
    ref = dict(panels)[PANELS[0][0]]
    if ref is None:
        return []
    md_prof = {k: per_residue(ref, k) for k, _, _ in CTYPES}
    md_bf = np.asarray(ref["bf"], float)
    md_dual = np.asarray(ref["dual"], float)[:20, :20]

    rows = []
    for label, z in panels[1:]:
        if z is None:
            continue
        cells = {}
        # ligand contacts == bound fraction profile
        for name, a, b in ([("bf", md_bf, np.asarray(z["bf"], float))]
                           + [(k, md_prof[k], per_residue(z, k))
                              for k, _, _ in CTYPES]
                           + [("dual", md_dual.ravel(),
                               np.asarray(z["dual"], float)[:20, :20].ravel())]):
            if a.std() < 1e-12 or b.std() < 1e-12:
                cells[name] = (float("nan"), float(np.sqrt(((a - b) ** 2).mean())))
            else:
                cells[name] = (float(np.corrcoef(a, b)[0, 1]),
                               float(np.sqrt(((a - b) ** 2).mean())))
        rows.append((label, cells))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--free-y", action="store_true",
                    help="scale each panel's y-axis independently")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_stats, mean_stack = {}, {lbl: {k: [] for k, _, _ in CTYPES}
                                 for lbl, _, _ in PANELS}
    dual_stack = {lbl: [] for lbl, _, _ in PANELS}

    for lig in args.ligs:
        panels = load_panels(lig)
        missing = [l for l, z in panels if z is None]
        if missing:
            print(f"lig{lig}: missing {missing}")
        p = figure_for(lig, panels, args.outdir, shared_y=not args.free_y)
        if p:
            print(f"lig{lig}: {os.path.basename(p)}", flush=True)
        all_stats[lig] = stats_row(lig, panels)
        for label, z in panels:
            if z is None:
                continue
            for k, _, _ in CTYPES:
                mean_stack[label][k].append(per_residue(z, k))
            dual_stack[label].append(np.asarray(z["dual"], float)[:20, :20])

    # ---- mean-over-ligands figure --------------------------------------
    class Fake:
        def __init__(self, d): self.d = d
        def __getitem__(self, k): return self.d[k]
    mean_panels = []
    for label, _, _ in PANELS:
        if not dual_stack[label]:
            mean_panels.append((label, None)); continue
        d = {k: np.array(mean_stack[label][k]).mean(0) for k, _, _ in CTYPES}
        d["dual"] = np.array(dual_stack[label]).mean(0)
        d["bf"] = np.zeros(20)
        mean_panels.append((label, Fake(d)))
    if any(z is not None for _, z in mean_panels):
        p = figure_for("_mean", mean_panels, args.outdir,
                       shared_y=not args.free_y)
        if p:
            os.replace(p, os.path.join(args.outdir, "mean_md_vs_docked.png"))
            print("mean_md_vs_docked.png", flush=True)

    # ---- correlation table ---------------------------------------------
    order = ["bf", "hphob", "aro", "hbond", "charge", "dual"]
    head = {"bf": "Ligand", "hphob": "Hydrophob", "aro": "Aromatic",
            "hbond": "H-bond", "charge": "Charge", "dual": "Dual-res"}
    print("\n=== MD vs docked:  r (RMSE) ===")
    print(f"{'lig':>4} {'ensemble':>10} | " +
          " | ".join(f"{head[k]:>13}" for k in order))
    acc = {}
    for lig in args.ligs:
        for label, cells in all_stats.get(lig, []):
            tag = "Holo" if "Holo" in label else "Apo"
            line = f"{lig:>4} {tag:>10} | "
            line += " | ".join(f"{cells[k][0]:>6.2f} ({cells[k][1]:.3f})"
                               for k in order)
            print(line)
            acc.setdefault(tag, {k: [] for k in order})
            for k in order:
                if not np.isnan(cells[k][0]):
                    acc[tag][k].append(cells[k])
    print("-" * 100)
    for tag, cells in acc.items():
        line = f"{'AVG':>4} {tag:>10} | "
        line += " | ".join(
            f"{np.mean([c[0] for c in cells[k]]):>6.2f} "
            f"({np.mean([c[1] for c in cells[k]]):.3f})" for k in order)
        print(line)

    print(f"\nfigures -> {args.outdir}/")


if __name__ == "__main__":
    main()
