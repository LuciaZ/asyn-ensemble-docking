#!/usr/bin/env python3
# ============================================================================
# plot_apo_vs_holo_paper.py -- paper-style interaction fingerprints, apo vs holo
#
# Three figure types per ligand:
#   1. *_side.png    two plot_ensemble_ii panels stacked (apo top, holo bottom).
#                    plot_ensemble_ii builds its own figure, so each is rendered
#                    to a buffer and the two images are composited -- the paper
#                    style is therefore untouched (hphob green / aro black /
#                    hbond red / charge blue, D121..E139 axis).
#   2. *_overlay.png all four contact types in ONE axes, apo solid vs holo
#                    dashed in the matching colour. Easiest read for spotting
#                    where the two ensembles diverge.
#   3. *_delta.png   holo minus apo per residue, four types, zero line. Makes
#                    the aromatic excess (~+11% in the sums) locatable.
#
# Plus summary_overlay.png / summary_delta.png: mean +/- sd over the ligands.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_apo_vs_holo_paper.py
#   python plot_apo_vs_holo_paper.py --ligs 38 53      # subset
#   python plot_apo_vs_holo_paper.py --no-side         # skip the composite
# ============================================================================
import os, sys, io, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, "/home/zhihanz/data/scripts")
from docking_tools import plotting

APO  = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/analysis"
HOLO = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/analysis_v2"
OUT  = "/home/zhihanz/data/VinaGPU/hpc_results/holo_08192026/apo_vs_holo_paper"
LIGS = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]

RES_SEQ = ['D121','','E123','','Y125','','M127','','S129','',
           'E131','','Y133','','D135','Y136','E137','','E139','']
TYR_IDX = [4, 12, 15]                  # Y125, Y133, Y136 (0-based)

# paper colours
CTYPES = [("hphob",  "hydrophobic", "green"),
          ("aro",    "aromatic",    "black"),
          ("hbond",  "H-bond",      "red"),
          ("charge", "charge",      "blue")]


def per_residue(z, key):
    """(n_frames, 20) -> (20,) mean; bf already (20,)."""
    a = np.asarray(z[key], dtype=float)
    return a.mean(axis=0) if a.ndim == 2 else a


def run_dict_of(z):
    """Shape for plot_ensemble_ii: run_dict[ctype] = (1, n_frames, 20)."""
    rd = {}
    for key, _, _ in CTYPES:
        arr = np.asarray(z[key], dtype=float)
        if arr.ndim != 2:
            arr = np.zeros((1, 20))
        rd[key] = arr[None, :, :]
    return rd


def render_ensemble_ii(z, title):
    """Call plot_ensemble_ii and return the figure as an RGBA array."""
    plotting.plot_ensemble_ii(run_dict_of(z), np.array([1.0]),
                              res_sequence=RES_SEQ)
    plt.suptitle(title, y=1.02)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close("all")
    buf.seek(0)
    return mpimg.imread(buf)


def fig_side(lig, za, zh, outdir):
    """Stack the two paper-style panels as images (style preserved verbatim)."""
    try:
        img_a = render_ensemble_ii(za, f"lig {lig} — APO receptor ensemble")
        img_h = render_ensemble_ii(zh, f"lig {lig} — HOLO receptor ensemble")
    except Exception as e:
        print(f"  lig{lig}: plot_ensemble_ii ERR: {e}")
        plt.close("all")
        return None
    h1, h2 = img_a.shape[0], img_h.shape[0]
    w = max(img_a.shape[1], img_h.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(w / 150, (h1 + h2) / 150))
    for ax, img in zip(axes, (img_a, img_h)):
        ax.imshow(img)
        ax.axis("off")
    fig.subplots_adjust(hspace=0.02, left=0, right=1, top=1, bottom=0)
    p = os.path.join(outdir, f"lig{lig}_side.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_overlay(lig, za, zh, outdir):
    """All four types in one axes; apo solid, holo dashed."""
    x = np.arange(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, colour in CTYPES:
        ya, yh = per_residue(za, key), per_residue(zh, key)
        ax.plot(x, ya, "-",  color=colour, lw=1.9, label=f"{label} apo")
        ax.plot(x, yh, "--", color=colour, lw=1.6, alpha=0.8,
                label=f"{label} holo")
    for t in TYR_IDX:
        ax.axvline(t, color="grey", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(RES_SEQ, rotation=45, size=8)
    ax.set_ylabel("contact fraction", size=10)
    ax.set_title(f"lig {lig} — interaction fingerprint, apo (solid) vs holo (dashed)",
                 size=11)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, f"lig{lig}_overlay.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_delta(lig, za, zh, outdir):
    """holo - apo per residue."""
    x = np.arange(20)
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, colour in CTYPES:
        d = per_residue(zh, key) - per_residue(za, key)
        ax.plot(x, d, "-o", color=colour, lw=1.5, ms=3, label=label)
    ax.axhline(0, color="k", lw=0.8)
    for t in TYR_IDX:
        ax.axvline(t, color="grey", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(RES_SEQ, rotation=45, size=8)
    ax.set_ylabel("holo - apo", size=10)
    ax.set_title(f"lig {lig} — difference (positive = holo higher)", size=11)
    ax.legend(fontsize=8, frameon=False, ncol=4)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, f"lig{lig}_delta.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_summary(stacks, outdir, n):
    x = np.arange(20)
    # overlay of means
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, colour in CTYPES:
        A, H = stacks[key]
        ma, sa = A.mean(0), A.std(0)
        mh = H.mean(0)
        ax.plot(x, ma, "-",  color=colour, lw=1.9, label=f"{label} apo")
        ax.plot(x, mh, "--", color=colour, lw=1.6, alpha=0.8,
                label=f"{label} holo")
        ax.fill_between(x, ma - sa, ma + sa, color=colour, alpha=0.10)
    for t in TYR_IDX:
        ax.axvline(t, color="grey", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(RES_SEQ, rotation=45, size=8)
    ax.set_ylabel("contact fraction", size=10)
    ax.set_title(f"mean over {n} ligands (band = sd, apo) — apo solid vs holo dashed",
                 size=11)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    p1 = os.path.join(outdir, "summary_overlay.png")
    fig.savefig(p1, dpi=150); plt.close(fig)

    # mean delta with spread across ligands
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, colour in CTYPES:
        A, H = stacks[key]
        D = H - A
        m, s = D.mean(0), D.std(0)
        ax.plot(x, m, "-o", color=colour, lw=1.5, ms=3, label=label)
        ax.fill_between(x, m - s, m + s, color=colour, alpha=0.12)
    ax.axhline(0, color="k", lw=0.8)
    for t in TYR_IDX:
        ax.axvline(t, color="grey", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(RES_SEQ, rotation=45, size=8)
    ax.set_ylabel("holo - apo", size=10)
    ax.set_title(f"mean difference over {n} ligands (band = sd)", size=11)
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    p2 = os.path.join(outdir, "summary_delta.png")
    fig.savefig(p2, dpi=150); plt.close(fig)
    return p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligs", nargs="*", type=int, default=LIGS)
    ap.add_argument("--no-side", action="store_true",
                    help="skip the plot_ensemble_ii composite")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    stacks = {k: ([], []) for k, _, _ in CTYPES}
    n_done = 0
    for lig in args.ligs:
        fa = os.path.join(APO,  f"lig{lig}_fingerprint.npz")
        fh = os.path.join(HOLO, f"lig{lig}_fingerprint.npz")
        if not (os.path.exists(fa) and os.path.exists(fh)):
            print(f"lig{lig}: npz missing, skip"); continue
        za = np.load(fa, allow_pickle=True)
        zh = np.load(fh, allow_pickle=True)

        if not args.no_side:
            fig_side(lig, za, zh, OUT)
        fig_overlay(lig, za, zh, OUT)
        fig_delta(lig, za, zh, OUT)
        for key, _, _ in CTYPES:
            stacks[key][0].append(per_residue(za, key))
            stacks[key][1].append(per_residue(zh, key))
        n_done += 1
        print(f"lig{lig}: side/overlay/delta done", flush=True)

    if n_done:
        st = {k: (np.array(v[0]), np.array(v[1])) for k, v in stacks.items()}
        fig_summary(st, OUT, n_done)

        print("\n=== per-type peak residue and apo/holo correlation ===")
        print(f"{'lig':>4} | " + " | ".join(f"{lab[:6]:>14}" for _, lab, _ in CTYPES))
        for lig in args.ligs:
            fa = os.path.join(APO,  f"lig{lig}_fingerprint.npz")
            fh = os.path.join(HOLO, f"lig{lig}_fingerprint.npz")
            if not (os.path.exists(fa) and os.path.exists(fh)):
                continue
            za = np.load(fa, allow_pickle=True)
            zh = np.load(fh, allow_pickle=True)
            cells = []
            for key, _, _ in CTYPES:
                ya, yh = per_residue(za, key), per_residue(zh, key)
                pa, ph = int(np.argmax(ya)) + 1, int(np.argmax(yh)) + 1
                r = (np.corrcoef(ya, yh)[0, 1]
                     if ya.std() and yh.std() else float("nan"))
                cells.append(f"r{pa:>2}/r{ph:<2} {r:>5.3f}")
            print(f"{lig:>4} | " + " | ".join(f"{c:>14}" for c in cells))

    print(f"\nfigures -> {OUT}/")


if __name__ == "__main__":
    main()
