#!/usr/bin/env python3
# ============================================================================
# plot_pilot_paper.py -- paper-style per-ligand interaction plots
#
# Uses docking_tools.plotting.plot_ensemble_ii (the same function used for the
# 23/41/47 figures) so the output matches the paper's style exactly: x-axis
# D121..E139 residue labels, four contact types (hphob green, aro black,
# hbond red, charge blue) overlaid, one figure per ligand.
#
# Uniform ensemble (decision B): the 5005-frame set is treated as ONE cluster
# with weight 1.0. plot_ensemble_ii computes  run_dict[ctype].mean(1).T @ weights
# so we reshape each (n_frames, 20) matrix to (1, n_frames, 20) and pass
# weights = np.array([1.0]).
#
# Also draws the dual-contact (20x20) heatmap per ligand via subplot_snsheat.
#
# Run (diffdock env, from /home/zhihanz/data/scripts):
#   python plot_pilot_paper.py
# ============================================================================
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/zhihanz/data/scripts")
from docking_tools import plotting

D    = "/home/zhihanz/data/VinaGPU/hpc_results/08142026/analysis"
LIGS = [38, 28, 5, 34, 24, 3, 17, 53, 9, 40]
OUT  = os.path.join(D, "paper_style")
os.makedirs(OUT, exist_ok=True)

RES_SEQ = ['D121','','E123','','Y125','','M127','','S129','',
           'E131','','Y133','','D135','Y136','E137','','E139','']

def main():
    for lig in LIGS:
        f = os.path.join(D, f"lig{lig}_fingerprint.npz")
        if not os.path.exists(f):
            print(f"lig{lig}: no npz, skip"); continue
        z = np.load(f, allow_pickle=True)

        # build run_dict in the shape plot_ensemble_ii expects:
        #   run_dict[ctype] : (n_cluster=1, n_frames, n_res=20)
        run_dict = {}
        for ctype in ["hphob", "aro", "hbond", "charge"]:
            arr = np.asarray(z[ctype], dtype=float)      # (n_frames, 20)
            if arr.ndim != 2:
                arr = np.zeros((1, 20))
            run_dict[ctype] = arr[None, :, :]            # (1, n_frames, 20)
        weights = np.array([1.0])

        # paper-style interaction fingerprint figure
        try:
            plotting.plot_ensemble_ii(run_dict, weights, res_sequence=RES_SEQ)
            plt.suptitle(f"lig {lig} — apo (uniform ensemble)", y=1.02)
            out = os.path.join(OUT, f"lig{lig}_interactions.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"lig{lig}: {out}")
        except Exception as e:
            print(f"lig{lig}: plot_ensemble_ii ERR: {e}")
            plt.close("all")

        # dual-contact heatmap
        dual = np.asarray(z["dual"], dtype=float)
        if dual.shape == (20, 20) and dual.sum() > 0:
            fig, ax = plt.subplots(figsize=(7, 6))
            try:
                plotting.subplot_snsheat(dual, ax=ax, cbar=True, cbar_ax=None, cbar_s=12)
            except Exception:
                im = ax.imshow(dual, cmap="jet", vmin=0, vmax=0.75)
                fig.colorbar(im, ax=ax)
            ax.set_xticks(range(20)); ax.set_xticklabels(RES_SEQ, rotation=45, size=8)
            ax.set_yticks(range(20)); ax.set_yticklabels(RES_SEQ, size=8)
            ax.invert_yaxis()
            ax.set_title(f"lig {lig} — dual-residue contact")
            out = os.path.join(OUT, f"lig{lig}_dual.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"lig{lig}: {out}")

    print(f"\npaper-style figures -> {OUT}/")

if __name__ == "__main__":
    main()
