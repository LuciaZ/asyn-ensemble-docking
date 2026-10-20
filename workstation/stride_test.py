#!/usr/bin/env python3
# ============================================================================
# stride_test.py -- is 1000 frames per ligand enough?
#
# For every ligand with a complete docked ensemble, take 1000 evenly spaced
# frames from the full ensemble (5005 apo / 5000 holo / ~20000 trio) and ask
# whether the subset reproduces the full-ensemble results within its own
# error bars, for
#
#   1. the mean best-of-20 docking score per ligand
#   2. the correlation of that score with the MD bound fraction
#   3. per-residue interaction propensities (hphob / aro / hbond / charge)
#
# Errors: frames are correlated, so sd/sqrt(N) understates the error. Every
# standard error here is sd * sqrt(g / N), with g = 1 + 2*tau_int from the
# FFT autocorrelation and Sokal's automatic window (c = 5) -- the method used
# in error_analysis.py.
#
# Criterion: a subset estimate "agrees" when the full-ensemble value lies
# within 2 * SE(subset). That is the question that decides whether the
# remaining ligands can be docked on 1000 frames: would the error bar you
# would publish from 1000 frames have covered the 5000-frame answer?
#
# Inputs (all already on the workstation):
#   ~/data/docked_trajectories/<job>/lig{N}_{xyz.npy,top.pdb,groups.json,frames.npy}
#   winners CSVs per job (frame_id + energy)
#   MD bound fractions: --bf-csv, or found automatically under
#   ~/data/VinaGPU/hpc_results/error_analysis/ (any CSV with a ligand column
#   and a column whose name contains "bf")
#
# Per-frame contacts are computed once with docking_tools (exactly as in the
# notebooks) and cached, so re-runs are fast.
#
# Run (diffdock env, from ~/data/scripts):
#   python stride_test.py --jobs apo_pilot10 --ligs 38        # quick check
#   python stride_test.py                                     # all jobs
# Outputs: ~/data/VinaGPU/hpc_results/stride_test/
# ============================================================================
import os, re, sys, csv, glob, json, argparse
import numpy as np

H = os.path.expanduser(os.environ.get("HPC_RESULTS", "~/data/VinaGPU/hpc_results"))
TRAJ = os.path.expanduser(os.environ.get("DOCKED_TRAJ", "~/data/docked_trajectories"))
OUT = os.path.join(H, "stride_test")
N_SUB = 1000
TYPES = ("hphob", "aro", "hbond", "charge")
RES = ["D121", "Q122", "E123", "A124", "Y125", "E126", "M127", "P128", "S129",
       "E130", "E131", "G132", "Y133", "Q134", "D135", "Y136", "E137", "P138",
       "E139", "A140"]

WINNERS = {
    "apo_pilot10": f"{H}/08142026/winners",
    "holo_pilot10": f"{H}/holo_08192026/winners",
    "apo_remaining37": f"{H}/apo_remaining/winners",
    "trio_apo": f"{H}/scores_474123/apo",
}
ENERGY_COLS = ("energy", "score", "best_energy", "vina_score", "affinity")


# ------------------------------------------------------------------ errors ---
def g_factor(x, c=5.0):
    """Statistical inefficiency g = 1 + 2*tau_int (Sokal window)."""
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return 1.0
    x = x - x.mean()
    v = np.dot(x, x) / n
    if v <= 0:
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (v * n)
    tau = 0.5
    for m in range(1, n):
        tau += acf[m]
        if m >= c * tau:
            break
    return max(1.0, 2.0 * tau)


def mean_se(x):
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return float(x.mean()) if n else np.nan, np.nan, 1.0
    g = g_factor(x)
    return float(x.mean()), float(x.std(ddof=1) * np.sqrt(g / n)), g


def subset_idx(n, k=N_SUB):
    """k evenly spaced frames across the whole ensemble."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


# ------------------------------------------------------------------- inputs ---
def ligands(job):
    return sorted(int(re.search(r"lig(\d+)_xyz", os.path.basename(f)).group(1))
                  for f in glob.glob(f"{TRAJ}/{job}/lig*_xyz.npy"))


def frame_ids(job, lig):
    return np.load(f"{TRAJ}/{job}/lig{lig}_frames.npy")


def scores_in_traj_order(job, lig):
    """Energies aligned to the trajectory frame order."""
    p = f"{WINNERS[job]}/lig{lig}_winners.csv"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        rdr = csv.DictReader(fh)
        col = next((c for c in ENERGY_COLS if c in rdr.fieldnames), None)
        if col is None:
            raise KeyError(f"{p}: no energy column in {rdr.fieldnames}")
        e = {int(r["frame_id"]): float(r[col]) for r in rdr if r.get(col)}
    fids = frame_ids(job, lig)
    out = np.array([e.get(int(f), np.nan) for f in fids])
    return out


def heavy_atoms(job, lig):
    g = json.load(open(f"{TRAJ}/{job}/lig{lig}_groups.json"))
    return sum(1 for t in g["ligand_autodock_types"] if t not in ("H", "HD", "HS"))


def _read_bf_csv(p):
    """{lig: (bf, se)} from a CSV with a ligand column and a 'bf' column."""
    with open(p) as fh:
        rdr = csv.DictReader(fh)
        names = rdr.fieldnames or []
        cols = [c.lower() for c in names]
        lc = next((names[i] for i, c in enumerate(cols)
                   if c in ("lig", "ligand", "ligand_id")), None)
        bc = next((names[i] for i, c in enumerate(cols)
                   if "bf" in c and "se" not in c and "sd" not in c
                   and "err" not in c), None)
        sc = next((names[i] for i, c in enumerate(cols)
                   if "bf" in c and ("se" in c or "err" in c)), None)
        if not (lc and bc):
            return {}, None
        out = {}
        for r in rdr:
            try:
                lig = int(re.sub(r"\D", "", r[lc]))
                out[lig] = (float(r[bc]), float(r[sc]) if sc and r.get(sc) else np.nan)
            except (ValueError, TypeError):
                continue
        return out, (lc, bc, sc)


def load_bf(path=None):
    """MD bound fractions. With no path, every CSV under error_analysis/ is
    tried and the one covering the most ligands wins, so an older partial
    table never shadows the full one."""
    cands = [path] if path else sorted(glob.glob(f"{H}/error_analysis/*.csv"))
    best, best_p, best_cols = {}, None, None
    for p in cands:
        if not p or not os.path.exists(p):
            continue
        out, cols = _read_bf_csv(p)
        if len(out) > len(best):
            best, best_p, best_cols = out, p, cols
    if best:
        lc, bc, sc = best_cols
        print(f"MD bound fractions: {len(best)} ligands from {best_p} "
              f"(columns {lc}, {bc}{', ' + sc if sc else ''})")
    return best


# ---------------------------------------------------------------- contacts ---
def compute_contacts(job, lig):
    """Per-frame, per-residue contacts, computed exactly as in the notebooks."""
    import mdtraj as md
    for p in ("~/data/scripts", "~/data/scripts/docking_analysis"):
        p = os.path.expanduser(p)
        if os.path.isdir(os.path.join(p, "docking_tools")) and p not in sys.path:
            sys.path.insert(0, p)
    from docking_tools import trajectory_analysis as ta

    top = md.load_topology(f"{TRAJ}/{job}/lig{lig}_top.pdb")
    traj = md.Trajectory(np.load(f"{TRAJ}/{job}/lig{lig}_xyz.npy"), top)
    g = json.load(open(f"{TRAJ}/{job}/lig{lig}_groups.json"))
    n_res = traj.atom_slice(traj.top.select("protein")).n_residues
    L, nf = n_res, traj.n_frames
    shp = (nf, n_res)

    def safe(f, what):
        # Always call: hbond() also counts protein-donor -> ligand-acceptor
        # H-bonds, so a ligand without donors of its own still has H-bond
        # contacts. An empty group list is only filled with zeros if the
        # docking_tools function itself rejects it.
        try:
            return np.asarray(f(), float).reshape(shp)
        except Exception as e:
            print(f"    lig{lig} {what}: {type(e).__name__}: {e} -- zeros")
            return np.zeros(shp)

    c = {}
    c["hphob"] = np.asarray(ta.hphob_contacts(traj, L), float).reshape(shp)
    c["aro"] = safe(lambda: ta.aro_contacts(traj, ligand_rings=g["aromatic_rings"]), "aro")
    c["hbond"] = safe(lambda: ta.hbond(traj, L, lig_hbond_donors=g["hbond_donors"]), "hbond")
    c["charge"] = safe(lambda: ta.charge_contacts(traj, Ligand_Pos_Charges=g["positive_charges"]), "charge")
    return c


def contacts_cached(job, lig):
    p = f"{OUT}/cache/{job}_lig{lig}_contacts.npz"
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in TYPES}
    c = compute_contacts(job, lig)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **c)
    return c


# ----------------------------------------------------------------- analysis ---
def score_test(job, lig):
    e = scores_in_traj_order(job, lig)
    if e is None:
        return None
    ok = np.isfinite(e)
    full = e[ok]
    sub = e[subset_idx(e.size)]
    sub = sub[np.isfinite(sub)]
    mf, sf, gf = mean_se(full)
    ms, ss, gs = mean_se(sub)
    return dict(job=job, lig=lig, n_full=full.size, mean_full=mf, se_full=sf,
                g_full=gf, n_sub=sub.size, mean_sub=ms, se_sub=ss, g_sub=gs,
                delta=ms - mf, agree=abs(ms - mf) <= 2 * ss)


def propensity_test(job, lig, contacts):
    nf = contacts["hphob"].shape[0]
    idx = subset_idx(nf)
    rows, prof = [], {}
    for t in TYPES:
        a = contacts[t]
        pf, ef, ps, es = [], [], [], []
        for r in range(a.shape[1]):
            m, s, _ = mean_se(a[:, r]); pf.append(m); ef.append(s)
            m, s, _ = mean_se(a[idx, r]); ps.append(m); es.append(s)
        pf, ef, ps, es = map(np.array, (pf, ef, ps, es))
        prof[t] = (pf, ef, ps, es)
        live = pf > 0.01                  # residues with a real signal
        d = np.abs(ps - pf)
        es_safe = np.where(np.isfinite(es) & (es > 0), es, np.inf)
        within = d <= 2 * es_safe
        r_prof = (np.corrcoef(pf, ps)[0, 1]
                  if pf.std() > 0 and ps.std() > 0 else np.nan)
        rows.append(dict(job=job, lig=lig, type=t, n_full=nf, n_sub=idx.size,
                         n_live=int(live.sum()), profile_r=r_prof,
                         max_abs_delta=float(d.max()),
                         frac_within_2se=float(within[live].mean()) if live.any() else np.nan,
                         peak_full=RES[int(np.argmax(pf))] if pf.max() > 0 else "-",
                         peak_sub=RES[int(np.argmax(ps))] if ps.max() > 0 else "-",
                         # a different argmax only matters if the full-ensemble
                         # peak is no longer statistically tied with the top
                         peak_ok=bool(pf.max() <= 0 or np.argmax(ps) == np.argmax(pf)
                                      or ps[np.argmax(pf)] + 2 * es[np.argmax(pf)] >= ps.max())))
    return rows, prof


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan


def partial_r(x, y, z):
    """corr(x, y | z) from the three pairwise correlations."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else np.nan


# -------------------------------------------------------------------- plots ---
def plot_scores(rows, outdir=None):
    import matplotlib.pyplot as plt
    jobs = sorted({r["job"] for r in rows})
    fig, axes = plt.subplots(1, len(jobs), figsize=(4.2 * len(jobs), 4.2), squeeze=False)
    for ax, job in zip(axes[0], jobs):
        rr = [r for r in rows if r["job"] == job]
        x = np.array([r["mean_full"] for r in rr]); y = np.array([r["mean_sub"] for r in rr])
        xe = np.array([r["se_full"] for r in rr]); ye = np.array([r["se_sub"] for r in rr])
        col = ["tab:blue" if r["agree"] else "tab:red" for r in rr]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor="0.6", lw=0.8, zorder=1)
        ax.scatter(x, y, c=col, s=22, zorder=2)
        lo, hi = min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        n_ok = sum(r["agree"] for r in rr)
        ax.set_title(f"{job}\n{n_ok}/{len(rr)} agree within 2 SE", fontsize=10)
        ax.set_xlabel("mean score, full ensemble (kcal/mol)")
        ax.set_ylabel(f"mean score, {N_SUB} frames (kcal/mol)")
    fig.tight_layout()
    if outdir:
        fig.savefig(f"{outdir}/scores_full_vs_{N_SUB}.png", dpi=150)
    return fig


def plot_md_vs_score(rows, bf, outdir=None):
    import matplotlib.pyplot as plt
    rr = [r for r in rows if r["lig"] in bf]
    if len(rr) < 3:
        return None, []
    jobs = sorted({r["job"] for r in rr})
    fig, axes = plt.subplots(1, len(jobs), figsize=(4.6 * len(jobs), 4.2), squeeze=False)
    summary = []
    for ax, job in zip(axes[0], jobs):
        j = [r for r in rr if r["job"] == job]
        if len(j) < 3:
            ax.set_visible(False); continue
        x = np.array([bf[r["lig"]][0] for r in j]); xe = np.array([bf[r["lig"]][1] for r in j])
        ha = np.array([heavy_atoms(job, r["lig"]) for r in j])
        for key, mk, lab in (("full", "o", "full ensemble"), ("sub", "s", f"{N_SUB} frames")):
            y = np.array([r[f"mean_{key}"] for r in j]); ye = np.array([r[f"se_{key}"] for r in j])
            ax.errorbar(x, y, xerr=np.nan_to_num(xe), yerr=ye, fmt=mk, ms=5,
                        mfc="none" if key == "sub" else None, capsize=2, lw=0.8,
                        label=f"{lab}: r = {pearson(x, y):+.2f}, partial r = {partial_r(x, y, ha):+.2f}")
            summary.append(dict(job=job, sample=key, n=len(j), r=pearson(x, y),
                                partial_r=partial_r(x, y, ha)))
        for r_, xx, yy in zip(j, x, [r["mean_full"] for r in j]):
            ax.annotate(str(r_["lig"]), (xx, yy), fontsize=7, xytext=(3, 3),
                        textcoords="offset points", color="0.35")
        ax.set_xlabel("MD bound fraction"); ax.set_ylabel("mean docking score (kcal/mol)")
        ax.set_title(job, fontsize=10); ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    if outdir:
        fig.savefig(f"{outdir}/md_bf_vs_score_full_and_{N_SUB}.png", dpi=150)
    return fig, summary


def plot_propensity(job, lig, prof, outdir=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), sharex=True)
    xs = np.arange(len(RES))
    for ax, t in zip(axes, TYPES):
        pf, ef, ps, es = prof[t]
        ef, es = np.nan_to_num(ef), np.nan_to_num(es)
        # lines for the profiles, shaded bands for +/- 2 SE; overlapping
        # bands mean the two estimates agree within error
        ax.fill_between(xs, np.clip(pf - 2 * ef, 0, None), pf + 2 * ef,
                        color="0.55", alpha=0.30, lw=0)
        ax.plot(xs, pf, "-o", color="0.25", ms=3, lw=1.3, label="full")
        ax.fill_between(xs, np.clip(ps - 2 * es, 0, None), ps + 2 * es,
                        color="tab:blue", alpha=0.25, lw=0)
        ax.plot(xs, ps, "--s", color="tab:blue", ms=3, lw=1.3,
                label=f"{N_SUB} frames")
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", color="0.92", lw=0.6)
        ax.set_title(t, fontsize=10)
        ax.set_xticks(xs); ax.set_xticklabels(RES, rotation=90, fontsize=7)
        if not (np.nanmax(pf) > 0 or np.nanmax(ps) > 0):
            # an empty panel is a property of the ligand (e.g. no H-bond donor
            # or no cationic group), not a missing plot -- say so
            ax.set_ylim(0, 1); ax.set_yticks([])
            ax.text(0.5, 0.5, "no contacts in\neither ensemble", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color="0.4")
    axes[0].set_ylabel("propensity (band = 2 SE)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{job}  ligand {lig}", fontsize=11)
    fig.tight_layout()
    if outdir:
        d = f"{outdir}/propensity/{job}"
        os.makedirs(d, exist_ok=True)
        fig.savefig(f"{d}/lig{lig}.png", dpi=130)
    return fig


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------- api ---
def run(jobs=None, ligs=None, contacts=True, outdir=None, verbose=True):
    """Score and propensity tests for every built ligand.
    Returns (score_rows, propensity_rows, profiles) where profiles maps
    (job, lig) -> {type: (p_full, se_full, p_sub, se_sub)}."""
    jobs = jobs or [j for j in WINNERS if os.path.isdir(f"{TRAJ}/{j}")]
    srows, prows, profs = [], [], {}
    for job in jobs:
        jl = ligs or ligands(job)
        if verbose:
            print(f"=== {job}: {len(jl)} ligands")
        for lig in jl:
            s_ = score_test(job, lig)
            if s_:
                srows.append(s_)
            line = (f"  lig{lig:<3d} score {s_['mean_full']:.3f}+/-{s_['se_full']:.3f} "
                    f"vs {s_['mean_sub']:.3f}+/-{s_['se_sub']:.3f} "
                    f"{'ok' if s_['agree'] else 'DIFF'}" if s_ else f"  lig{lig}: no scores")
            if contacts:
                try:
                    rows, prof = propensity_test(job, lig, contacts_cached(job, lig))
                    prows.extend(rows)
                    profs[(job, lig)] = prof
                    fr = [r["frac_within_2se"] for r in rows if np.isfinite(r["frac_within_2se"])]
                    line += f" | propensities within 2SE: {np.mean(fr):.0%}" if fr else ""
                except Exception as e:
                    line += f" | contacts FAILED: {type(e).__name__}: {e}"
            if verbose:
                print(line, flush=True)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        write_csv(f"{outdir}/scores_stride.csv", srows)
        write_csv(f"{outdir}/propensity_stride.csv", prows)
    return srows, prows, profs


def verdict(srows, prows):
    out = []
    if srows:
        n_ok = sum(r["agree"] for r in srows)
        out.append(f"mean scores: {n_ok}/{len(srows)} ligands, {N_SUB}-frame value "
                   f"within 2 SE of the full value")
        bad = [f"{r['job']}:lig{r['lig']}" for r in srows if not r["agree"]]
        if bad:
            out.append("   outside: " + ", ".join(bad))
    for t in TYPES:
        sel = [r for r in prows if r["type"] == t]
        v = [r["frac_within_2se"] for r in sel if np.isfinite(r["frac_within_2se"])]
        rr = [r["profile_r"] for r in sel if np.isfinite(r["profile_r"])]
        pk = [r["peak_ok"] for r in sel if r["peak_full"] != "-"]
        if v:
            out.append(f"{t:7s}: residues within 2 SE {np.mean(v):.1%} (worst ligand {min(v):.0%}), "
                       f"profile r median {np.median(rr):.3f}, peak residue kept (or tied) "
                       f"{sum(pk)}/{len(pk)}")
    if prows:
        out.append("(~95% within 2 SE is what sampling noise alone gives)")
    return out


# --------------------------------------------------------------------- main ---
def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--ligs", nargs="*", type=int, default=None)
    ap.add_argument("--bf-csv", default=None)
    ap.add_argument("--no-contacts", action="store_true",
                    help="scores and MD correlation only")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()

    srows, prows, profs = run(args.jobs, args.ligs, not args.no_contacts, args.outdir)
    for (job, lig), prof in profs.items():
        plt.close(plot_propensity(job, lig, prof, args.outdir))
    if srows:
        plt.close(plot_scores(srows, args.outdir))
        print(f"\nscores figure: {args.outdir}/scores_full_vs_{N_SUB}.png")
    bf = load_bf(args.bf_csv)
    if bf:
        fig, summ = plot_md_vs_score(srows, bf, args.outdir)
        if fig is not None:
            plt.close(fig)
            for s_ in summ:
                print(f"  {s_['job']:16s} {s_['sample']:4s} n={s_['n']:<3d} "
                      f"r={s_['r']:+.3f}  partial r (heavy atoms)={s_['partial_r']:+.3f}")
            write_csv(f"{args.outdir}/md_vs_score.csv", summ)
    else:
        print("\nno MD bound fractions found -- pass --bf-csv lig,bf[,bf_se]")
    print("\n================ verdict ================")
    print("\n".join(verdict(srows, prows)))


if __name__ == "__main__":
    main()
