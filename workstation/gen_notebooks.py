import json, os, textwrap

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n").splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": textwrap.dedent(s).strip("\n").splitlines(keepends=True)}

IMPORTS = """
import os, sys, json
import numpy as np
import mdtraj as md

for p in ("~/data/scripts", "~/data/scripts/docking_analysis"):
    p = os.path.expanduser(p)
    if os.path.isdir(os.path.join(p, "docking_tools")) and p not in sys.path:
        sys.path.insert(0, p)
from docking_tools import trajectory_analysis
from docking_tools import plotting
"""

HELPERS = """
def load_docked(lig):
    \"\"\"Docked trajectory + ligand atom groups written by build_docked_trajectories.py.\"\"\"
    top = md.load_topology(os.path.join(DATA, f"lig{lig}_top.pdb"))
    traj = md.Trajectory(np.load(os.path.join(DATA, f"lig{lig}_xyz.npy")), top)
    with open(os.path.join(DATA, f"lig{lig}_groups.json")) as fh:
        groups = json.load(fh)
    return traj, groups


def analyse(lig):
    traj, g = load_docked(lig)
    n_frames = traj.n_frames
    # number of residues in the protein; the ligand is the last residue (0 indexed)
    n_res = traj.atom_slice(traj.top.select('protein')).n_residues
    ligand_idx = n_res
    shape = (N_CLUS, n_frames, n_res)
    empty = lambda: np.zeros(shape)

    contacts = {}
    contacts['hphob'] = trajectory_analysis.hphob_contacts(traj, ligand_idx).reshape(shape)
    contacts['aro'] = (trajectory_analysis.aro_contacts(traj, ligand_rings=g['aromatic_rings']).reshape(shape)
                       if g['aromatic_rings'] else empty())
    contacts['hbond'] = (trajectory_analysis.hbond(traj, ligand_idx, lig_hbond_donors=g['hbond_donors']).reshape(shape)
                         if g['hbond_donors'] else empty())
    contacts['charge'] = (trajectory_analysis.charge_contacts(traj, Ligand_Pos_Charges=g['positive_charges']).reshape(shape)
                          if g['positive_charges'] else empty())
    contacts['dual'] = np.array([trajectory_analysis.dual_contact(traj, residue_idx=ligand_idx)])
    print(f"ligand {lig}: {n_frames} frames, {n_res} residues")
    return contacts
"""

GROUPS = """
# detected ligand atom groups -- check these against each ligand's chemistry
# before reading the aromatic, H-bond and charge panels
for lig in LIGANDS:
    try:
        with open(os.path.join(DATA, f"lig{lig}_groups.json")) as fh:
            g = json.load(fh)
    except FileNotFoundError:
        print(f"ligand {lig:>3}: not built yet"); continue
    print(f"ligand {lig:>3}: {g['n_frames']:>6} frames | rings {len(g['aromatic_rings'])} "
          f"| donors {len(g['hbond_donors'])} | cation {g['positive_charges']} "
          f"| anion {g['negative_charges']}")
"""

CHECK = """
# The same fingerprints were computed earlier by the batch pipeline; per-residue
# profiles from this notebook should agree with them closely.
for lig in LIGANDS:
    p = os.path.join(PIPELINE, f"lig{lig}_fingerprint.npz")
    if lig not in contacts or not os.path.exists(p):
        continue
    old = np.load(p, allow_pickle=True)
    row = []
    for key in ('hphob', 'aro', 'hbond', 'charge'):
        if key not in old.files:
            row.append(f"{key} -"); continue
        a = np.asarray(old[key], float)
        a = a.mean(axis=0) if a.ndim == 2 else a
        b = contacts[lig][key][0].mean(axis=0)
        n = min(a.size, b.size)
        r = np.corrcoef(a[:n], b[:n])[0, 1] if a[:n].std() > 0 and b[:n].std() > 0 else float('nan')
        row.append(f"{key} r={r:.3f}")
    print(f"ligand {lig:>3}: " + "  ".join(row))
"""

def notebook(title, intro, job, ligands, pipeline=None, extra_note=None):
    cells = [md(f"# {title}\n\n{intro}")]
    if extra_note:
        cells.append(md(extra_note))
    cells.append(md("""
    Each frame is the receptor conformation that was docked against, plus the best-of-20
    pose (the lowest-scoring of the 20 per-residue boxes). The example notebook reweights
    t-SNE clusters by their MD populations; these ensembles are instead stride-sampled from
    equilibrium trajectories, so every frame carries equal weight and the ensemble is treated
    as a single cluster with weight 1.

    Build the input once with
    `python build_docked_trajectories.py --job %s`.
    """ % job))
    cells.append(code(IMPORTS))
    cfg = f"""
    JOB = "{job}"
    DATA = os.path.expanduser(f"~/data/docked_trajectories/{{JOB}}")
    LIGANDS = {ligands}
    """
    if pipeline:
        cfg += f'PIPELINE = os.path.expanduser("{pipeline}")\n'
    cfg += """
    # one cluster with weight 1 (uniform, stride-sampled ensemble)
    N_CLUS = 1
    weights = np.array([1.0])
    contacts = {}
    """
    cells.append(code(cfg))
    cells.append(code(HELPERS))
    cells.append(md("## Ligand atom groups"))
    cells.append(code(GROUPS))
    for lig in ligands:
        cells.append(md(f"## Ligand {lig}\n\nPopulations of intermolecular interactions and "
                        f"ligand dual residue contact probability, ensemble average."))
        cells.append(code(f"""
        contacts[{lig}] = analyse({lig})
        # intermolecular interactions
        plotting.plot_ensemble_ii(contacts[{lig}], weights)
        """))
        cells.append(code(f"""
        # ligand dual residue contact probability
        plotting.plot_ensemble_dual(contacts[{lig}]['dual'], weights)
        """))
    if pipeline:
        cells.append(md("## Consistency check against the batch-pipeline fingerprints"))
        cells.append(code(CHECK))
    return {"cells": cells, "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}

PILOT = [38, 5, 28, 34, 3, 24, 17, 53, 9, 40]
REMAINING = [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
             25, 26, 27, 29, 30, 33, 35, 36, 37, 39, 46, 48, 49, 50, 51, 52, 54, 55, 56]

books = {
 "apo_pilot10.ipynb": notebook(
    "Pilot ligands docked onto the apo ensemble of a-syn-C-term with AutoDock Vina-GPU",
    "Populations of intermolecular interactions and dual residue contact probability for the "
    "10 pilot ligands (38, 5, 28, 34, 3, 24, 17, 53, 9, 40), docked with AutoDock Vina-GPU 2.1 "
    "onto 5,005 frames (every 4th of 20,020) of the apo MD simulation of a-syn-C-term "
    "(D121-A140) -- apo docking. 20 boxes per frame, one per residue centroid, uniform "
    "18.75 A edge, num_modes = 20.",
    "apo_pilot10", PILOT,
    pipeline="~/data/VinaGPU/hpc_results/08142026/analysis"),
 "holo_pilot10.ipynb": notebook(
    "Pilot ligands docked onto their holo ensembles of a-syn-C-term with AutoDock Vina-GPU",
    "Populations of intermolecular interactions and dual residue contact probability for the "
    "same 10 pilot ligands, each docked onto 5,000 bound frames of its own MD simulation with "
    "a-syn-C-term (minimum protein-ligand heavy-atom distance < 6 A) -- holo docking. Box "
    "geometry as in the apo job.",
    "holo_pilot10", PILOT,
    pipeline="~/data/VinaGPU/hpc_results/holo_08192026/analysis_v2",
    extra_note="Receptors are the all-atom PDB frames. The PDBQT receptors used for docking "
               "have all but ~29 polar hydrogens stripped, which under-counts H-bonds about "
               "threefold, so they are not used here."),
 "trio_47_41_23_apo.ipynb": notebook(
    "Ligands 47, 41 (fasudil) and 23 docked onto the apo ensemble of a-syn-C-term",
    "Populations of intermolecular interactions and dual residue contact probability for the "
    "three published ligands, docked with AutoDock Vina-GPU onto all 20,020 frames of the "
    "apo MD simulation. The best box per frame is taken from the extracted score table; "
    "frames where fewer than 15 boxes completed are excluded.",
    "trio_apo", [47, 41, 23],
    extra_note="These runs used Rg-scaled boxes (47: 16.58 A, 41: 13.01 A, 23: 13.17 A) "
               "rather than the uniform 18.75 A box of the other jobs, so their interaction "
               "profiles are comparable in shape but the protocols are not identical."),
 "apo_remaining37.ipynb": notebook(
    "Remaining 37 ligands docked onto the apo ensemble of a-syn-C-term with AutoDock Vina-GPU",
    "Populations of intermolecular interactions and dual residue contact probability for the "
    "37 ligands outside the pilot, docked with the same protocol as the apo pilot "
    "(5,005 apo frames, 20 boxes per frame, uniform 18.75 A edge).",
    "apo_remaining37", REMAINING,
    extra_note="Requires the collapsed winners for this job: run the best-of-20 collapse on "
               "the cluster once docking completes, copy `outputs_vinagpu/winners/` to "
               "`~/data/VinaGPU/hpc_results/apo_remaining/winners/`, then build."),
}
for name, nb in books.items():
    out = os.path.expanduser(f"~/asyn-ensemble-docking/notebooks/{name}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"{name}: {len(nb['cells'])} cells")
