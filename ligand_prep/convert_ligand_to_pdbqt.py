import os
import subprocess
import pandas as pd
from rdkit import Chem

def batch_convert_ligands_final(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sdf_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.sdf')])
    center_data = []

    print(f"converting {len(sdf_files)} ligands (Open Babel CLI)...")

    for sdf in sdf_files:
        input_path = os.path.join(input_dir, sdf)
        output_path = os.path.join(output_dir, sdf.replace('.sdf', '.pdbqt'))

        try:
            mol = Chem.MolFromMolFile(input_path, removeHs=False, sanitize=True)
            if mol is not None:
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                center = pos.mean(axis=0)
                center_data.append([sdf, center[0], center[1], center[2]])
            else:
                print(f"RDKit could not read {sdf}")
                continue

            cmd = [
                "obabel", 
                "-isdf", input_path, 
                "-opdbqt", "-O", output_path,
                "-h",
                "--partialcharge", "gasteiger"
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"ok: {sdf} (hydrogens added, charges computed)")

        except Exception as e:
            print(f"error processing {sdf}: {e}")

    df = pd.DataFrame(center_data, columns=['file', 'center_x', 'center_y', 'center_z'])
    df.to_csv(os.path.join(output_dir, 'ligand_centers.csv'), index=False)
    print(f"\ncentres written to: {output_dir}/ligand_centers.csv")

if __name__ == "__main__":
    LIGAND_SDF_DIR = "./asyn_ligands" 
    LIGAND_PDBQT_DIR = "./ligands_pdbqt"
    batch_convert_ligands_final(LIGAND_SDF_DIR, LIGAND_PDBQT_DIR)


"""
if __name__ == "__main__":
    # --- adjust to the actual paths on prl-desktop1 ---
    LIGAND_SDF_DIR = "./asyn_ligands" 
    LIGAND_PDBQT_DIR = "./asyn_ligands_pdbqt"
    
    batch_convert_ligands_with_hs(LIGAND_SDF_DIR, LIGAND_PDBQT_DIR)
"""
