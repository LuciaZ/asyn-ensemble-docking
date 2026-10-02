import os
import subprocess

def batch_convert_to_pdbqt(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdb_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pdb')])
    
    if not pdb_files:
        print(f"no PDB files found in {input_dir}")
        return

    print(f"found {len(pdb_files)} files, converting with Open Babel...")

    for pdb in pdb_files:
        input_path = os.path.join(input_dir, pdb)
        output_path = os.path.join(output_dir, pdb.replace('.pdb', '.pdbqt'))

        cmd = [
            "obabel", 
            "-ipdb", input_path, 
            "-opdbqt", "-O", output_path,
            "-p", "7.4",
            "-xr"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"ok: {pdb} -> {os.path.basename(output_path)}")
        except subprocess.CalledProcessError as e:
            print(f"error: processing {pdb} failed. reason: {e.stderr}")
        except FileNotFoundError:
            print("error: 'obabel' not found. Install it with 'conda install -c conda-forge openbabel' or 'sudo apt install openbabel'")
            return







if __name__ == "__main__":
    PROTEIN_FOLDER = "./protein_frames_apo"
    RESULT_FOLDER = "./protein_apo_pdbqt"

    batch_convert_to_pdbqt(PROTEIN_FOLDER, RESULT_FOLDER)
