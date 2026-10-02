from rdkit import Chem
from rdkit.Chem import AllChem



ligand_smiles = []
with open('all_asyn_smiles.smi', 'r') as mf:
    for smi in mf:
        print(smi)
        ligand_smiles.append(smi.strip('\n'))


for i, smiles in enumerate(ligand_smiles):
    ligand_mol = Chem.MolFromSmiles(smiles)
    ligand_mol = Chem.AddHs(ligand_mol)
    AllChem.EmbedMolecule(ligand_mol)
    filename = str(i) + '.sdf'
    writer = Chem.SDWriter(filename)
    writer.write(ligand_mol)
    writer.close()
