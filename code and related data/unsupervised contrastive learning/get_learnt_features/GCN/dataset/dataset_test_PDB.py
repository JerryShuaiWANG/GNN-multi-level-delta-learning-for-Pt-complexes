import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.DATIVE]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(PDB, include_chirality=False):
    mol = Chem.MolFromPDBFile(PDB)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = dataset.len()
    print(data_len)

    print("About to generate scaffolds")
    for ind, PDB in enumerate(dataset.PDB_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(PDB)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * int(dataset.len())
    valid_cutoff = (train_size + valid_size) * int(dataset.len())
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds



def get_pdb_name(data_path, target, task):
    PDB_data, labels  = [], []

    parent_directory = os.path.abspath(".")
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        print(csv_reader)
        for i, row in enumerate(csv_reader):
            if i != -1:#if the first two rows are name, this value need to be 0

                print(row)
                PDB = str(row["\ufeffStructure"]).lstrip() + ".pdb"
                label = row[target]

                mol = Chem.MolFromPDBFile(os.path.join(parent_directory,"REVISED_PDB_FILES", PDB))

                if mol != None and label != '':

                    PDB_data.append(os.path.join(parent_directory,"REVISED_PDB_FILES", PDB))

                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print("num of available PDB files:",len(PDB_data))
    return PDB_data, labels
def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

def set_dative_bonds(mol, fromAtoms=(6,7,8)):
    """ convert some bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
    with dative bonds.

    Returns the modified molecule.

    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]

    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in (7,):
                nbr.SetFormalCharge(0)

            if nbr.GetAtomicNum() in fromAtoms and \
                nbr.GetExplicitValence()!=pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                    rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)

            if nbr.GetAtomicNum() in (6,) and \
                len(nbr.GetNeighbors())==2:
                for idx in range(len(nbr.GetNeighbors())):
                    if nbr.GetNeighbors()[idx].GetAtomicNum() not in [n for n in range (72,80)] and \
                        rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE and \
                            rwmol.GetBondBetweenAtoms(nbr.GetIdx(),nbr.GetNeighbors()[idx].GetIdx()).GetBondType() == Chem.BondType.TRIPLE:
                        rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                        rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)                
    return rwmol

def set_formal_charge(mol):
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    for atom in rwmol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    return rwmol


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.PDB_data, self.labels = get_pdb_name(data_path, target, task)
        self.task = task
        self.conversion = 1
        global index_list 
        index_list = []
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        mol = Chem.MolFromPDBFile(self.PDB_data[index])
        mol = set_dative_bonds(mol)

        mol = set_formal_charge(mol)        


        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            print(self.labels)
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
            index_list.append(index)
        with open("index_list.csv", 'w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(index_list)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def len(self):
        return len(self.PDB_data)
    def indices(self):
        return range(self.len()) if self._indices is None else self._indices

    def set_indices(self, indices):
        self._indices = indices

class MolTestDatasetWrapper(object):
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting 
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting

        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):

        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)

        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':

            num_train = train_dataset.len()
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
