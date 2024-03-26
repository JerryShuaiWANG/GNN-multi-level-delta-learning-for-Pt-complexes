import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
import glob
import logging
import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
# note that this Dataset is different with the from torch.utils.data import Dataset
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='example.log')
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC,
    BT.DATIVE
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles_original(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
            # mol = Chem.MolFromSmiles(smiles)
            # if mol != None:
            #     smiles_data.append(smiles)
    return smiles_data

"""
Remove a connected subgraph from the original molecule graph. 
Args:
    1. Original graph (networkx graph)
    2. Index of the starting atom from which the removal begins (int)
    3. Percentage of the number of atoms to be removed from original graph

Outputs:
    1. Resulting graph after subgraph removal (networkx graph)
    2. Indices of the removed atoms (list)
"""

def read_PDB_name(data_path):
    PDB_true_list = [] 
    PDBlist = glob.glob(os.path.join(data_path, '*.pdb'))
    for i in range(len(PDBlist)):
        mol = Chem.MolFromPDBFile(PDBlist[i])
    # mol = Chem.AddHs(mol)
        if mol in [None,]:
            logging.info("Below structure is Not correctly read by rdkit")
            logging.info(PDBlist[i])
            pass
        else:
            PDB_true_list.append(PDBlist[i])
        # print(PDBlist)
    return PDB_true_list
def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)

def set_dative_bonds(mol, fromAtoms=(6,7,8)):
    """ convert some (single) bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
    with dative bonds. The replacement is only done if the atom is in abnormal valence.

    Returns the modified molecule.

    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    # metals = [Pt]
    # atom.SetNoImplicit(True)
    # m.GetAtomWithIdx(0).SetAtomicNum(7)
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in (7,):
                nbr.SetFormalCharge(0)
                # if nbr.GetNeighbors() in (1,):
                #     rwmol.RemoveAtom(nbr.GetNeighbors().GetIdx())    
                # nbr.SetNoImplicit(True)
                # m.GetAtomWithIdx(nbr.GetIdx()).SetAtomicNum(7)

            if nbr.GetAtomicNum() in fromAtoms and \
                nbr.GetExplicitValence()!=pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                    rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
                # nbr.GetExplicitValence() = pt.GetDefaultValence(nbr.GetAtomicNum())
    return rwmol

def set_formal_charge(mol):
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    for atom in rwmol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    return rwmol

def removeSubgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed


class MoleculeDataset(Dataset):
    def __init__(self, data_path,transform=None):

        super(Dataset, self).__init__()
        self.PDB_data = read_PDB_name(data_path)
        self._indices = None
        self.transform = transform

    def get(self, index):
        # mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)

        mol = Chem.MolFromPDBFile(self.PDB_data[index])
        mol = set_dative_bonds(mol)

        mol = set_formal_charge(mol)        


        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        # Sample 2 different centers to start for i and j
        start_i, start_j = random.sample(list(range(N)), 2)

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)
        
        # Get the graph for i and j after removing subgraphs
        # G_i, removed_i = removeSubgraph(molGraph, start_i)
        # G_j, removed_j = removeSubgraph(molGraph, start_j)

        # percent_i, percent_j = random.uniform(0, 0.25), random.uniform(0, 0.25)
        percent_i, percent_j = 0.25, 0.25
        # percent_i, percent_j = 0.2, 0.2
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)
        
        for atom in atoms:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        # x shape (N, 2) [type, chirality]

        # Mask the atoms in the removed list
        x_i = deepcopy(x)
        for atom_idx in removed_i:
            # Change atom type to 118, and chirality to 0
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        x_j = deepcopy(x)
        for atom_idx in removed_j:
            # Change atom type to 118, and chirality to 0
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])

        # Only consider bond still exist after removing subgraph
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)
            if (start, end) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)

        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)
        
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        return data_i, data_j

    def len(self):
        return len(self.PDB_data)
    def indices(self):
        return range(self.len()) if self._indices is None else self._indices

    def set_indices(self, indices):
        self._indices = indices


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size





    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = train_dataset.len()
        indices = list(range(num_train))
        
        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        print("num of train data",len(train_idx))
        print("num of valid data",len(valid_idx))
        # logging.info("num of train data",len(train_idx))
        logging.info("num of train data: {}".format(len(train_idx)))
        # logging.info("num of valid data",len(valid_idx))
        logging.info("num of valid data: {}".format(len(valid_idx)))


        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader

