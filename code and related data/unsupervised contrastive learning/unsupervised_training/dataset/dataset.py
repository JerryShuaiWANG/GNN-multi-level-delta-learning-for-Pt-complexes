import os
import csv
import glob
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
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


class MoleculeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(Dataset, self).__init__()
        self.PDB_name = read_PDB_name(data_path)
        self._indices = None
        self.transform = transform
    def get(self, index):
        try:
            mol = Chem.MolFromPDBFile(self.PDB_name[index])
            N = mol.GetNumAtoms()
            M = mol.GetNumBonds()

            type_idx = []
            chirality_idx = []
            atomic_number = []
            # aromatic = []
            # sp, sp2, sp3, sp3d = [], [], [], []
            # num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                atomic_number.append(atom.GetAtomicNum())
                # aromatic.append(1 if atom.GetIsAromatic() else 0)
                # hybridization = atom.GetHybridization()
                # sp.append(1 if hybridization == HybridizationType.SP else 0)
                # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

            # z = torch.tensor(atomic_number, dtype=torch.long)
            x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
            x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
            x = torch.cat([x1, x2], dim=-1)
            # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
            #                     dtype=torch.float).t().contiguous()
            # x = torch.cat([x1.to(torch.float), x2], dim=-1)

            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
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

            # random mask a subgraph of the molecule
            num_mask_nodes = max([1, math.floor(0.25*N)])
            num_mask_edges = max([0, math.floor(0.25*M)])
            mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
            mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
            mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
            mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
            mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
            mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

            x_i = deepcopy(x)
            for atom_idx in mask_nodes_i:
                x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
            edge_index_i = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
            edge_attr_i = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
            count = 0
            for bond_idx in range(2*M):
                if bond_idx not in mask_edges_i:
                    edge_index_i[:,count] = edge_index[:,bond_idx]
                    edge_attr_i[count,:] = edge_attr[bond_idx,:]
                    count += 1
            data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

            x_j = deepcopy(x)
            for atom_idx in mask_nodes_j:
                x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
            edge_index_j = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
            edge_attr_j = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
            count = 0
            for bond_idx in range(2*M):
                if bond_idx not in mask_edges_j:
                    edge_index_j[:,count] = edge_index[:,bond_idx]
                    edge_attr_j[count,:] = edge_attr[bond_idx,:]
                    count += 1
            data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
            
            return data_i, data_j

        except Exception as e:
            logging.info("********************wronginfo********************")
            logging.info(self.PDB_name[index],e)


    def len(self):
        return len(self.PDB_name)

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
        # print("initial train_dataset", train_dataset)
        # print("initial train_dataset type", type(train_dataset))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = train_dataset.len()
        # print("num_train",num_train)
        indices = list(range(num_train))
        # print("indices",indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        # print("num of train data",len(train_idx))
        # print("num of valid data",len(valid_idx))
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
