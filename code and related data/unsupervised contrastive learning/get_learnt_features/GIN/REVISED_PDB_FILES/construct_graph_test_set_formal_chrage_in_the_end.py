import os
import csv
import math
import time
import random
# import networkx as nx
import numpy as np
from copy import deepcopy
from rdkit.Chem import Draw

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

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
# no coordination bond here

BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]



    
def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)
def set_dative_bonds(mol, fromAtoms=(6,7,8)):
    """ convert some bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
    with dative bonds. The replacement is only done if the atom has "too many" bonds.

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
    

your_need_process_type = [".pdb"]
Filelist = []
PDBlist = []
PDB_path = os.getcwd()
x_list = []
edge_index_list = [] 
edge_attr_list = []
for home, dirs, files in os.walk(PDB_path):
    for filename in files:
        Filelist.append(os.path.join(home, filename))

for file in  Filelist :
    filetype = os.path.splitext(file)[1]
    if filetype in your_need_process_type:
        PDBlist.append(file)
file = open("output.txt", "w")
for i in range(len(PDBlist)):
    print(PDBlist[i], file=file)
    print(PDBlist[i])

    mol = Chem.MolFromPDBFile(PDBlist[i])
    mol = set_dative_bonds(mol)
    # img0 = Draw.MolToImage(mol)
    # img0.save('{}_before_set_formal_charge.png'.format(PDBlist[i]), size=(600,600))

    mol = set_formal_charge(mol)
    # img = Draw.MolToImage(mol)
    # img.save('{}.png'.format(PDBlist[i]), size=(600,600))
    # mol = Chem.AddHs(mol)

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
        ])#具体来说，对于每个化学键，第一个append操作添加的是该化学键的正向信息，即化学键类型和化学键方向。而第二个append操作添加的是该化学键的反向信息，即化学键类型和相反的化学键方向。这样做的目的是为了保证在构建图时，图中的所有化学键都能够被正确地表示出来，不受化学键的方向影响。

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    x_list.append(x)
    edge_index_list.append(edge_attr) 
    edge_attr_list.append(edge_attr)
    print("x:",x, "\nedge index:",edge_index, "\nedge_attr:",edge_attr, file=file)
    print("x_shape:",x.size(), "\nedge index_shape:",edge_index.size(), "\nedge_attr_shape:",edge_attr.size(),file=file)
    # print(edge_index.shape[1])
    # print(x.size()[0])
    # print(edge_index[0][-5].item())
    if int(x.size()[0]) - int(torch.max(edge_index[0])) == 1:
        pass
        # print("Successfully recognize all the heavy atoms")
    else:
        print("WRONGLY recognize all the heavy atoms, BE CAUTIOUS")
    n = 0
    for k in range(int(int(edge_index.shape[1])/2)):
        # print("edge index:",edge_index[0][(k*2)],edge_index[0][(k*2+1)], "edge_attr:",edge_attr[int(2*k)], file=file)

        # print((edge_attr[int(2*k)]==torch.tensor([0,0])).all())
        # print((edge_attr[int(2*k)]==torch.tensor([4,0])).all())

        if (edge_attr[int(2*k)]==torch.tensor([0,0])).all():
            # print("[0,0]")
            print("edge index:",edge_index[0][(k*2)],edge_index[0][(2*k+1)], "edge_attr:",edge_attr[int(2*k)], file=file)
        if (edge_attr[int(2*k)]==torch.tensor([4,0])).all():
            # print("[4,0]")
            print("edge index:",edge_index[0][(k*2)],edge_index[0][(k*2+1)], "edge_attr:",edge_attr[int(2*k)], file=file)
            n = n + 1
            
        if (edge_attr[int(2*k)]==torch.tensor([1,0])).all() or (edge_attr[int(2*k)]==torch.tensor([2,0])).all():
            print("!!!!Super cautious on this bond","edge index:",edge_index[0][(k*2)],edge_index[0][(k*2+1)], "edge_attr:",edge_attr[int(2*k)], file=file)
            print("!!!!Super cautious on this bond","edge index:",edge_index[0][(k*2)],edge_index[0][(k*2+1)], "edge_attr:",edge_attr[int(2*k)])

    if n == 2:
        pass
    else:
        print("Cautious, the number of coordination bonds is NOT 2, need to check")
file.close()





