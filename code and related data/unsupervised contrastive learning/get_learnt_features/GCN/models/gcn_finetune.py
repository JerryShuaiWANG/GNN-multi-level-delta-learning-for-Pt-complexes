import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes
import sys
import pandas as pd
import csv

 


num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 # including aromatic, self-loop edge and dative bond
num_bond_direction = 3 


def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def read_csv_to_list(file_path):
    result = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            result.append(row)
    return result



class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 5 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        edge_index, __ = gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


class GCN(nn.Module):
    def __init__(self, task='classification',  num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task
 
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        # self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim) 

        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus(),
                nn.Linear(self.feat_dim//2, 2)
            )
        elif self.task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.Softplus(),
                nn.Linear(self.feat_dim//2, 1)
            )

    def forward(self, data):

        x = data.x 
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        print("h",h)
        print("h.size",h.size())
        h = self.feat_lin(h)
        print("train/valid/test h (after feat_lin)",h)
        print("train/valid/test h size (after feat_lin)",h.size())
        print("save csv here")
        h_with_exp = torch.concat((h, data.y), 1)
        h_with_exp_cpu_tensor = h_with_exp.cpu()
        df = pd.DataFrame(h_with_exp_cpu_tensor.detach().numpy())
        df.to_csv('./GCN_with_Calc_kr_{}_{}_linear_out.csv.csv'.format(h.size()[0],h.size()[1]), header=False, index=False)
        

        index_list = read_csv_to_list("index_list.csv")
        print("index_list",index_list)        
        whole_original_df_path = './data/pdb/all_QM_features_and_Calc_kr_clean_FINAL_for_Calc_kr.csv'#for training and testing sets
        # whole_original_df_path = './data/pdb/external_test_with_structure_name_and_QM_used.csv'#for external testing set

        whole_original_df = pd.read_csv(whole_original_df_path)
        print("whole_original_df", whole_original_df)
        df_QM_fea= whole_original_df.iloc[:,1:45]
        print("df_QM_fea",df_QM_fea)
        df_solvent= whole_original_df.iloc[:,47:50]
        df_QM_and_solvet= pd.concat([df_QM_fea, df_solvent], axis=1, ignore_index=True)
        print("df_QM_and_solvet",df_QM_and_solvet)
        df_reindex =  df_QM_and_solvet.reindex(int(i) for i in index_list[0]) # cautiouos that the element in index_list[0] is str
        print("df_reindex",df_reindex)
        df_reindex_reset = df_reindex.reset_index()
        print(df_reindex_reset)
        df_final_with_QM= pd.concat([df, df_reindex_reset], axis=1, ignore_index=True)
        print("df_final_with_QM",df_final_with_QM)
        df_final_with_QM.to_csv('./emb+QM_GCN_with_Calc_kr_{}_{}_linear_out_with_QM.csv.csv'.format(h.size()[0],h.size()[1]), header=False, index=False)


        return h, self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


if __name__ == "__main__":
    model = GCN()
    print(model)
