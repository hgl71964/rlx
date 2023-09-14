import time
import datetime
import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.models.basic_gnn import GAT


##############################
############ Utils ###########
##############################
class CategoricalMasked(Categorical):
    def __init__(self,
                 probs=None,
                 logits=None,
                 validate_config=None,
                 mask=None,
                 device=torch.device("cpu")):
        self.mask = mask
        self.device = device
        if self.mask is None:
            super(CategoricalMasked, self).__init__(probs, logits,
                                                    validate_config)
        else:
            self.mask = mask.type(torch.BoolTensor).to(device)
            logits = torch.where(self.mask, logits,
                                 torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits,
                                                    validate_config)

    def entropy(self):
        if self.mask is None:
            return super(CategoricalMasked, self).entropy()

        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p,
                              torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


##############################
############ GNNs ############
##############################
class EdgeModel(torch.nn.Module):
    def __init__(self, hidden_size, out):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_size, out),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        return self.head(torch.cat([src, dest, edge_attr], dim=1))


class GATNetwork(nn.Module):
    """A Graph Attentional Network (GAT)
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 n_actions: int,
                 n_layers: int,
                 hidden_size: int,
                 num_head: int,
                 vgat: int,
                 out_std: float = np.sqrt(2),
                 dropout=0.0,
                 use_edge_attr=True,
                 add_self_loops=False,
                 edge_dim=None):
        super().__init__()
        self.n_actions = n_actions
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            assert (edge_dim is not None), "use edge attr but edge_dim is None"
            self.edge_layer = MetaLayer(
                EdgeModel(2 * num_node_features + num_edge_features,
                          num_edge_features), None, None)
            self.edge_layer2 = MetaLayer(
                EdgeModel(2 * n_actions + num_edge_features, n_actions), None,
                None)

        self.gnn = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       heads=num_head,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True if vgat == 2 else False,
                       edge_dim=(edge_dim if self.use_edge_attr else None))

        if dropout == 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, 2 * hidden_size),
                # nn.LeakyReLU(),
                nn.Tanh(),
                nn.Linear(2 * hidden_size, hidden_size))
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, 2 * hidden_size),
                # nn.LeakyReLU(),
                nn.Tanh(),
                nn.Linear(2 * hidden_size, hidden_size))

        # a final layer to transform node features
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, self.n_actions),
            nn.Tanh(),
            layer_init(nn.Linear(self.n_actions, self.n_actions), std=out_std),
        )

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        # 1. update edges if possible
        edge_index = data.edge_index
        if self.use_edge_attr:
            x, edge_attr, _ = self.edge_layer(data.x, data.edge_index,
                                              data.edge_attr, None, None)
        else:
            x = data.x
            edge_attr = None

        # 2. update nodes
        x = self.gnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # print("[GATNetwork] ", x.shape)
        x = self.head(x)
        # ff will weight nodes differently
        x = self.ff(x)
        # print("[GATNetwork] ", x.shape)

        # 3. update edge in the end
        if self.use_edge_attr:
            x, edge_attr, _ = self.edge_layer2(x, edge_index, edge_attr, None,
                                               None)
        return x, edge_attr


class GATNetwork_with_global(nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 n_actions: int,
                 n_layers: int,
                 hidden_size: int,
                 num_head: int,
                 vgat: int,
                 out_std: float = np.sqrt(2),
                 dropout=0.0,
                 use_edge_attr=True,
                 add_self_loops=False,
                 edge_dim=None):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            assert (edge_dim is not None), "use edge attr but edge_dim is None"
            self.edge_layer = MetaLayer(
                EdgeModel(2 * num_node_features + num_edge_features,
                          num_edge_features), None, None)

        self.gnn = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       heads=num_head,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True if vgat == 2 else False,
                       edge_dim=(edge_dim if self.use_edge_attr else None))

        if dropout == 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, 2 * hidden_size),
                # nn.LeakyReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(2 * hidden_size, n_actions), std=out_std))
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, 2 * hidden_size),
                # nn.LeakyReLU(),
                nn.Tanh(),
                layer_init(nn.Linear(2 * hidden_size, n_actions), std=out_std))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        # 1. update edges if possible
        edge_index = data.edge_index
        if self.use_edge_attr:
            x, edge_attr, _ = self.edge_layer(data.x, data.edge_index,
                                              data.edge_attr, None, None)
        else:
            x = data.x
            edge_attr = None

        # 2. update nodes
        x = self.gnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # print("[GNN global] ", x.shape)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        # print("[GNN global] ", x.shape)
        x = self.head(x)
        # print("[GNN global] ", x.shape)
        return x, edge_attr
