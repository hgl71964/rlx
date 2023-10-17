import numpy as np
from typing import Union

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.utils import scatter
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
########## MetaLayer #########
##############################
# see: torch_geometric/nn/models/meta.py
class EdgeModel(torch.nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        return self.head(torch.cat([src, dest, edge_attr], dim=1))


class NodeModel(torch.nn.Module):

    def __init__(self, num_node_features, num_edge_features, final_out_size):
        super().__init__()
        in_size = num_node_features + num_edge_features
        hidden_size = in_size * 2
        out_size = hidden_size // 2
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size),
        )

        in_size = out_size + num_node_features
        hidden_size = in_size * 2
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, final_out_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        # out = torch.cat([x, out, u[batch]], dim=1)
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # print(u.shape)
        # print(x.shape)
        # print(scatter(x, batch, dim=0, reduce='mean').shape)
        out = torch.cat([
            u,
            scatter(x, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)


##############################
############ GNNs ############
##############################
class GATNetwork(nn.Module):
    """A Graph Attentional Network (GAT)
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
    """

    def __init__(
            self,
            num_node_features: int,
            num_edge_features: int,
            n_actions: int,
            n_layers: int,
            hidden_size: int,
            num_head: int,
            vgat: int,
            out_std: float = np.sqrt(2),
            dropout=0.0,
            add_self_loops=False,
    ):
        super().__init__()

        in_size = 2 * num_node_features + num_edge_features
        self.edge_and_node_layer = MetaLayer(
            EdgeModel(in_size, 2 * in_size, num_edge_features),
            NodeModel(num_node_features, num_edge_features, num_node_features),
            None,
        )
        in_size = 2 * hidden_size + num_edge_features
        self.edge_and_node_layer_2 = MetaLayer(
            EdgeModel(in_size, 2 * in_size, num_edge_features),
            NodeModel(hidden_size, num_edge_features, hidden_size),
            None,
        )

        self.gat = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       heads=num_head,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True if vgat == 2 else False,
                       edge_dim=(num_edge_features))

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
            nn.Linear(hidden_size, n_actions),
            nn.Tanh(),
            layer_init(nn.Linear(n_actions, n_actions), std=out_std),
        )

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        # 1. update edges and nodes
        edge_index = data.edge_index
        x, edge_attr, _ = self.edge_and_node_layer(
            data.x,
            edge_index,
            data.edge_attr,
            None,
            None,
        )

        # 2. update nodes
        x = self.gat(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # 3. update edges and nodes
        x, edge_attr, _ = self.edge_and_node_layer_2(
            x,
            edge_index,
            edge_attr,
            None,
            None,
        )
        # 4. final layer
        # print("[GATNetwork] ", x.shape)
        x = self.head(x)
        # ff will weight nodes differently
        x = self.ff(x)
        # print("[GATNetwork] ", x.shape)
        return x, edge_attr


class GATNetwork_with_global(nn.Module):

    def __init__(
            self,
            num_node_features: int,
            num_edge_features: int,
            n_actions: int,
            n_layers: int,
            hidden_size: int,
            num_head: int,
            vgat: int,
            out_std: float = np.sqrt(2),
            dropout=0.0,
            add_self_loops=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.out_std = out_std

        in_size = 2 * num_node_features + num_edge_features
        self.edge_and_node_layer = MetaLayer(
            EdgeModel(in_size, 2 * in_size, num_edge_features),
            NodeModel(num_node_features, num_edge_features, num_node_features),
            None,
        )
        in_size = 2 * hidden_size + num_edge_features
        self.edge_and_node_layer_2 = MetaLayer(
            EdgeModel(in_size, 2 * in_size, num_edge_features),
            NodeModel(hidden_size, num_edge_features, hidden_size),
            None,
        )

        self.gat = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       heads=num_head,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True if vgat == 2 else False,
                       edge_dim=(num_edge_features))

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

    def fine_tuning(self):
        # TODO freeze prev layers?
        # for param in self.parameters():
        #     param.requires_grad = False

        # re-init last layer
        self.head[-1] = layer_init(nn.Linear(2 * self.hidden_size,
                                             self.n_actions),
                                   std=self.out_std)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        # 1. update edges and nodes
        edge_index = data.edge_index
        x, edge_attr, _ = self.edge_and_node_layer(
            data.x,
            edge_index,
            data.edge_attr,
            None,
            None,
        )

        # 2. update nodes
        x = self.gat(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # 3. pooling
        # print("[GNN global] ", x.shape)
        x, edge_attr, _ = self.edge_and_node_layer_2(
            x,
            edge_index,
            edge_attr,
            None,
            None,
        )
        # x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = pyg.nn.global_mean_pool(x=x, batch=data.batch)
        # print("[GNN global] ", x.shape)

        # 4. fial output
        x = self.head(x)
        # print("[GNN global] ", x.shape)
        return x, edge_attr


class PoolingLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        out_size,
        out_std,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_std = out_std
        self.out_size = out_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(2 * hidden_size, out_size), std=out_std),
        )

    def forward(self, x, batch):
        out = pyg.nn.global_mean_pool(x=x, batch=batch)
        return self.head(out)

    def fine_tuning(self):
        # TODO freeze prev layers?
        # for param in self.parameters():
        #     param.requires_grad = False
        self.head[-1] = layer_init(nn.Linear(2 * self.hidden_size,
                                             self.out_size),
                                   std=self.out_std)


class GlobalLayer(nn.Module):

    def __init__(
        self,
        hidden_size,
        out_size,
        out_std,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_std = out_std
        self.out_size = out_size
        self.global_layer = MetaLayer(
            None,
            None,
            GlobalModel(1 + hidden_size, hidden_size * 2, hidden_size),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size), nn.Tanh(),
            layer_init(nn.Linear(2 * hidden_size, out_size), std=out_std))

    def forward(self, x, edge_index, edge_attr, u, batch):
        x, edge_attr, u = self.global_layer(x, edge_index, edge_attr, u, batch)
        return self.head(u)

    def fine_tuning(self):
        # TODO freeze prev layers?
        # for param in self.parameters():
        #     param.requires_grad = False
        self.head[-1] = layer_init(nn.Linear(2 * self.hidden_size,
                                             self.out_size),
                                   std=self.out_std)
