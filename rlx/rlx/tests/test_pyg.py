from typing import Union
import numpy as np

import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import Data

from torch_geometric.nn.conv import (GATConv, GATv2Conv, GCNConv, GINConv,
                                     MessagePassing, PNAConv, SAGEConv)
from torch_geometric.nn.models.basic_gnn import BasicGNN, GAT

from rlx.rw_engine.agents.ppo_agent import GraphPPO

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("node_lim", 500, "enode limit")


class GATNetwork(nn.Module):

    def __init__(
            self,
            num_node_features: int,
            n_actions: int,  # fix action space = num of nodes (Eclass + Enode)
            n_layers: int = 3,
            hidden_size: int = 128,
            out_std=np.sqrt(2),
            dropout=0.0,
            use_edge_attr=True,
            add_self_loops=False,
            edge_dim=None):
        super().__init__()
        self.n_actions = n_actions
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            assert (edge_dim is not None), "use edge attr but edge_dim is None"
        self.gnn = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       num_layers=n_layers,
                       add_self_loops=add_self_loops,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True,
                       edge_dim=(edge_dim if self.use_edge_attr else None))

        if dropout == 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                # convert node-level feature to scalar, representing its logit
                layer_init(nn.Linear(hidden_size, 1), std=out_std))
        else:
            self.head = nn.Sequential(nn.Dropout(p=dropout),
                                      nn.Linear(hidden_size, hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_size, 1))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=(data.edge_attr if self.use_edge_attr else None))
        # print("[GNN] ", x.shape)
        x = self.head(x).reshape(-1, self.n_actions)
        # print("[GNN] ", x.shape)
        return x


def main(_):
    agent = GraphPPO(
        actor_n_action=2,
        critic_n_action=2,
        n_node_features=1,
        n_edge_features=1,
        use_edge_attr=False,
    )

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    d1 = Data(x=x, edge_index=edge_index)

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    d2 = Data(x=x, edge_index=edge_index)

    print(d1)
    g = pyg.data.Batch.from_data_list([d1, d2])
    print("num of graph: ", g.num_graphs)
    print(g)

    print(g.to_data_list())

    agent.get_action_and_value(g, None)

    # TODO inspect if GAT update edge features
    gnn = GAT(
        in_channels=10,
        hidden_channels=4,
        out_channels=4,
        num_layers=2,
        add_self_loops=False,
        act="leaky_relu",
        v2=True,
        edge_dim=0,
    )


if __name__ == "__main__":
    app.run(main)
