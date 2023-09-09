from collections import namedtuple

import matplotlib.pyplot as plt  # type: ignore
import torch_geometric as pyg

from rlx.frontend import Node, Edge, Graph, RewriteRule, EdgePattern, NodePattern, node_pattern, const_pattern, symbol_pattern
from rlx.frontend.registry import register_node_type, get_node_type, clear_node_type

from rlx.rw_engine.parser import Parser
from rlx.rw_engine.environment.env import make_env

from rlx.extern.expr.expr_utils import expr_edge, expr_node

from rlx.extern.expr.math_def import (r1,
                                      r2,
                                      r3,
                                      r4,
                                      r11,)  # yapf: disable


config = namedtuple('config', [
    "max_loc",
    "h",
    "normalize_reward",
])

from absl import app
from absl import flags

FLAGS = flags.FLAGS


###############
#### Test 1
###############
class G(Graph):
    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G1(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, 1, node_types["Const"], None)
        b = expr_edge(1, 2, node_types["Const"], None)
        add = expr_node(2, None, node_types["Add"], [a, b])
        add_out = add.out(3)

        self.edges.extend([a, b, add_out])
        self.nodes.extend([add])


def rw1(node_types):
    return [r1(node_types)]


class G2(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        t = expr_edge(0, None, node_types["Const"], None)
        tw = expr_edge(0, None, node_types["Const"], None)

        # a = expr_edge(0, None, node_types["Const"], None)
        b = expr_edge(0, None, node_types["Const"], None)
        x = expr_edge(0, None, node_types["Const"], None)
        more = expr_edge(0, None, node_types["Const"], None)

        # gen a
        a_node = expr_node(0, None, node_types["Add"], [t, tw])
        a = a_node.out()

        # x*a
        m1 = expr_node(0, None, node_types["Mul"], [a, x])
        m1_out = m1.out()
        # x*b
        m2 = expr_node(0, None, node_types["Mul"], [b, x])
        m2_out = m2.out()

        add = expr_node(0, None, node_types["Add"], [m1_out, m2_out])
        add_out = add.out()

        more_node = expr_node(0, None, node_types["Add"], [add_out, more])
        more_out = more_node.out()

        self.edges.extend(
            [t, tw, b, x, more, a, m1_out, m2_out, add_out, more_out])
        self.nodes.extend([a_node, m1, m2, add, more_node])


def rw2(node_types):
    # asysmetric
    return [r11(node_types)]


def plot_embedding(g, r, n, action=None, viz=False):
    n = str(n)
    gym_id = "env-v0"
    parser = Parser(g)
    for _r in r:
        _r.initialise()
    env = make_env(gym_id,
                   parser,
                   lambda x, y, z: 0,
                   r,
                   seed=0,
                   config=config(
                       max_loc=10,
                       h=100,
                       normalize_reward=1,
                   ))()
    (pyg_g, m, _, _), _ = env.reset()
    print("+++reset+++")
    for rule_id, pmaps in m.items():
        locs = len(pmaps)
        print(f"rule ID: {rule_id}", end="; ")
        print(f"{locs} matches")

    if viz:
        f = f"embedding_{n}_reset"

        # import networkx as nx
        # print("+++viz+++")
        # print(pyg_g['x'])
        # print(pyg_g['edge_attr'])
        # g = pyg.utils.to_networkx(pyg_g, remove_self_loops=True)
        # # g = pyg.utils.to_networkx(pyg_g, [str(x) for x in pyg_g['x']], [str(x) for x in pyg_g['edge_attr']], to_undirected=False)
        # nx.draw(g)
        # plt.show()

        import graphviz  # type: ignore
        g = graphviz.Digraph("embedding", format="pdf", filename=f"{f}")
        x = pyg_g['x']
        n_node = len(x)
        for i in range(n_node):
            attr = str(x[i])
            label = f"Node-{i}-{attr}"
            g.node(label, label, **{"shape": "circle"})

        edge_index = pyg_g['edge_index'].T
        for i, (src, dest) in enumerate(edge_index):
            src, dest = int(src), int(dest)
            src_label = f"Node-{src}-{x[src]}"
            dest_label = f"Node-{dest}-{x[dest]}"
            g.edge(src_label, dest_label)

        g.render()

        print(edge_index.shape, pyg_g["edge_attr"].shape)
        assert edge_index.shape[0] == pyg_g["edge_attr"].shape[0]
        for i, (edge_index,
                edge) in enumerate(zip(edge_index, pyg_g["edge_attr"])):
            print(i, edge_index, edge)


def test_expr():
    print("=======")
    print("test expr")
    print("=======")
    node_types = [
        "Diff",
        "Integral",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        # "Ln",
        "Sqrt",
        "Sin",
        "Cos",
        "d",

        # leaf;
        "Var",
        "Const",
    ]
    clear_node_type()
    register_node_type(node_types)
    node_types, _, _ = get_node_type()

    # Test1: x+y=>y+x
    print("Test1: ")
    g = G1(node_types)
    r = rw1(node_types)
    plot_embedding(g, r, 1, None, viz=True)

    # Test2: substitution
    print("Test2: ")
    g = G2(node_types)
    r = rw2(node_types)
    plot_embedding(g, r, 2, None, viz=True)


def main(_):
    test_expr()


if __name__ == "__main__":
    app.run(main)
