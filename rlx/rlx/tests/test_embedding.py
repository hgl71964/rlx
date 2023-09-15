from collections import namedtuple

from rlx.frontend import Node, Edge, Graph
from rlx.frontend.registry import register_node_type, get_node_type, clear_node_type

from rlx.rw_engine.parser import Parser
from rlx.rw_engine.environment.env import make_env

from rlx.extern.expr.expr_utils import expr_edge, expr_node

from rlx.extern.expr.math_def import (r1,
                                      r11,
                                      r14,
                                      )  # yapf: disable


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


class G14(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 1, node_types["Const"], None)
        c = expr_edge(2, 2, node_types["Const"], None)
        d = expr_edge(3, 2, node_types["Const"], None)

        mul0 = expr_node(4, None, node_types["Mul"], [a, b])
        mul0_out = mul0.out(5)

        pow1 = expr_node(6, None, node_types["Pow"], [mul0_out, c])
        pow1_out = pow1.out(7)

        mul = expr_node(8, None, node_types["Mul"], [pow1_out, d])
        mul_out = mul.out(9)

        self.edges.extend([a, b, c, d, mul0_out, pow1_out, mul_out])
        self.nodes.extend([mul0, pow1, mul])


def rw14(node_types):
    return [r14(node_types)]


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

        g.render(cleanup=True, format="png")

        # print edge; because we don't plot
        print("edge_index shape:  edge_attr shape")
        print(pyg_g['edge_index'].shape, pyg_g["edge_attr"].shape)
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

    # Test2:
    print("Test2: ")
    g = G2(node_types)
    r = rw2(node_types)
    plot_embedding(g, r, 2, None, viz=True)

    n = 14
    print(f"Test{n}: ")
    g = G14(node_types)
    r = rw14(node_types)
    plot_embedding(g, r, n, None, viz=True)


def main(_):
    test_expr()


if __name__ == "__main__":
    app.run(main)
