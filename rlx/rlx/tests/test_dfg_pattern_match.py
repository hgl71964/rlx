from rlx.frontend.rewrite_rule import PATTERN_ID_MAP
from rlx.frontend import Graph
from rlx.frontend.registry import get_node_type

from rlx.rw_engine.parser import Parser
from rlx.rw_engine.environment.pattern_match import PatternMatch

from absl import app
from absl import flags

# NOTE: this defines graph and rewrite rules
from rlx.extern.hidet.hidet_utils import *
from rlx.extern.hidet.hidet_def import *

FLAGS = flags.FLAGS


def print_test(g, r, test_name="test1", verbose=True, viz=False):
    for _r in r:
        _r.initialise()
    parser = Parser(g)
    parser._build_mapping()
    if viz:
        parser.viz(parser.edges, test_name, check=False)
    pattern_map = PatternMatch().build(parser.edges, r)
    for rule_id, pmaps in pattern_map.items():
        locs = len(pmaps)
        print(f"++rule ID: {rule_id};; n_match: {locs}++")
        if verbose:
            for loc_id, pmap in enumerate(pmaps):
                print(f"loc ID: {loc_id}")
                for k, v in pmap.items():
                    if v[0] == 0:
                        vid = v[1]
                        v = parser.edge_map[vid]
                        k = PATTERN_ID_MAP[k]
                        print("edge id", vid, v.get_type(), k.is_const)
                    elif v[0] == 1:
                        vid = v[1]
                        v = parser.node_map[vid]
                        k = PATTERN_ID_MAP[k]
                        print("node", vid, v.get_type(), k.node_type)
                    else:
                        raise
    print()


def out(op, attr=None):
    # utility to auto-generate output; edge_type will be inferred
    o = DFG_Edge(-1, attr=attr, edge_type=None, trace=op)
    op.outputs.append(o)
    return o


class DG1(Graph):
    def __init__(self, node_types):
        v1 = DFG_Edge(0, None, node_types["Var"], None)
        c1 = DFG_Edge(0, None, node_types["Const"], None)
        conv1 = DFG_Op(0, None, node_types["conv2d"], [v1, c1])
        conv1_out = out(conv1)

        c2 = DFG_Edge(0, None, node_types["Const"], None)
        add1 = DFG_Op(1, None, node_types["add"], [conv1_out, c2])
        add1_out = out(add1)

        v4 = DFG_Edge(0, None, node_types["Var"], None)
        c4 = DFG_Edge(0, None, node_types["Const"], None)
        conv2 = DFG_Op(0, None, node_types["conv2d"], [v4, c4])
        conv2_out = out(conv2)

        c5 = DFG_Edge(0, None, node_types["Const"], None)
        add2 = DFG_Op(1, None, node_types["add"], [conv2_out, c5])
        add2_out = out(add2)

        last_add = DFG_Op(-1, None, node_types["add"], [add1_out, add2_out])
        last_add_out = out(last_add)

        self.nodes = [conv1, add1, conv2, add2, last_add]
        self.edges = [
            v1, c1, conv1_out, conv2_out, c2, add1_out, v4, c4, c5, add2_out,
            last_add_out
        ]

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def drw1(node_types):
    return [
        # NOTE: how ar1 and ar5 demonstrate asymmetric, when consider var pattern
        ar1(node_types),
        ar5(node_types),
        ar6(node_types),
        ar7(node_types),
    ]


def test_dataflow():
    print("=======")
    print("test dfg")
    print("=======")
    node_types, _, _ = get_node_type()

    # Test1:
    g1 = DG1(node_types)
    r1 = drw1(node_types)
    print_test(g1, r1, test_name="test1", verbose=True, viz=False)


def main(_):
    test_dataflow()


if __name__ == "__main__":
    app.run(main)
