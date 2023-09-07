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


def print_test(g, r, verbose=True):
    for _r in r:
        _r.initialise()
    parser = Parser(g)
    parser._build_mapping()
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


class DG1(Graph):
    def __init__(self, node_types):
        data = DFG_Edge(0, None, node_types["Var"], None)
        K = DFG_Edge(1, None, node_types["Const"], None)
        Q = DFG_Edge(2, None, node_types["Const"], None)
        m1 = DFG_Op(3, None, node_types["Matmul"], [data, K])
        m2 = DFG_Op(4, None, node_types["Matmul"], [data, Q])
        out1 = DFG_Edge(5, None, node_types["Matmul"], trace=m1)
        out2 = DFG_Edge(6, None, node_types["Matmul"], trace=m2)

        self.nodes = []
        self.edges = []
        self.edges.append(data)
        self.edges.append(K)
        self.edges.append(Q)
        self.edges.append(out1)
        self.edges.append(out2)
        self.nodes.append(m1)
        self.nodes.append(m2)

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def drw1(node_types):
    return [
        ar5(node_types),
    ]


def test_dataflow():
    print("=======")
    print("test dfg")
    print("=======")
    node_types, _, _ = get_node_type()

    # Test1:
    g1 = DG1(node_types)
    r1 = drw1(node_types)
    for r in r1:
        r.initialise()

    # parser + pattern match
    parser = Parser(g1)
    parser._build_mapping()
    pattern_map = PatternMatch().build(parser.edges, r1)
    for rule_id, pmaps in pattern_map.items():
        print(f"rule ID: {rule_id}")
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


def main(_):
    test_dataflow()


if __name__ == "__main__":
    app.run(main)
