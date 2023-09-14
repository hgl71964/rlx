from rlx.frontend.rewrite_rule import PATTERN_ID_MAP
from rlx.frontend import Node, Edge, Graph, RewriteRule, EdgePattern, NodePattern, node_pattern, const_pattern, symbol_pattern
from rlx.frontend.registry import register_node_type, get_node_type, clear_node_type

from rlx.rw_engine.parser import Parser
from rlx.rw_engine.environment.pattern_match import PatternMatch

from rlx.extern.expr.expr_utils import expr_edge, expr_node

from rlx.extern.expr.math_def import (r1,
                                      r2,
                                      r3,
                                      r4,
                                      r11,)  # yapf: disable




from absl import app
from absl import flags

FLAGS = flags.FLAGS


###############
#### Test 1
###############
class G1(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []

        e1 = expr_edge(0, 0, node_types["Const"], None)
        e2 = expr_edge(1, 0, node_types["Const"], None)
        n = expr_node(2, None, node_types["Add"], [e1, e2])
        e3 = expr_edge(3, None, node_types["Add"], n)
        n.outputs = [e3]

        self.edges.append(e1)
        self.edges.append(e2)
        self.edges.append(e3)
        self.nodes.append(n)

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw1(node_types):
    return [r1(node_types)]


class G2(Graph):
    def __init__(self, node_types):
        # ((t + tw) * x + b * x) + more
        self.nodes = []
        self.edges = []
        # input
        t = expr_edge(0, None, node_types["Const"], None)
        tw = expr_edge(1, None, node_types["Const"], None)

        # a = expr_edge(0, None, node_types["Const"], None)
        b = expr_edge(2, None, node_types["Const"], None)
        x = expr_edge(3, None, node_types["Const"], None)
        more = expr_edge(4, None, node_types["Const"], None)

        # a = t + tw
        a_node = expr_node(5, None, node_types["Add"], [t, tw])
        a = a_node.out(6)

        # x
        m1 = expr_node(7, None, node_types["Mul"], [a, x])
        m1_out = m1.out(8)
        # x*b
        m2 = expr_node(9, None, node_types["Mul"], [b, x])
        m2_out = m2.out(10)

        add = expr_node(11, None, node_types["Add"], [m1_out, m2_out])
        add_out = add.out(12)

        more_node = expr_node(13, None, node_types["Add"], [add_out, more])
        more_out = more_node.out(14)

        self.edges.extend(
            [t, tw, b, x, more, a, m1_out, m2_out, add_out, more_out])
        self.nodes.extend([a_node, m1, m2, add, more_node])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw2(node_types):
    # asysmetric
    return [r11(node_types)]


class G3(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, None, node_types["Const"], None)
        b = expr_edge(1, None, node_types["Const"], None)
        x = expr_edge(2, None, node_types["Var"], None)
        more = expr_edge(3, None, node_types["Var"], None)
        # pow1
        p1 = expr_node(4, None, node_types["Pow"], [x, a])
        p1_out = p1.out(5)
        # pow2
        p2 = expr_node(6, None, node_types["Pow"], [x, b])
        p2_out = p2.out(7)

        add_node = expr_node(8, None, node_types["Add"], [p1_out, p2_out])
        add_out = add_node.out(9)

        more_node = expr_node(10, None, node_types["Add"], [add_out, more])
        more_out = more_node.out(11)

        self.edges.extend([a, b, x, more, p1_out, p2_out, add_out, more_out])
        self.nodes.extend([p1, p2, add_node, more_node])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw3(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "pow(x, b) + pow(x, c) => pow(x, b+c)"
            self.x, self.tx = symbol_pattern()
            self.b, self.tb = const_pattern()
            self.c, self.tc = const_pattern()

        def source_pattern(self):
            pow1 = node_pattern(node_types["Pow"], [self.x, self.b],
                                n_outputs=1)
            pow2 = node_pattern(node_types["Pow"], [self.x, self.c],
                                n_outputs=1)
            out = node_pattern(node_types["Add"], [pow1, pow2], n_outputs=1)
            return [out]

        def target_pattern(self):
            add = node_pattern(node_types["Add"], [self.tb, self.tc],
                               n_outputs=1)
            out = node_pattern(node_types["Pow"], [self.tx, add], n_outputs=1)
            return [out]

    return [r1()]


class G4(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)
        zero = expr_edge(0, 0, node_types["Const"], None)
        more = expr_edge(0, None, node_types["Var"], None)

        add = expr_node(0, None, node_types["Add"], [x, zero])
        add_out = add.out()
        mul = expr_node(0, None, node_types["Mul"], [add_out, more])
        mul_out = mul.out()

        self.edges.extend([x, zero, more, add_out, mul_out])
        self.nodes.extend([add, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G4_alter(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)
        one = expr_edge(0, 1, node_types["Const"], None)
        more = expr_edge(0, None, node_types["Var"], None)

        add = expr_node(0, None, node_types["Add"], [x, one])
        add_out = add.out()
        mul = expr_node(0, None, node_types["Mul"], [add_out, more])
        mul_out = mul.out()

        self.edges.extend([x, one, more, add_out, mul_out])
        self.nodes.extend([add, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G4_cancel_sub(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)
        sub = expr_node(0, None, node_types["Sub"], [x, x])
        sub_out = sub.out()

        self.edges.extend([x, sub_out])
        self.nodes.extend([sub])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw4(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "x + 0 => x"
            self.x, self.tx = symbol_pattern()
            self.zero, _ = const_pattern(attr=0)

        def source_pattern(self):
            out = node_pattern(node_types["Add"], [self.x, self.zero],
                               n_outputs=1)
            return [out]

        def target_pattern(self):
            return [self.tx]

    class r2(RewriteRule):
        def __init__(self):
            self.name = "x - x = 0"
            self.x, self.tx = symbol_pattern()
            _, self.tzero = const_pattern(attr=0)

        def source_pattern(self):
            out = node_pattern(node_types["Sub"], [self.x, self.x],
                               n_outputs=1)
            return [out]

        def target_pattern(self):
            return [self.tzero]

    return [r1(), r2()]


class G5_eq(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, ((1, 2), 3), node_types["Const"], None)
        b = expr_edge(-1, ((1, 2), 3), node_types["Const"], None)
        add = expr_node(-1, None, node_types["Add"], [a, b])
        add_out = add.out()

        self.edges.extend([a, b, add_out])
        self.nodes.extend([add])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G5_partial_eq(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, ((1, 0), 3), node_types["Const"], None)
        b = expr_edge(-1, ((1, 2), 3), node_types["Const"], None)
        add = expr_node(-1, None, node_types["Add"], [a, b])
        add_out = add.out()

        self.edges.extend([a, b, add_out])
        self.nodes.extend([add])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw5(node_types):
    class r(RewriteRule):
        def __init__(self):
            self.name = "a + b = b + a"
            self.a, self.ta = const_pattern()
            self.b, self.tb = const_pattern()

        def source_pattern(self):
            out = node_pattern(node_types["Add"], [self.a, self.b],
                               n_outputs=1)
            return [out]

        def target_pattern(self):
            out = node_pattern(node_types["Add"], [self.tb, self.ta],
                               n_outputs=1)
            return [out]

    class r_eq(r):
        def register_deps(self):
            self.a.attr == self.b.attr

    class r_paritial_eq1(r):
        def register_deps(self):
            self.a.attr[0][1] == self.b.attr[0][1]

    class r_paritial_eq2(r):
        def register_deps(self):
            self.a.attr[0][0] == self.b.attr[0][0]

    class r_const(r):
        def register_deps(self):
            self.a.attr[1] == 3

    return [r_eq(), r_paritial_eq1(), r_paritial_eq2(), r_const()]


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


def test_expr():
    print("=======")
    print("test expr")
    print("=======")
    node_types = [
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        # "Ln",
        "Sqrt",
        "Sin",
        "Cos",

        # leaf;
        "Var",
        "Const",
    ]
    clear_node_type()
    register_node_type(node_types)
    node_types, _, _ = get_node_type()

    # Test1:
    print("Test1: ")
    g1 = G1(node_types)
    r1 = rw1(node_types)
    for r in r1:
        r.initialise()

    # parser + pattern match
    parser = Parser(g1)
    parser._build_mapping()

    # print parser rlx_Graph
    # for e in parser.edges:
    #     print(f"Edge: {e.idx}")
    #     if e.trace is None:
    #         print("trace: None")
    #     else:
    #         print(f"trace: {e.trace.idx}")

    #     print("Uses", end="::")
    #     for n in e.uses:
    #         print(f" {n.idx}",end=" ")
    #     print()

    # print()
    # for n in parser.nodes:
    #     print(f"Node: {n.idx}")

    #     print("Input", end="::")
    #     for e in n.inputs:
    #         print(f" {e.idx}",end=" ")
    #     print()
    #     print("Output", end="::")
    #     for e in n.outputs:
    #         print(f" {e.idx}",end=" ")
    #     print()
    # print()

    pattern_map = PatternMatch().build(parser.edges, r1)
    for rule_id, pmaps in pattern_map.items():
        locs = len(pmaps)
        print(f"++rule ID: {rule_id};; n_match: {locs}++")
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

    # Test2:
    t = "PM2"
    print(f"{t}: ")
    g = G2(node_types)
    r = rw2(node_types)
    print_test(g, r, t, viz=False)

    # Test3:
    # print("Test3: ")
    # g = G3(node_types)
    # r = rw3(node_types)
    # print_test(g, r)

    # # Test4: data-dependent; (+ ?a 0) => ?a
    # print("Test4: ")
    # g = G4(node_types)
    # r = rw4(node_types)
    # print_test(g, r)

    # print("Test4-alternative: ")
    # g = G4_alter(node_types)
    # r = rw4(node_types)
    # print_test(g, r)

    # # cancel-sub; a - a = 0
    # print("Test4-cancel-sub: ")
    # g = G4_cancel_sub(node_types)
    # r = rw4(node_types)
    # print_test(g, r)

    # print("Test5: input dependencies test")
    # g = G5_eq(node_types)
    # r = rw5(node_types)
    # print_test(g, r)

    # g = G5_partial_eq(node_types)
    # r = rw5(node_types)
    # print_test(g, r)

    # print("Test: full expr")
    # clear_node_type()
    # define_node_types()
    # print("=" * 40)
    # fn = "data/rlx/inputs/MATH-5-0.pkl"
    # lang = get_lang("MATH")()
    # expr = load_expr(lang, fn)
    # print("Loaded expression: %s", expr)
    # print("=" * 40)
    # nt, _, _ = get_node_type()
    # g = expr_graph(expr, nt)
    # r = define_rewrite_rules(nt)
    # print_test(g, r, False)


###############
#### Test 1
###############
def main(_):
    test_expr()


if __name__ == "__main__":
    app.run(main)
