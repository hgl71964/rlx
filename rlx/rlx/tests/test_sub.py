from collections import namedtuple

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

import torch

Config = namedtuple('Config', [
    "max_loc",
    "h",
    "normalize_reward",
])
CONFIG = Config(
    max_loc=10,
    h=100,
    normalize_reward=1,
)

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
        t = expr_edge(1, None, node_types["Const"], None)
        tw = expr_edge(2, None, node_types["Const"], None)

        b = expr_edge(3, None, node_types["Const"], None)
        x = expr_edge(4, None, node_types["Const"], None)
        more = expr_edge(5, None, node_types["Const"], None)

        # gen a
        a_node = expr_node(6, None, node_types["Add"], [t, tw])
        a = a_node.out(7)

        # x*a
        m1 = expr_node(8, None, node_types["Mul"], [a, x])
        m1_out = m1.out(9)
        # x*b
        m2 = expr_node(10, None, node_types["Mul"], [b, x])
        m2_out = m2.out(11)

        add = expr_node(12, None, node_types["Add"], [m1_out, m2_out])
        add_out = add.out(13)

        more_node = expr_node(14, None, node_types["Add"], [add_out, more])
        more_out = more_node.out(15)

        self.edges.extend(
            [t, tw, b, x, more, a, m1_out, m2_out, add_out, more_out])
        self.nodes.extend([a_node, m1, m2, add, more_node])


def rw2(node_types):
    return [r11(node_types)]


class G3(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, None, node_types["Const"], None)
        b = expr_edge(0, None, node_types["Const"], None)
        x = expr_edge(0, None, node_types["Var"], None)
        more = expr_edge(0, None, node_types["Var"], None)
        # pow1
        p1 = expr_node(0, None, node_types["Pow"], [x, a])
        p1_out = p1.out()
        # pow2
        p2 = expr_node(0, None, node_types["Pow"], [x, b])
        p2_out = p2.out()

        add_node = expr_node(0, None, node_types["Add"], [p1_out, p2_out])
        add_out = add_node.out()

        more_node = expr_node(0, None, node_types["Add"], [add_out, more])
        more_out = more_node.out()

        self.edges.extend([a, b, x, more, p1_out, p2_out, add_out, more_out])
        self.nodes.extend([p1, p2, add_node, more_node])


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

        add = expr_node(0, None, node_types["Add"], [x, zero])
        add_out = add.out()

        self.edges.extend([x, zero, add_out])
        self.nodes.extend([add])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G4_v2(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)
        y = expr_edge(0, None, node_types["Var"], None)
        zero = expr_edge(0, 0, node_types["Const"], None)
        more = expr_edge(0, None, node_types["Var"], None)

        mul = expr_node(0, None, node_types["Mul"], [x, y])
        mul_out = mul.out()

        add = expr_node(0, None, node_types["Add"], [mul_out, zero])
        add_out = add.out()

        more_pow = expr_node(0, None, node_types["Pow"], [add_out, more])
        pow_out = more_pow.out()

        self.edges.extend([x, y, zero, more, mul_out, add_out, pow_out])
        self.nodes.extend([mul, add, more_pow])

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


class G4_cancel_sub_v2(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)
        a = expr_edge(0, None, node_types["Const"], None)

        sub = expr_node(0, None, node_types["Sub"], [x, x])
        sub_out = sub.out()

        add = expr_node(0, None, node_types["Add"], [sub_out, a])
        add_out = add.out()

        self.edges.extend([x, a, sub_out, add_out])
        self.nodes.extend([sub, add])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G4_cancel_sub_v3(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)
        y = expr_edge(0, None, node_types["Var"], None)
        a = expr_edge(0, None, node_types["Const"], None)

        mul = expr_node(0, None, node_types["Mul"], [x, y])
        mul_out = mul.out()

        sub = expr_node(0, None, node_types["Sub"], [mul_out, mul_out])
        sub_out = sub.out()

        add = expr_node(0, None, node_types["Add"], [sub_out, a])
        add_out = add.out()

        self.edges.extend([x, y, a, mul_out, sub_out, add_out])
        self.nodes.extend([mul, sub, add])

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
            self.x, _ = symbol_pattern()
            _, self.tzero = const_pattern(attr=0)

        def source_pattern(self):
            out = node_pattern(node_types["Sub"], [self.x, self.x],
                               n_outputs=1)
            return [out]

        def target_pattern(self):
            return [self.tzero]

    return [r1(), r2()]


class G5(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Var"], None)

        sin = expr_node(0, None, node_types["Sin"], [x])
        sin_out = sin.out()

        d = expr_node(0, None, node_types["d"], [x, sin_out])
        d_out = d.out()

        self.edges.extend([x, sin_out, d_out])
        self.nodes.extend([sin, d])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw5(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "d x (sin x) => cos x "
            self.x, self.tx = symbol_pattern()

        def source_pattern(self):
            sin = node_pattern(node_types["Sin"], [self.x], n_outputs=1)
            out = node_pattern(node_types["d"], [self.x, sin], n_outputs=1)
            return [out]

        def target_pattern(self):
            out = node_pattern(node_types["Cos"], [self.tx], n_outputs=1)
            return [out]

    return [r1()]


class G5_sub_canon(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, -3, node_types["Const"], None)

        sub = expr_node(-1, None, node_types["Sub"], [a, b])
        sub_out = sub.out()

        self.edges.extend([a, b, sub_out])
        self.nodes.extend([sub])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G5_sub_canon_more(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, -3, node_types["Const"], None)
        c = expr_edge(-1, -10, node_types["Const"], None)
        d = expr_edge(-1, 2, node_types["Const"], None)

        add = expr_node(-1, None, node_types["Add"], [a, b])
        add_out = add.out()

        sub = expr_node(-1, None, node_types["Sub"], [add_out, c])
        sub_out = sub.out()

        mul = expr_node(-1, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out()

        self.edges.extend([a, b, c, d, add_out, sub_out, mul_out])
        self.nodes.extend([add, sub, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw5_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "a-b => a+(-1*b)"
            self.a, self.ta = const_pattern()
            self.b, self.tb = const_pattern()
            _, self.t_minus_one = const_pattern(attr=-1)

        def source_pattern(self):
            sub = node_pattern(node_types["Sub"], [self.a, self.b],
                               n_outputs=1)
            return [sub]

        def target_pattern(self):
            mul = node_pattern(node_types["Mul"], [self.t_minus_one, self.tb],
                               n_outputs=1)
            add = node_pattern(node_types["Add"], [self.ta, mul], n_outputs=1)
            return [add]

    return [r1()]


class G6(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 0, node_types["Const"], None)

        add = expr_node(-1, None, node_types["Add"], [a, b])
        add_out = add.out()

        self.edges.extend([a, b, add_out])
        self.nodes.extend([add])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G6_more(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 0, node_types["Const"], None)
        c = expr_edge(-1, -10, node_types["Const"], None)
        d = expr_edge(-1, 2, node_types["Const"], None)

        add = expr_node(-1, None, node_types["Add"], [a, b])
        add_out = add.out()

        sub = expr_node(-1, None, node_types["Sub"], [add_out, c])
        sub_out = sub.out()

        mul = expr_node(-1, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out()

        self.edges.extend([a, b, c, d, add_out, sub_out, mul_out])
        self.nodes.extend([add, sub, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw6_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "zero-add"
            self.a, self.ta = const_pattern()
            self.zero, _ = const_pattern(attr=0)

        def source_pattern(self):
            add = node_pattern(node_types["Add"], [self.a, self.zero],
                               n_outputs=1)
            return [add]

        def target_pattern(self):
            return [self.ta]

    return [r1()]


class G7(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 0, node_types["Const"], None)

        mul = expr_node(-1, None, node_types["Mul"], [a, b])
        mul_out = mul.out()

        self.edges.extend([a, b, mul_out])
        self.nodes.extend([mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class G7_more(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 0, node_types["Const"], None)
        c = expr_edge(-1, -10, node_types["Const"], None)
        d = expr_edge(-1, 2, node_types["Const"], None)

        mul0 = expr_node(-1, None, node_types["Mul"], [a, b])
        mul_out0 = mul0.out()

        sub = expr_node(-1, None, node_types["Sub"], [mul_out0, c])
        sub_out = sub.out()

        mul = expr_node(-1, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out()

        self.edges.extend([a, b, c, d, mul_out0, sub_out, mul_out])
        self.nodes.extend([mul0, sub, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw7_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "zero-mul"
            self.a, _ = const_pattern(None)
            self.zero, self.tzero = const_pattern(attr=0)

        def source_pattern(self):
            mul = node_pattern(node_types["Mul"], [self.a, self.zero],
                               n_outputs=1)
            return [mul]

        def target_pattern(self):
            return [self.tzero]

    return [r1()]


class G9(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 1, node_types["Const"], None)
        d = expr_edge(-1, 2, node_types["Const"], None)

        mul0 = expr_node(-1, None, node_types["Mul"], [a, b])
        mul_out0 = mul0.out()

        sub = expr_node(-1, None, node_types["Sub"], [mul_out0, mul_out0])
        sub_out = sub.out()

        mul = expr_node(-1, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out()

        self.edges.extend([a, b, d, mul_out0, sub_out, mul_out])
        self.nodes.extend([mul0, sub, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw9_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "cancel-sub"
            self.a, self.ta = const_pattern()
            _, self.tzero = const_pattern(attr=0)

        def source_pattern(self):
            out = node_pattern(node_types["Sub"], [self.a, self.a],
                               n_outputs=1)
            return [out]

        def target_pattern(self):
            return [self.tzero]

    return [r1()]


class G14(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 1, node_types["Const"], None)
        c = expr_edge(-1, 2, node_types["Const"], None)
        d = expr_edge(-1, 2, node_types["Const"], None)

        mul0 = expr_node(-1, None, node_types["Mul"], [a, b])
        mul0_out = mul0.out()

        pow1 = expr_node(-1, None, node_types["Pow"], [mul0_out, c])
        pow1_out = pow1.out()

        mul = expr_node(-1, None, node_types["Mul"], [pow1_out, d])
        mul_out = mul.out()

        self.edges.extend([a, b, c, d, mul0_out, pow1_out, mul_out])
        self.nodes.extend([mul0, pow1, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw14_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "pow2"
            self.a, self.ta = const_pattern()
            self.two, _ = const_pattern(attr=2)

        def source_pattern(self):
            pow1 = node_pattern(node_types["Pow"], [self.a, self.two],
                                n_outputs=1)
            return [pow1]

        def target_pattern(self):
            mul = node_pattern(node_types["Mul"], [self.ta, self.ta],
                               n_outputs=1)
            return [mul]

    return [r1()]


class G18(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, -2, node_types["Const"], None)
        b = expr_edge(-1, 1, node_types["Const"], None)
        # c = expr_edge(-1, 2, node_types["Const"], None)
        d = expr_edge(-1, 2, node_types["Const"], None)

        add = expr_node(-1, None, node_types["Add"], [a, b])
        add_out = add.out()

        cos = expr_node(-1, None, node_types["Cos"], [add_out])
        cos_out = cos.out()

        diff = expr_node(-1, None, node_types["Diff"], [add_out, cos_out])
        diff_out = diff.out()

        mul = expr_node(-1, None, node_types["Mul"], [diff_out, d])
        mul_out = mul.out()

        self.edges.extend([a, b, d, add_out, cos_out, diff_out, mul_out])
        self.nodes.extend([add, cos, diff, mul])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw18_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "d-cos"
            self.a, self.ta = const_pattern()
            _, self.tmone = const_pattern(-1)

        def source_pattern(self):
            cos = node_pattern(node_types["Cos"], [self.a], 1)
            diff = node_pattern(node_types["Diff"], [self.a, cos], 1)
            return [diff]

        def target_pattern(self):
            sin = node_pattern(node_types["Sin"], [self.ta], 1)
            mul = node_pattern(node_types["Mul"], [sin, self.tmone], 1)
            return [mul]

    return [r1()]


class G24(Graph):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(-1, 1, node_types["Const"], None)
        b = expr_edge(-1, 2, node_types["Const"], None)
        c = expr_edge(-1, 3, node_types["Const"], None)

        mul = expr_node(-1, None, node_types["Mul"], [a, b])
        mul_out = mul.out()

        ing = expr_node(-1, None, node_types["Integral"], [mul_out, c])
        ing_out = ing.out()

        self.edges.extend([a, b, c, mul_out, ing_out])
        self.nodes.extend([mul, ing])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def rw24_(node_types):
    class r1(RewriteRule):
        def __init__(self):
            self.name = "i-parts"
            self.a, self.ta = const_pattern()
            self.b, self.tb = const_pattern()
            self.c, self.tc = const_pattern()

        def source_pattern(self):
            mul = node_pattern(node_types["Mul"], [self.a, self.b], 1)
            ing = node_pattern(node_types["Integral"], [mul, self.c], 1)
            return [ing]

        def target_pattern(self):
            # first part
            ing1 = node_pattern(node_types["Integral"], [self.tb, self.tc], 1)
            mul1 = node_pattern(node_types["Mul"], [self.ta, ing1], 1)

            # second part
            diff = node_pattern(node_types["Diff"], [self.tc, self.ta], 1)
            ing2 = node_pattern(node_types["Integral"], [self.tb, self.tc], 1)
            mul2 = node_pattern(node_types["Mul"], [diff, ing2], 1)
            ing3 = node_pattern(node_types["Integral"], [mul2, self.tc], 1)

            sub = node_pattern(node_types["Sub"], [mul1, ing3], 1)
            return [sub]

    return [r1()]


def print_env(g, r, n, action=None, viz=False):
    n = str(n)
    gym_id = "env-v0"
    parser = Parser(g)
    for _r in r:
        _r.initialise()
    env = make_env(
        gym_id,
        parser,
        lambda x, y, z: 0,
        r,
        seed=0,
        config=CONFIG,
    )()
    (pyg_g, m, _, _), _ = env.reset()
    print("+++reset+++")
    # print(pyg_g)
    if action is None:
        action = tuple([torch.tensor(0).long() for _ in range(2)])
    for rule_id, pmaps in m.items():
        locs = len(pmaps)
        print(f"rule ID: {rule_id}", end="; ")
        print(f"{locs} matches")

    if viz:
        f = f"sub_{n}_reset"
        env.unwrapped.viz(f)

    print("+++step+++")
    obs, reward, terminated, truncated, info = env.step(action)

    if viz:
        f = f"sub_{n}_step"
        env.unwrapped.viz(f)


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
    n = 1
    print(f"Test{n}: ")
    g = G1(node_types)
    r = rw1(node_types)
    print_env(g, r, n, None, viz=False)

    # Test2: substitution
    n = 2
    print(f"Test{n}: ")
    g = G2(node_types)
    r = rw2(node_types)
    print_env(g, r, n, None, viz=True)


#
#    # Test3: substitution at inputs boundary
#    print("Test3: ")
#    g = G3(node_types)
#    r = rw3(node_types)
#    print_env(g, r, 3, None, viz=False)
#
#    # Test4: edge cases? e.g. (+ ?a 0) => ?a
#    print("Test4: ")
#    g = G4(node_types)
#    r = rw4(node_types)
#    print_env(g, r, 4, None, False)
#
#    print("Test4_v2: ")
#    g = G4_v2(node_types)
#    r = rw4(node_types)
#    print_env(g, r, 4.02, None, False)
#
#    print("Test4-cancel-sub: ")
#    g = G4_cancel_sub(node_types)
#    r = rw4(node_types)
#    a = (torch.tensor(1).long(), torch.tensor(0).long())
#    print_env(g, r, 4.1, a, viz=False)
#
#    print("Test4-cancel-sub_v2: ")
#    g = G4_cancel_sub_v2(node_types)
#    r = rw4(node_types)
#    a = (torch.tensor(1).long(), torch.tensor(0).long())
#    print_env(g, r, 4.2, a, viz=False)
#
#    print("Test4-cancel-sub_v3: ")
#    g = G4_cancel_sub_v3(node_types)
#    r = rw4(node_types)
#    a = (torch.tensor(1).long(), torch.tensor(0).long())
#    print_env(g, r, 4.3, a, viz=False)
#
#    # Test5: edge cases? e.g. (d ?x (sin ?x) )
#    print("Test5: ")
#    g = G5(node_types)
#    r = rw5(node_types)
#    print_env(g, r, 5, None, viz=False)
#
#    # Test all: expr rewrite rules
#    print("TestAll:")
#    print("sub_canon:")
#    g = G5_sub_canon(node_types)
#    r = rw5_(node_types)
#    print_env(g, r, 5.1, None, viz=False)
#
#    g = G5_sub_canon_more(node_types)
#    r = rw5_(node_types)
#    print_env(g, r, 5.2, None, viz=False)
#
#    print("zero-add:")
#    g = G6(node_types)
#    r = rw6_(node_types)
#    print_env(g, r, 6.1, None, viz=False)
#
#    g = G6_more(node_types)
#    r = rw6_(node_types)
#    print_env(g, r, 6.2, None, viz=False)
#
#    print("zero-mul:")
#    g = G7(node_types)
#    r = rw7_(node_types)
#    print_env(g, r, 7.1, None, viz=False)
#
#    print("cancel-sub:")
#    g = G9(node_types)
#    r = rw9_(node_types)
#    print_env(g, r, 9, None, viz=False)
#
#    print("pow2:")
#    g = G14(node_types)
#    r = rw14_(node_types)
#    print_env(g, r, 14, None, viz=False)
#
#    print("d-cos:")
#    g = G18(node_types)
#    r = rw18_(node_types)
#    print_env(g, r, 18, None, viz=False)
#
#    print("i-part:")
#    g = G24(node_types)
#    r = rw24_(node_types)
#    print_env(g, r, 24, None, viz=True)


def main(_):
    test_expr()


if __name__ == "__main__":
    app.run(main)
