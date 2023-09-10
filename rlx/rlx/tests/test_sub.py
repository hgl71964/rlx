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
                                      r5,
                                      r6,
                                      r7,
                                      r9,
                                      r11,
                                      r12,
                                      r14,
                                      r17,
                                      r18,
                                      r24,
                                      )  # yapf: disable

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
        a = expr_edge(1, None, node_types["Const"], None)
        b = expr_edge(2, None, node_types["Const"], None)
        x = expr_edge(3, None, node_types["Const"], None)
        more = expr_edge(4, None, node_types["Const"], None)
        # pow1
        p1 = expr_node(5, None, node_types["Pow"], [x, a])
        p1_out = p1.out(6)
        # pow2
        p2 = expr_node(7, None, node_types["Pow"], [x, b])
        p2_out = p2.out(8)

        mul_node = expr_node(9, None, node_types["Mul"], [p1_out, p2_out])
        add_out = mul_node.out(10)

        more_node = expr_node(11, None, node_types["Add"], [add_out, more])
        more_out = more_node.out(12)

        self.edges.extend([a, b, x, more, p1_out, p2_out, add_out, more_out])
        self.nodes.extend([p1, p2, mul_node, more_node])


def rw3(node_types):
    return [r12(node_types)]


class G4(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Const"], None)
        zero = expr_edge(1, 0, node_types["Const"], None)

        add = expr_node(2, None, node_types["Add"], [x, zero])
        add_out = add.out(3)

        self.edges.extend([x, zero, add_out])
        self.nodes.extend([add])


class G4_v2(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Const"], None)
        y = expr_edge(1, None, node_types["Const"], None)
        zero = expr_edge(2, 0, node_types["Const"], None)
        more = expr_edge(3, None, node_types["Const"], None)

        mul = expr_node(4, None, node_types["Mul"], [x, y])
        mul_out = mul.out(5)

        add = expr_node(6, None, node_types["Add"], [mul_out, zero])
        add_out = add.out(7)

        more_pow = expr_node(8, None, node_types["Pow"], [add_out, more])
        pow_out = more_pow.out(9)

        self.edges.extend([x, y, zero, more, mul_out, add_out, pow_out])
        self.nodes.extend([mul, add, more_pow])


class G4_cancel_sub(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Const"], None)
        sub = expr_node(1, None, node_types["Sub"], [x, x])
        sub_out = sub.out(2)

        self.edges.extend([x, sub_out])
        self.nodes.extend([sub])


class G4_cancel_sub_v2(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Const"], None)
        a = expr_edge(1, None, node_types["Const"], None)

        sub = expr_node(2, None, node_types["Sub"], [x, x])
        sub_out = sub.out(3)

        add = expr_node(4, None, node_types["Add"], [sub_out, a])
        add_out = add.out(5)

        self.edges.extend([x, a, sub_out, add_out])
        self.nodes.extend([sub, add])


class G4_cancel_sub_v3(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Const"], None)
        y = expr_edge(1, None, node_types["Const"], None)
        a = expr_edge(2, None, node_types["Const"], None)

        mul = expr_node(3, None, node_types["Mul"], [x, y])
        mul_out = mul.out(4)

        sub = expr_node(5, None, node_types["Sub"], [mul_out, mul_out])
        sub_out = sub.out(6)

        add = expr_node(7, None, node_types["Add"], [sub_out, a])
        add_out = add.out(8)

        self.edges.extend([x, y, a, mul_out, sub_out, add_out])
        self.nodes.extend([mul, sub, add])


def rw4(node_types):
    return [r6(node_types), r9(node_types)]


class G5(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        x = expr_edge(0, None, node_types["Const"], None)

        sin = expr_node(1, None, node_types["Sin"], [x])
        sin_out = sin.out(2)

        d = expr_node(3, None, node_types["Diff"], [x, sin_out])
        d_out = d.out(4)

        self.edges.extend([x, sin_out, d_out])
        self.nodes.extend([sin, d])


def rw5(node_types):
    return [r17(node_types)]


class G5_sub_canon(G):
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


class G5_sub_canon_more(G):
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


def rw5_(node_types):
    return [r5(node_types)]


class G6(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 0, node_types["Const"], None)

        add = expr_node(2, None, node_types["Add"], [a, b])
        add_out = add.out(3)

        self.edges.extend([a, b, add_out])
        self.nodes.extend([add])


class G6_more(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 0, node_types["Const"], None)
        c = expr_edge(2, -10, node_types["Const"], None)
        d = expr_edge(3, 2, node_types["Const"], None)

        add = expr_node(4, None, node_types["Add"], [a, b])
        add_out = add.out(5)

        sub = expr_node(6, None, node_types["Sub"], [add_out, c])
        sub_out = sub.out(7)

        mul = expr_node(8, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out(9)

        self.edges.extend([a, b, c, d, add_out, sub_out, mul_out])
        self.nodes.extend([add, sub, mul])


def rw6_(node_types):
    return [r6(node_types)]


class G7(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 0, node_types["Const"], None)

        mul = expr_node(2, None, node_types["Mul"], [a, b])
        mul_out = mul.out(3)

        self.edges.extend([a, b, mul_out])
        self.nodes.extend([mul])


class G7_more(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 0, node_types["Const"], None)
        c = expr_edge(2, -10, node_types["Const"], None)
        d = expr_edge(3, 2, node_types["Const"], None)

        mul0 = expr_node(4, None, node_types["Mul"], [a, b])
        mul_out0 = mul0.out(5)

        sub = expr_node(6, None, node_types["Sub"], [mul_out0, c])
        sub_out = sub.out(7)

        mul = expr_node(8, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out(9)

        self.edges.extend([a, b, c, d, mul_out0, sub_out, mul_out])
        self.nodes.extend([mul0, sub, mul])


def rw7_(node_types):
    return [r7(node_types)]


class G9(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 1, node_types["Const"], None)
        d = expr_edge(2, 2, node_types["Const"], None)

        mul0 = expr_node(3, None, node_types["Mul"], [a, b])
        mul_out0 = mul0.out(4)

        sub = expr_node(5, None, node_types["Sub"], [mul_out0, mul_out0])
        sub_out = sub.out(6)

        mul = expr_node(7, None, node_types["Mul"], [sub_out, d])
        mul_out = mul.out(8)

        self.edges.extend([a, b, d, mul_out0, sub_out, mul_out])
        self.nodes.extend([mul0, sub, mul])


def rw9_(node_types):
    return [r9(node_types)]


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


def rw14_(node_types):
    return [r14(node_types)]


class G18(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, -2, node_types["Const"], None)
        b = expr_edge(1, 1, node_types["Const"], None)
        # c = expr_edge(-1, 2, node_types["Const"], None)
        d = expr_edge(2, 2, node_types["Const"], None)

        add = expr_node(3, None, node_types["Add"], [a, b])
        add_out = add.out(4)

        cos = expr_node(5, None, node_types["Cos"], [add_out])
        cos_out = cos.out(6)

        diff = expr_node(7, None, node_types["Diff"], [add_out, cos_out])
        diff_out = diff.out(8)

        mul = expr_node(9, None, node_types["Mul"], [diff_out, d])
        mul_out = mul.out(10)

        self.edges.extend([a, b, d, add_out, cos_out, diff_out, mul_out])
        self.nodes.extend([add, cos, diff, mul])


def rw18_(node_types):
    return [r18(node_types)]


class G24(G):
    def __init__(self, node_types):
        self.nodes = []
        self.edges = []
        # input
        a = expr_edge(0, 1, node_types["Const"], None)
        b = expr_edge(1, 2, node_types["Const"], None)
        c = expr_edge(2, 3, node_types["Const"], None)

        mul = expr_node(3, None, node_types["Mul"], [a, b])
        mul_out = mul.out(4)

        ing = expr_node(5, None, node_types["Integral"], [mul_out, c])
        ing_out = ing.out(6)

        self.edges.extend([a, b, c, mul_out, ing_out])
        self.nodes.extend([mul, ing])


def rw24_(node_types):
    return [r24(node_types)]


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
    print_env(g, r, n, None, viz=False)

    # Test3: substitution at inputs boundary
    n = 3
    print(f"Test{n}: ")
    g = G3(node_types)
    r = rw3(node_types)
    print_env(g, r, n, None, viz=False)

    # Test4: edge cases? e.g. (+ ?a 0) => ?a
    n = 4
    print(f"Test{n}: ")
    g = G4(node_types)
    r = rw4(node_types)
    print_env(g, r, n, None, True)

    n = 4.2
    print(f"Test{n}: ")
    g = G4_v2(node_types)
    r = rw4(node_types)
    print_env(g, r, n, None, True)

    n = 4.3
    print("Test4-cancel-sub: ")
    g = G4_cancel_sub(node_types)
    r = rw4(node_types)
    a = (torch.tensor(1).long(), torch.tensor(0).long())
    print_env(g, r, n, a, viz=True)

    n = 4.4
    print("Test4-cancel-sub_v2: ")
    g = G4_cancel_sub_v2(node_types)
    r = rw4(node_types)
    a = (torch.tensor(1).long(), torch.tensor(0).long())
    print_env(g, r, n, a, viz=True)

    n = 4.5
    print("Test4-cancel-sub_v3: ")
    g = G4_cancel_sub_v3(node_types)
    r = rw4(node_types)
    a = (torch.tensor(1).long(), torch.tensor(0).long())
    print_env(g, r, n, a, viz=True)

    # Test5: edge cases? e.g. (d ?x (sin ?x) )
    n = 5
    print(f"Test{n}: ")
    g = G5(node_types)
    r = rw5(node_types)
    print_env(g, r, n, None, viz=True)

    # Test all: expr rewrite rules
    n = 5.1
    print("sub_canon:")
    g = G5_sub_canon(node_types)
    r = rw5_(node_types)
    print_env(g, r, n, None, viz=True)

    # n = 5.2
    # print(f"Test{n}: ")
    # g = G5_sub_canon_more(node_types)
    # r = rw5_(node_types)
    # print_env(g, r, n, None, viz=False)

    print("zero-add:")
    n = 6.1
    print(f"Test{n}: ")
    g = G6(node_types)
    r = rw6_(node_types)
    print_env(g, r, n, None, viz=True)

    n = 6.2
    g = G6_more(node_types)
    r = rw6_(node_types)
    print_env(g, r, n, None, viz=True)

    n = 7
    print("zero-mul:")
    g = G7(node_types)
    r = rw7_(node_types)
    print_env(g, r, n, None, viz=True)

    n = 9
    print("cancel-sub:")
    g = G9(node_types)
    r = rw9_(node_types)
    print_env(g, r, n, None, viz=True)

    n = 14
    print("pow2:")
    g = G14(node_types)
    r = rw14_(node_types)
    print_env(g, r, n, None, viz=True)

    n = 18
    print("d-cos:")
    g = G18(node_types)
    r = rw18_(node_types)
    print_env(g, r, n, None, viz=True)

    print("i-part:")
    g = G24(node_types)
    r = rw24_(node_types)
    print_env(g, r, 24, None, viz=True)


def main(_):
    test_expr()


if __name__ == "__main__":
    app.run(main)
