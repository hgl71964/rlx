import math

from rlx.frontend.registry import get_node_type
from rlx.frontend.registry import register_node_type
from rlx.frontend import RewriteRule, Graph, Node, Edge, node_pattern, const_pattern, symbol_pattern
from rlx.extern.expr.expr_utils import expr_edge, expr_node

_prop_id = 10000


def get_id():
    global _prop_id
    local = _prop_id
    _prop_id += 1
    return local


NODE_TYPES = [
    "And",
    "Not",
    "Or",
    "Implies",

    # leaf; must define!
    "Var",
    "Const",
]


def define_node_type():
    register_node_type(NODE_TYPES)
    return get_node_type()


#########################################
################ utility ################
#########################################
def reward_func(
    graph: Graph,
    init: bool,
    terminated: bool,
    stats: dict,
) -> float:
    cost = 0
    for e in graph.get_edges():
        cost += len(e.uses)

    if init:
        # print(f"init cost {cost}")
        assert (cost != 0), f"initial reward cannot be zero"
        stats["init_cost"] = cost
        stats["last_cost"] = cost
        return cost

    # print(f"last: {stats['last_cost']}; now {cost}; init: {stats['init_cost']} ")
    reward = (stats["last_cost"] - cost) / stats["init_cost"]
    stats["last_cost"] = cost
    return reward


def verify(expr):
    raise

    def dfs(node):
        # leaf
        if isinstance(node, int):
            return node

        children = []
        for child in node._fields:
            child_node = dfs(getattr(node, str(child)))
            children.append(child_node)

        my_type = type(node).__name__
        if my_type == "Add":
            ret = children[0] + children[1]
        elif my_type == "Sub":
            ret = children[0] - children[1]
        elif my_type == "Mul":
            ret = children[0] * children[1]
        elif my_type == "Div":
            assert (children[1] != 0), "Div by zero"
            ret = children[0] / children[1]
        elif my_type == "Pow":
            assert (children[0] > 0), "Pow by negative"
            try:
                ret = math.pow(children[0], children[1])
            except:
                raise RuntimeError(f"pow: {children[0]}, {children[1]}")
        elif my_type == "Sqrt":
            assert (children[0] >= 0)
            ret = math.sqrt(children[0])
        elif my_type == "Sin":
            ret = math.sin(children[0])
        elif my_type == "Cos":
            ret = math.cos(children[0])
        else:
            raise RuntimeError(f"Unsupport Op {my_type}")
        return ret

    return dfs(expr)


def rlxGraph2Prop(ops, edges: list[Edge]):
    # find outputs
    outputs = []
    for e in edges:
        if len(e.uses) == 0:
            outputs.append(e)

    assert len(outputs) == 1, f"expect 1 output, got {len(outputs)}"
    output = outputs[0]

    def lookup(node_type):
        for op in ops:
            if op.__name__ == node_type.name:
                return op
        raise RuntimeError(f"Unsupport node type {node_type.name}")

    # build expr
    built = {}  # obj -> expr obj

    def dfs(obj):
        if obj in built:
            return built[obj]

        if isinstance(obj, Node):
            inputs = []
            for inp in obj.inputs:
                out = dfs(inp)
                inputs.append(out)

            op = lookup(obj.get_type())
            node = op(*inputs)
            built[obj] = node
            return node

        if isinstance(obj, Edge):
            if obj.trace is None:
                # input (must be const)
                assert obj.attr is not None, f"expect int, got None {obj.idx}"
                assert isinstance(
                    obj.attr,
                    int), f"expect int, got {type(obj.attr)} {obj.idx}"
                # convert back to bool
                v = bool(obj.attr)
                built[obj] = v
                return v
            else:
                node = dfs(obj.trace)
                built[obj] = node
                return node

        raise RuntimeError("unreachable")

    return dfs(output)


#########################################
########### rewrite rules! ##############
#########################################
class r1(RewriteRule):

    def __init__(self, node_types):
        self.name = "def_imply"
        self.node_types = node_types
        self.a = const_pattern(None)
        self.b = const_pattern(None)

    def source_pattern(self):
        imp = node_pattern(self.node_types["Implies"], [self.a, self.b], 1)
        return [imp]

    def target_pattern(self, matched):
        # out = node_pattern(self.node_types["Not"], [self.ta], 1)
        # out2 = node_pattern(self.node_types["Or"], [out, self.tb], 1)

        a, b = [matched[pat] for pat in [self.a, self.b]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Not"],
                        inputs=[a]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[out, b]).out(get_id())
        return [out]


class r2(RewriteRule):

    def __init__(self, node_types):
        self.name = "double_neg"
        self.node_types = node_types
        self.a = const_pattern()

    def source_pattern(self):
        o1 = node_pattern(self.node_types["Not"], [self.a], 1)
        o2 = node_pattern(self.node_types["Not"], [o1], 1)
        return [o2]

    def target_pattern(self, matched):
        a = matched[self.a]
        return [a]


class r3(RewriteRule):

    def __init__(self, node_types):
        self.name = "def_imply_flip"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()

    def source_pattern(self):
        out = node_pattern(self.node_types["Not"], [self.a], 1)
        out2 = node_pattern(self.node_types["Or"], [out, self.b], 1)
        return [out2]

    def target_pattern(self, matched):
        # o = node_pattern(self.node_types["Implies"], [self.ta, self.tb], 1)
        a, b = [matched[pat] for pat in [self.a, self.b]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Implies"],
                        inputs=[a, b]).out(get_id())
        return [out]


class r4(RewriteRule):

    def __init__(self, node_types):
        self.name = "double_neg_flip"
        self.node_types = node_types
        self.a = const_pattern()

    def source_pattern(self):
        return [self.a]

    def target_pattern(self, matched):
        a = matched[self.a]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[a]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[out]).out(get_id())
        return [out]


class r5(RewriteRule):

    def __init__(self, node_types):
        self.name = "assoc_or"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()

    def source_pattern(self):
        or1 = node_pattern(self.node_types["Or"], [self.b, self.c], 1)
        or2 = node_pattern(self.node_types["Or"], [self.a, or1], 1)
        return [or2]

    def target_pattern(self, matched):
        # or1 = node_pattern(self.node_types["Or"], [self.ta, self.tb], 1)
        # or2 = node_pattern(self.node_types["Or"], [or1, self.tc], 1)
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[a, b]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[out, c]).out(get_id())
        return [out]


class r6(RewriteRule):

    def __init__(self, node_types):
        self.name = "dist_and_or"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()

    def source_pattern(self):
        or1 = node_pattern(self.node_types["Or"], [self.b, self.c], 1)
        and1 = node_pattern(self.node_types["And"], [self.a, or1], 1)
        return [and1]

    def target_pattern(self, matched):
        # and1 = node_pattern(self.node_types["And"], [self.ta, self.tb], 1)
        # and2 = node_pattern(self.node_types["And"], [self.ta, self.tc], 1)
        # or1 = node_pattern(self.node_types["Or"], [and1, and2], 1)
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["And"],
                        inputs=[a, b]).out(get_id())
        out2 = expr_node(get_id(),
                         attr=None,
                         node_type=self.node_types["And"],
                         inputs=[a, c]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[out, out2]).out(get_id())
        return [out]


class r7(RewriteRule):

    def __init__(self, node_types):
        self.name = "dist_or_and"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()

    def source_pattern(self):
        and1 = node_pattern(self.node_types["And"], [self.b, self.c], 1)
        or1 = node_pattern(self.node_types["Or"], [self.a, and1], 1)
        return [or1]

    def target_pattern(self, matched):
        # or1 = node_pattern(self.node_types["Or"], [self.ta, self.tb], 1)
        # or2 = node_pattern(self.node_types["Or"], [self.ta, self.tc], 1)
        # and1 = node_pattern(self.node_types["And"], [or1, or2], 1)
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[a, b]).out(get_id())
        out2 = expr_node(get_id(),
                         attr=None,
                         node_type=self.node_types["Or"],
                         inputs=[a, c]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["And"],
                        inputs=[out, out2]).out(get_id())
        return [out]


class r8(RewriteRule):

    def __init__(self, node_types):
        self.name = "comm_or"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()

    def source_pattern(self):
        or1 = node_pattern(self.node_types["Or"], [self.a, self.b], 1)
        return [or1]

    def target_pattern(self, matched):
        # or1 = node_pattern(self.node_types["Or"], [self.tb, self.ta], 1)
        a, b = [matched[pat] for pat in [self.a, self.b]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[b, a]).out(get_id())
        return [out]


class r9(RewriteRule):

    def __init__(self, node_types):
        self.name = "comm_and"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()

    def source_pattern(self):
        out = node_pattern(self.node_types["And"], [self.a, self.b], 1)
        return [out]

    def target_pattern(self, matched):
        # out = node_pattern(self.node_types["And"], [self.tb, self.ta], 1)
        a, b = [matched[pat] for pat in [self.a, self.b]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["And"],
                        inputs=[b, a]).out(get_id())
        return [out]


class r10(RewriteRule):

    def __init__(self, node_types):
        self.name = "lem"
        self.node_types = node_types
        self.a = const_pattern()

    def source_pattern(self):
        not1 = node_pattern(self.node_types["Not"], [self.a], 1)
        or1 = node_pattern(self.node_types["Or"], [self.a, not1], 1)
        return [or1]

    def target_pattern(self, matched):
        out = expr_edge(
            get_id(),
            attr=1,  # True
            edge_type=self.node_types["Const"],
            trace=None)
        return [out]


class r11(RewriteRule):

    def __init__(self, node_types):
        self.name = "or_true"
        self.node_types = node_types
        self.a = const_pattern()
        self.true = const_pattern(attr=1)

    def source_pattern(self):
        or1 = node_pattern(self.node_types["Or"], [self.a, self.true], 1)
        return [or1]

    def target_pattern(self, matched):
        out = expr_edge(
            get_id(),
            attr=1,  # True
            edge_type=self.node_types["Const"],
            trace=None)
        return [out]


class r12(RewriteRule):

    def __init__(self, node_types):
        self.name = "and_true"
        self.node_types = node_types
        self.a = const_pattern()
        self.true = const_pattern(attr=1)

    def source_pattern(self):
        and1 = node_pattern(self.node_types["And"], [self.a, self.true], 1)
        return [and1]

    def target_pattern(self, matched):
        a = matched[self.a]
        return [a]


class r13(RewriteRule):

    def __init__(self, node_types):
        self.name = "contrapositive"
        self.node_types = node_types
        self.a = const_pattern()
        self.b = const_pattern()

    def source_pattern(self):
        o1 = node_pattern(self.node_types["Implies"], [self.a, self.b], 1)
        return [o1]

    def target_pattern(self, matched):
        # not1 = node_pattern(self.node_types["Not"], [self.ta], 1)
        # not2 = node_pattern(self.node_types["Not"], [self.tb], 1)
        # or1 = node_pattern(self.node_types["Or"], [not1, not2], 1)
        a, b = [matched[pat] for pat in [self.a, self.b]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Not"],
                        inputs=[a]).out(get_id())
        out2 = expr_node(get_id(),
                         attr=None,
                         node_type=self.node_types["Not"],
                         inputs=[b]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Or"],
                        inputs=[out, out2]).out(get_id())
        return [out]


def define_rewrite_rules(node_types):

    return [
        r1(node_types),
        r2(node_types),
        r3(node_types),
        # r4(node_types),
        r5(node_types),
        r6(node_types),
        r7(node_types),
        r8(node_types),
        r9(node_types),
        r10(node_types),
        r11(node_types),
        r12(node_types),
        r13(node_types),
    ]
