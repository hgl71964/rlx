import math

from rlx.frontend.registry import get_node_type
from rlx.frontend.registry import register_node_type
from rlx.frontend import RewriteRule, Graph, Node, Edge, node_pattern, const_pattern, symbol_pattern, EdgePattern, NodePattern

from rlx.extern.expr.expr_utils import expr_edge, expr_node

_math_id = 1000  # for substituted obj id, for debugging


def get_id():
    global _math_id
    local = _math_id
    _math_id += 1
    return local


NODE_TYPES = [
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
def cnt_op_cost(expr):
    raise
    # m = {
    #     "Add": 2,
    #     "Sub": 2,
    #     "Mul": 4,
    #     "Div": 4,
    #     "Pow": 6,
    #     "Sqrt": 2,
    #     "Sin": 2,
    #     "Cos": 2,
    # }
    # cnt = 0

    # def dfs(node):
    #     nonlocal cnt
    #     if isinstance(node, int):
    #         return

    #     my_type = type(node).__name__
    #     cnt += m[my_type]
    #     for child in node._fields:
    #         dfs(getattr(node, str(child)))

    # dfs(expr)
    return cnt


def rlxGraph2math(ops, edges: list[Edge]):
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
                built[obj] = obj.attr
                return obj.attr
            else:
                node = dfs(obj.trace)
                built[obj] = node
                return node

        raise RuntimeError("unreachable")

    return dfs(output)


def verify(expr):
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


#########################################
########### rewrite rules! ##############
#########################################
class r1(RewriteRule):
    def __init__(self, node_types):
        # ["comm-add", op.add(a, b), op.add(b, a)],
        self.name = "comm-add"
        self.a = const_pattern()
        self.b = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["Add"], [self.a, self.b], 1)
        return [add]

    def target_pattern(self, matched: dict[EdgePattern, Edge]):
        a, b = [matched[pat] for pat in [self.a, self.b]]
        new = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Add"],
                        inputs=[b, a])
        out = new.out(get_id())
        return [out]


class r2(RewriteRule):
    def __init__(self, node_types):
        self.name = "a*b => b*a"
        self.a = const_pattern()
        self.b = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["Mul"], [self.a, self.b], 1)
        return [add]

    def target_pattern(self, matched):
        a, b = [matched[pat] for pat in [self.a, self.b]]
        new = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Mul"],
                        inputs=[b, a])
        out = new.out(get_id())
        return [out]


class r3(RewriteRule):
    def __init__(self, node_types):
        # ["assoc-add", op.add(op.add(a, b), c), op.add(a, op.add(b, c))],
        self.name = "assoc-add"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add1 = node_pattern(self.node_types["Add"], [self.a, self.b], 1)
        add2 = node_pattern(self.node_types["Add"], [add1, self.c], 1)
        return [add2]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Add"],
                        inputs=[b, c]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Add"],
                        inputs=[a, out]).out(get_id())
        return [out]


class r4(RewriteRule):
    def __init__(self, node_types):
        # ["assoc-mul", op.mul(op.mul(a, b), c), op.mul(a, op.mul(b, c))],
        self.name = "assoc-mul"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        mul1 = node_pattern(self.node_types["Mul"], [self.a, self.b], 1)
        mul2 = node_pattern(self.node_types["Mul"], [mul1, self.c], 1)
        return [mul2]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Mul"],
                        inputs=[b, c]).out(get_id())
        out = expr_node(get_id(),
                        attr=None,
                        node_type=self.node_types["Mul"],
                        inputs=[a, out]).out(get_id())
        return [out]


class r5(RewriteRule):
    def __init__(self, node_types):
        # ["sub-canon", op.sub(a, b), op.add(a, op.mul(-1, b))],
        self.name = "sub-canon"
        self.a = const_pattern()
        self.b = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        sub = node_pattern(self.node_types["Sub"], [self.a, self.b], 1)
        return [sub]

    def target_pattern(self, matched):
        a, b = [matched[pat] for pat in [self.a, self.b]]
        out = expr_edge(
            get_id(),
            attr=-1,
            edge_type=self.node_types["Const"],
            trace=None,
        )
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[out, b],
        ).out(get_id())
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[a, out],
        ).out(get_id())
        return [out]


class r6(RewriteRule):
    def __init__(self, node_types):
        # ["zero-add", op.add(a, 0), a],
        self.name = "zero-add"
        self.a = const_pattern()
        self.zero = const_pattern(attr=0)
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["Add"], [self.a, self.zero], 1)
        return [add]

    def target_pattern(self, matched):
        a = matched[self.a]
        return [a]


class r7(RewriteRule):
    def __init__(self, node_types):
        # ["zero-mul", op.mul(a, 0), 0],
        self.name = "zero-mul"
        self.a = const_pattern()
        self.zero = const_pattern(attr=0)
        self.node_types = node_types

    def source_pattern(self):
        mul = node_pattern(self.node_types["Mul"], [self.a, self.zero], 1)
        return [mul]

    def target_pattern(self, matched):
        out = expr_edge(
            get_id(),
            attr=0,
            edge_type=self.node_types["Const"],
            trace=None,
        )
        return [out]


class r8(RewriteRule):
    def __init__(self, node_types):
        # ["one-mul", op.mul(a, 1), a],
        self.name = "one-mul"
        self.a = const_pattern()
        self.one = const_pattern(attr=1)
        self.node_types = node_types

    def source_pattern(self):
        mul = node_pattern(self.node_types["Mul"], [self.a, self.one], 1)
        return [mul]

    def target_pattern(self, matched):
        a = matched[self.a]
        return [a]


class r9(RewriteRule):
    def __init__(self, node_types):
        # ["cancel-sub", op.sub(a, a), 0],
        self.name = "cancel-sub"
        self.a = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        out = node_pattern(self.node_types["Sub"], [self.a, self.a], 1)
        return [out]

    def target_pattern(self, matched):
        out = expr_edge(
            get_id(),
            attr=0,
            edge_type=self.node_types["Const"],
            trace=None,
        )
        return [out]


class r10(RewriteRule):
    def __init__(self, node_types):
        # [ "distribute", op.mul(a, op.add(b, c)), op.add(op.mul(a, b), op.mul(a, c))],
        self.name = "distribute"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["Add"], [self.b, self.c], 1)
        mul = node_pattern(self.node_types["Mul"], [self.a, add], 1)
        return [mul]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[a, b],
        ).out(get_id())
        out2 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[a, c],
        ).out(get_id())
        out3 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[out, out2],
        ).out(get_id())
        return [out3]


class r11(RewriteRule):
    def __init__(self, node_types):
        # [ "factor", op.add(op.mul(a, b), op.mul(a, c)), op.mul(a, op.add(b, c)) ],
        self.name = "factor"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        mul1 = node_pattern(self.node_types["Mul"], [self.b, self.a], 1)
        mul2 = node_pattern(self.node_types["Mul"], [self.c, self.a], 1)
        add = node_pattern(self.node_types["Add"], [mul1, mul2], 1)
        return [add]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[b, c],
        ).out(get_id())
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[a, out],
        ).out(get_id())
        return [out]


class r12(RewriteRule):
    def __init__(self, node_types):
        # [ "pow-mul", op.mul(op.pow(a, b), op.pow(a, c)), op.pow(a, op.add(b, c)) ],
        self.name = "pow-mul"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        pow1 = node_pattern(self.node_types["Pow"], [self.a, self.b], 1)
        pow2 = node_pattern(self.node_types["Pow"], [self.a, self.c], 1)
        mul = node_pattern(self.node_types["Mul"], [pow1, pow2], 1)
        return [mul]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[b, c],
        ).out(get_id())
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Pow"],
            inputs=[a, out],
        ).out(get_id())
        return [out]


class r13(RewriteRule):
    def __init__(self, node_types):
        # ["pow1", op.pow(x, 1), x],
        self.name = "pow1"
        self.a = const_pattern()
        self.one = const_pattern(attr=1)
        self.node_types = node_types

    def source_pattern(self):
        pow1 = node_pattern(self.node_types["Pow"], [self.a, self.one], 1)
        return [pow1]

    def target_pattern(self, matched):
        a = matched[self.a]
        return [a]


class r14(RewriteRule):
    def __init__(self, node_types):
        # ["pow2", op.pow(x, 2), op.mul(x, x)],
        self.name = "pow2"
        self.a = const_pattern()
        self.two = const_pattern(attr=2)
        self.node_types = node_types

    def source_pattern(self):
        pow1 = node_pattern(self.node_types["Pow"], [self.a, self.two], 1)
        return [pow1]

    def target_pattern(self, matched):
        a, _ = [matched[pat] for pat in [self.a, self.two]]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[a, a],
        ).out(get_id())
        return [out]


class r15(RewriteRule):
    def __init__(self, node_types):
        # ["d-add", op.diff(x, op.add(a, b)), op.add(op.diff(x, a), op.diff(x, b))],
        self.name = "d-add"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["Add"], [self.a, self.b], 1)
        diff = node_pattern(self.node_types["Diff"], [self.c, add], 1)
        return [diff]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Diff"],
            inputs=[c, a],
        ).out(get_id())
        out2 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Diff"],
            inputs=[c, b],
        ).out(get_id())
        out3 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[out, out2],
        ).out(get_id())
        return [out3]


class r16(RewriteRule):
    def __init__(self, node_types):
        # ["d-mul", op.diff(x, op.mul(a, b)), op.add(op.mul(a, op.diff(x, b)), op.mul(b, op.diff(x, a)))],
        self.name = "d-mul"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        mul = node_pattern(self.node_types["Mul"], [self.a, self.b], 1)
        diff = node_pattern(self.node_types["Diff"], [self.c, mul], 1)
        return [diff]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        diff_a = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Diff"],
            inputs=[c, a],
        ).out(get_id())
        mul_a = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[b, diff_a],
        ).out(get_id())
        diff_b = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Diff"],
            inputs=[c, b],
        ).out(get_id())
        mul_b = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[a, diff_b],
        ).out(get_id())
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[mul_a, mul_b],
        ).out(get_id())
        return [out]


class r17(RewriteRule):
    def __init__(self, node_types):
        # ["d-sin", op.diff(x, op.sin(x)), op.cos(x)],
        self.name = "d-sin"
        self.a = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        sin = node_pattern(self.node_types["Sin"], [self.a], 1)
        diff = node_pattern(self.node_types["Diff"], [self.a, sin], 1)
        return [diff]

    def target_pattern(self, matched):
        a = matched[self.a]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Cos"],
            inputs=[a],
        ).out(get_id())
        return [out]


class r18(RewriteRule):
    def __init__(self, node_types):
        # ["d-cos", op.diff(x, op.cos(x)), op.mul(-1, op.sin(x))],
        self.name = "d-cos"
        self.a = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        cos = node_pattern(self.node_types["Cos"], [self.a], 1)
        diff = node_pattern(self.node_types["Diff"], [self.a, cos], 1)
        return [diff]

    def target_pattern(self, matched):
        a = matched[self.a]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Sin"],
            inputs=[a],
        ).out(get_id())
        minus_one = expr_edge(
            get_id(),
            attr=-1,
            edge_type=self.node_types["Const"],
            trace=None,
        )
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[minus_one, out],
        ).out(get_id())
        return [out]


class r19(RewriteRule):
    def __init__(self, node_types):
        # ["i-one", op.integral(1, x), x],
        self.name = "i-one"
        self.a = const_pattern()
        self.one = const_pattern(1)
        self.node_types = node_types

    def source_pattern(self):
        ing = node_pattern(self.node_types["Integral"], [self.one, self.a], 1)
        return [ing]

    def target_pattern(self, matched):
        a = matched[self.a]
        return [a]


class r20(RewriteRule):
    def __init__(self, node_types):
        # ["i-cos", op.integral(op.cos(x), x), op.sin(x)],
        self.name = "i-cos"
        self.a = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        cos = node_pattern(self.node_types["Cos"], [self.a], 1)
        ing = node_pattern(self.node_types["Integral"], [cos, self.a], 1)
        return [ing]

    def target_pattern(self, matched):
        a = matched[self.a]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Sin"],
            inputs=[a],
        ).out(get_id())
        return [out]


class r21(RewriteRule):
    def __init__(self, node_types):
        # ["i-sin", op.integral(op.sin(x), x), op.mul(-1, op.cos(x))],
        self.name = "i-sin"
        self.a = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        sin = node_pattern(self.node_types["Sin"], [self.a], 1)
        ing = node_pattern(self.node_types["Integral"], [sin, self.a], 1)
        return [ing]

    def target_pattern(self, matched):
        a = matched[self.a]
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Cos"],
            inputs=[a],
        ).out(get_id())
        minus_one = expr_edge(
            get_id(),
            attr=-1,
            edge_type=self.node_types["Const"],
            trace=None,
        )
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[minus_one, out],
        ).out(get_id())
        return [out]


class r22(RewriteRule):
    def __init__(self, node_types):
        # ["i-sum", op.integral(op.add(f, g), x), op.add(op.integral(f, x), op.integral(g, x))],
        self.name = "i-sum"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["Add"], [self.a, self.b], 1)
        ing = node_pattern(self.node_types["Integral"], [add, self.c], 1)
        return [ing]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        ing1 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[a, c],
        ).out(get_id())
        ing2 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[b, c],
        ).out(get_id())
        out = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Add"],
            inputs=[ing1, ing2],
        ).out(get_id())
        return [out]


class r23(RewriteRule):
    def __init__(self, node_types):
        # ["i-dif", op.integral(op.sub(f, g), x), op.sub(op.integral(f, x), op.integral(g, x))],
        self.name = "i-dif"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        sub = node_pattern(self.node_types["Sub"], [self.a, self.b], 1)
        ing = node_pattern(self.node_types["Integral"], [sub, self.c], 1)
        return [ing]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        ing1 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[a, c],
        ).out(get_id())
        ing2 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[b, c],
        ).out(get_id())
        sub = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Sub"],
            inputs=[ing1, ing2],
        ).out(get_id())
        return [sub]


class r24(RewriteRule):
    def __init__(self, node_types):
        # ["i-parts", op.integral(op.mul(a, b), x),
        # op.sub(op.mul(a, op.integral(b, x)), op.integral(op.mul(op.diff(x, a), op.integral(b, x)), x))],
        self.name = "i-parts"
        self.a = const_pattern()
        self.b = const_pattern()
        self.c = const_pattern()
        self.node_types = node_types

    def source_pattern(self):
        mul = node_pattern(self.node_types["Mul"], [self.a, self.b], 1)
        ing = node_pattern(self.node_types["Integral"], [mul, self.c], 1)
        return [ing]

    def target_pattern(self, matched):
        a, b, c = [matched[pat] for pat in [self.a, self.b, self.c]]
        # first part
        ing1 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[b, c],
        ).out(get_id())
        mul1 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[a, ing1],
        ).out(get_id())

        # second part
        diff = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Diff"],
            inputs=[c, a],
        ).out(get_id())
        ing2 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[b, c],
        ).out(get_id())
        mul2 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Mul"],
            inputs=[diff, ing2],
        ).out(get_id())
        ing3 = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Integral"],
            inputs=[mul2, c],
        ).out(get_id())
        sub = expr_node(
            get_id(),
            attr=None,
            node_type=self.node_types["Sub"],
            inputs=[mul1, ing3],
        ).out(get_id())
        return [sub]

    #["add-zero", a, op.add(a, 0)],
    #["mul-one", a, op.mul(a, 1)],


def define_rewrite_rules(node_types):
    return [
        r1(node_types),
        r2(node_types),
        r3(node_types),
        r4(node_types),
        r5(node_types),
        r6(node_types),
        r7(node_types),
        r8(node_types),
        r9(node_types),
        r10(node_types),
        r11(node_types),
        r12(node_types),
        r13(node_types),
        r14(node_types),
        r15(node_types),
        r16(node_types),
        r17(node_types),
        r18(node_types),
        r19(node_types),
        r20(node_types),
        r21(node_types),
        r22(node_types),
        r23(node_types),
        r24(node_types),
    ]
