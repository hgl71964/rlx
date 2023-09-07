import math

from rlx.frontend.registry import get_node_type
from rlx.frontend.registry import register_node_type
from rlx.frontend import RewriteRule, Graph, Node, Edge, node_pattern, const_pattern, symbol_pattern

from rlx.extern.expr.expr_utils import expr_edge, expr_node, out

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

    def target_pattern(self, matched):
        a, b = [matched[idx] for idx in [self.ta, self.tb]]
        new = expr_node(-1,
                        attr=None,
                        node_type=self.node_types["Add"],
                        inputs=[b, a])
        out = out(new)
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
        a, b = [matched[idx] for idx in [self.ta, self.tb]]
        new = expr_node(-1,
                        attr=None,
                        node_type=self.node_types["Mul"],
                        inputs=[b, a])
        out = out(new)
        return [out]


class r3(RewriteRule):
    def __init__(self):
        # ["assoc-add", op.add(op.add(a, b), c), op.add(a, op.add(b, c))],
        self.name = "assoc-add"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        add1 = node_pattern(node_types["Add"], [self.a, self.b], 1)
        add2 = node_pattern(node_types["Add"], [add1, self.c], 1)
        return [add2]

    def target_pattern(self):
        add1 = node_pattern(node_types["Add"], [self.tb, self.tc], 1)
        add2 = node_pattern(node_types["Add"], [self.ta, add1], 1)
        return [add2]


class r4(RewriteRule):
    def __init__(self):
        # ["assoc-mul", op.mul(op.mul(a, b), c), op.mul(a, op.mul(b, c))],
        self.name = "assoc-mul"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        mul1 = node_pattern(node_types["Mul"], [self.a, self.b], 1)
        mul2 = node_pattern(node_types["Mul"], [mul1, self.c], 1)
        return [mul2]

    def target_pattern(self):
        mul1 = node_pattern(node_types["Mul"], [self.tb, self.tc], 1)
        mul2 = node_pattern(node_types["Mul"], [self.ta, mul1], 1)
        return [mul2]


class r5(RewriteRule):
    def __init__(self):
        # ["sub-canon", op.sub(a, b), op.add(a, op.mul(-1, b))],
        self.name = "sub-canon"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        _, self.t_minus_one = const_pattern(attr=-1)

    def source_pattern(self):
        sub = node_pattern(node_types["Sub"], [self.a, self.b], 1)
        return [sub]

    def target_pattern(self):
        mul = node_pattern(node_types["Mul"], [self.t_minus_one, self.tb], 1)
        add = node_pattern(node_types["Add"], [self.ta, mul], 1)
        return [add]


class r6(RewriteRule):
    def __init__(self):
        # ["zero-add", op.add(a, 0), a],
        self.name = "zero-add"
        self.a, self.ta = const_pattern()
        self.zero, _ = const_pattern(attr=0)

    def source_pattern(self):
        add = node_pattern(node_types["Add"], [self.a, self.zero], 1)
        return [add]

    def target_pattern(self):
        return [self.ta]


class r7(RewriteRule):
    def __init__(self):
        # ["zero-mul", op.mul(a, 0), 0],
        self.name = "zero-mul"
        self.a, _ = const_pattern()
        self.zero, self.tzero = const_pattern(attr=0)

    def source_pattern(self):
        mul = node_pattern(node_types["Mul"], [self.a, self.zero], 1)
        return [mul]

    def target_pattern(self):
        return [self.tzero]


class r8(RewriteRule):
    def __init__(self):
        # ["one-mul", op.mul(a, 1), a],
        self.name = "one-mul"
        self.a, self.ta = const_pattern()
        self.one, _ = const_pattern(attr=1)

    def source_pattern(self):
        mul = node_pattern(node_types["Mul"], [self.a, self.one], 1)
        return [mul]

    def target_pattern(self):
        return [self.ta]


class r9(RewriteRule):
    def __init__(self):
        # ["cancel-sub", op.sub(a, a), 0],
        self.name = "cancel-sub"
        self.a, self.ta = const_pattern()
        _, self.tzero = const_pattern(attr=0)

    def source_pattern(self):
        out = node_pattern(node_types["Sub"], [self.a, self.a], 1)
        return [out]

    def target_pattern(self):
        return [self.tzero]


class r10(RewriteRule):
    def __init__(self):
        # [ "distribute", op.mul(a, op.add(b, c)), op.add(op.mul(a, b), op.mul(a, c))],
        self.name = "distribute"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        add = node_pattern(node_types["Add"], [self.b, self.c], 1)
        mul = node_pattern(node_types["Mul"], [self.a, add], 1)
        return [mul]

    def target_pattern(self):
        mul1 = node_pattern(node_types["Mul"], [self.ta, self.tb], 1)
        mul2 = node_pattern(node_types["Mul"], [self.ta, self.tc], 1)
        add = node_pattern(node_types["Add"], [mul1, mul2], 1)
        return [add]


class r11(RewriteRule):
    def __init__(self):
        # [ "factor", op.add(op.mul(a, b), op.mul(a, c)), op.mul(a, op.add(b, c)) ],
        self.name = "factor"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        mul1 = node_pattern(node_types["Mul"], [self.a, self.b], 1)
        mul2 = node_pattern(node_types["Mul"], [self.a, self.c], 1)
        add = node_pattern(node_types["Add"], [mul1, mul2], 1)
        return [add]

    def target_pattern(self):
        add = node_pattern(node_types["Add"], [self.tb, self.tc], 1)
        mul = node_pattern(node_types["Mul"], [self.ta, add], 1)
        return [mul]


class r12(RewriteRule):
    def __init__(self):
        # [ "pow-mul", op.mul(op.pow(a, b), op.pow(a, c)), op.pow(a, op.add(b, c)) ],
        self.name = "pow-mul"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        pow1 = node_pattern(node_types["Pow"], [self.a, self.b], 1)
        pow2 = node_pattern(node_types["Pow"], [self.a, self.c], 1)
        mul = node_pattern(node_types["Mul"], [pow1, pow2], 1)
        return [mul]

    def target_pattern(self):
        add = node_pattern(node_types["Add"], [self.tb, self.tc], 1)
        pow1 = node_pattern(node_types["Pow"], [self.ta, add], 1)
        return [pow1]


class r13(RewriteRule):
    def __init__(self):
        # ["pow1", op.pow(x, 1), x],
        self.name = "pow1"
        self.a, self.ta = const_pattern()
        self.one, _ = const_pattern(attr=1)

    def source_pattern(self):
        pow1 = node_pattern(node_types["Pow"], [self.a, self.one], 1)
        return [pow1]

    def target_pattern(self):
        return [self.ta]


class r14(RewriteRule):
    def __init__(self):
        # ["pow2", op.pow(x, 2), op.mul(x, x)],
        self.name = "pow2"
        self.a, self.ta = const_pattern()
        self.two, _ = const_pattern(attr=2)

    def source_pattern(self):
        pow1 = node_pattern(node_types["Pow"], [self.a, self.two], 1)
        return [pow1]

    def target_pattern(self):
        mul = node_pattern(node_types["Mul"], [self.ta, self.ta], 1)
        return [mul]


class r15(RewriteRule):
    def __init__(self):
        # ["d-add", op.diff(x, op.add(a, b)), op.add(op.diff(x, a), op.diff(x, b)) ],
        self.name = "d-add"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        add = node_pattern(node_types["Add"], [self.a, self.b], 1)
        diff = node_pattern(node_types["Diff"], [self.c, add], 1)
        return [diff]

    def target_pattern(self):
        diff1 = node_pattern(node_types["Diff"], [self.tc, self.ta], 1)
        diff2 = node_pattern(node_types["Diff"], [self.tc, self.tb], 1)
        add = node_pattern(node_types["Add"], [diff1, diff2], 1)
        return [add]


class r16(RewriteRule):
    def __init__(self):
        # ["d-mul", op.diff(x, op.mul(a, b)), op.add(op.mul(a, op.diff(x, b)), op.mul(b, op.diff(x, a)))],
        self.name = "d-mul"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        mul = node_pattern(node_types["Mul"], [self.a, self.b], 1)
        diff = node_pattern(node_types["Diff"], [self.c, mul], 1)
        return [diff]

    def target_pattern(self):
        diff_a = node_pattern(node_types["Diff"], [self.tc, self.ta], 1)
        diff_b = node_pattern(node_types["Diff"], [self.tc, self.tb], 1)
        mul_a = node_pattern(node_types["Mul"], [self.tb, diff_a], 1)
        mul_b = node_pattern(node_types["Mul"], [self.ta, diff_b], 1)
        add = node_pattern(node_types["Add"], [mul_a, mul_b], 1)
        return [add]


class r17(RewriteRule):
    def __init__(self):
        # ["d-sin", op.diff(x, op.sin(x)), op.cos(x)],
        self.name = "d-sin"
        self.a, self.ta = const_pattern()

    def source_pattern(self):
        sin = node_pattern(node_types["Sin"], [self.a], 1)
        diff = node_pattern(node_types["Diff"], [self.a, sin], 1)
        return [diff]

    def target_pattern(self):
        cos = node_pattern(node_types["Cos"], [self.ta], 1)
        return [cos]


class r18(RewriteRule):
    def __init__(self):
        # ["d-cos", op.diff(x, op.cos(x)), op.mul(-1, op.sin(x))],
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


class r19(RewriteRule):
    def __init__(self):
        # ["i-one", op.integral(1, x), x],
        self.name = "i-one"
        self.a, self.ta = const_pattern()
        self.one, _ = const_pattern(1)

    def source_pattern(self):
        ing = node_pattern(node_types["Integral"], [self.one, self.a], 1)
        return [ing]

    def target_pattern(self):
        return [self.ta]


class r20(RewriteRule):
    def __init__(self):
        # ["i-cos", op.integral(op.cos(x), x), op.sin(x)],
        self.name = "i-cos"
        self.a, self.ta = const_pattern()

    def source_pattern(self):
        cos = node_pattern(node_types["Cos"], [self.a], 1)
        ing = node_pattern(node_types["Integral"], [cos, self.a], 1)
        return [ing]

    def target_pattern(self):
        sin = node_pattern(node_types["Sin"], [self.ta], 1)
        return [sin]


class r21(RewriteRule):
    def __init__(self):
        # ["i-sin", op.integral(op.sin(x), x), op.mul(-1, op.cos(x))],
        self.name = "i-sin"
        self.a, self.ta = const_pattern()
        _, self.tmone = const_pattern(1)

    def source_pattern(self):
        sin = node_pattern(node_types["Sin"], [self.a], 1)
        ing = node_pattern(node_types["Integral"], [sin, self.a], 1)
        return [ing]

    def target_pattern(self):
        cos = node_pattern(node_types["Cos"], [self.ta], 1)
        mul = node_pattern(node_types["Mul"], [self.tmone, cos], 1)
        return [mul]


class r22(RewriteRule):
    def __init__(self):
        # ["i-sum", op.integral(op.add(f, g), x), op.add(op.integral(f, x), op.integral(g, x))],
        self.name = "i-sum"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        add = node_pattern(node_types["Add"], [self.a, self.b], 1)
        ing = node_pattern(node_types["Integral"], [add, self.c], 1)
        return [ing]

    def target_pattern(self):
        ing1 = node_pattern(node_types["Integral"], [self.ta, self.tc], 1)
        ing2 = node_pattern(node_types["Integral"], [self.tb, self.tc], 1)
        add = node_pattern(node_types["Add"], [ing1, ing2], 1)
        return [add]


class r23(RewriteRule):
    def __init__(self):
        # ["i-dif", op.integral(op.sub(f, g), x), op.sub(op.integral(f, x), op.integral(g, x))],
        self.name = "i-dif"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.c, self.tc = const_pattern()

    def source_pattern(self):
        sub = node_pattern(node_types["Sub"], [self.a, self.b], 1)
        ing = node_pattern(node_types["Integral"], [sub, self.c], 1)
        return [ing]

    def target_pattern(self):
        ing1 = node_pattern(node_types["Integral"], [self.ta, self.tc], 1)
        ing2 = node_pattern(node_types["Integral"], [self.tb, self.tc], 1)
        sub = node_pattern(node_types["Sub"], [ing1, ing2], 1)
        return [sub]


class r24(RewriteRule):
    def __init__(self):
        # ["i-parts", op.integral(op.mul(a, b), x),
        # op.sub(op.mul(a, op.integral(b, x)), op.integral(op.mul(op.diff(x, a), op.integral(b, x)), x))],
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

    #["add-zero", a, op.add(a, 0)],
    #["mul-one", a, op.mul(a, 1)],


def define_rewrite_rules(node_types):
    return [
        r1(node_types),
        r2(node_types),
        # r3(node_types),
        # r4(node_types),
        # r5(node_types),
        # r6(node_types),
        # r7(node_types),
        # r8(node_types),
        # r9(node_types),
        # r10(node_types),
        # r11(node_types),
        # r12(node_types),
        # r13(node_types),
        # r14(node_types),
        # r15(node_types),
        # r16(node_types),
        # r17(node_types),
        # r18(node_types),
        # r19(node_types),
        # r20(node_types),
        # r21(node_types),
        # r22(node_types),
        # r23(node_types),
        # r24(node_types),
    ]
