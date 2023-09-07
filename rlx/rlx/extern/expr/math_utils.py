import math

from rlx.frontend.registry import get_node_type
from rlx.frontend.registry import register_node_type
from rlx.frontend import RewriteRule, Graph, Node, Edge, node_pattern, const_pattern, symbol_pattern

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
def define_rewrite_rules(node_types):
    class r1(RewriteRule):
        def __init__(self):
            # ["comm-add", op.add(a, b), op.add(b, a)],
            self.name = "comm-add"
            # NOTE: None attr to match any Const
            self.a, self.ta = const_pattern(None)
            self.b, self.tb = const_pattern(None)

        def source_pattern(self):
            add = node_pattern(node_types["Add"], [self.a, self.b], 1)
            return [add]

        def target_pattern(self):
            out = node_pattern(node_types["Add"], [self.tb, self.ta], 1)
            return [out]

    class r2(RewriteRule):
        def __init__(self):
            self.name = "a*b => b*a"
            self.a, self.ta = const_pattern()
            self.b, self.tb = const_pattern()

        def source_pattern(self):
            add = node_pattern(node_types["Mul"], [self.a, self.b], 1)
            return [add]

        def target_pattern(self):
            out = node_pattern(node_types["Mul"], [self.tb, self.ta], 1)
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
            mul = node_pattern(node_types["Mul"], [self.t_minus_one, self.tb],
                               1)
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

    return [
        r1(),
        r2(),
        r3(),
        r4(),
        r5(),
        r6(),
        r7(),
        r8(),
        r9(),
        r10(),
        r11(),
        r12(),
        r13(),
        r14(),
        r15(),
        r16(),
        r17(),
        r18(),
        r19(),
        r20(),
        r21(),
        r22(),
        r23(),
        r24(),
    ]
