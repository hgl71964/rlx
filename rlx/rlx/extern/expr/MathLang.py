import sys
import random
import functools
import numpy as np
# import torch

from rlx.extern.expr.lib import Language, TestExprs

from rust_lib import vars

# needed for safe expression generation
sys.setrecursionlimit(10**5)


class MathLang(Language):
    """A simple Math language for testing."""
    def get_supported_datatypes(self):
        return ["integers"]

    @functools.cache
    def all_operators(self) -> list[tuple[str]]:
        return list(
            map(
                self.op_tuple,
                [
                    ("Diff", "x", "y"),
                    ("Integral", "x", "y"),
                    ("Add", "x", "y"),
                    ("Sub", "x", "y"),
                    ("Mul", "x", "y"),
                    ("Div", "x", "y"),
                    ("Pow", "x", "y"),
                    # ("Ln", "x"),
                    ("Sqrt", "x"),
                    ("Sin", "x"),
                    ("Cos", "x")
                ]))

    @functools.cache
    def all_rules(self) -> list[list]:
        a, b, c, x, f, g, y = vars("a b c x f g y")
        op = self.all_operators_obj()
        # e.g. ['comm-add', Add(x=Var(?a), y=Var(?b)), Add(x=Var(?b), y=Var(?a))], ...
        # yapf: disable
        return [
        ["comm-add", op.add(a, b), op.add(b, a)],
        ["comm-mul", op.mul(a, b), op.mul(b, a)],
        ["assoc-add", op.add(op.add(a, b), c), op.add(a, op.add(b, c))],
        ["assoc-mul", op.mul(op.mul(a, b), c), op.mul(a, op.mul(b, c))],
        ["sub-canon", op.sub(a, b), op.add(a, op.mul(-1, b))],
        ["zero-add", op.add(a, 0), a],
        ["zero-mul", op.mul(a, 0), 0],
        ["one-mul", op.mul(a, 1), a],
        #["add-zero", a, op.add(a, 0)],
        #["mul-one", a, op.mul(a, 1)],
        ["cancel-sub", op.sub(a, a), 0],
        [ "distribute", op.mul(a, op.add(b, c)), op.add(op.mul(a, b), op.mul(a, c))],
        [ "factor", op.add(op.mul(a, b), op.mul(a, c)), op.mul(a, op.add(b, c)) ],
        [ "pow-mul", op.mul(op.pow(a, b), op.pow(a, c)), op.pow(a, op.add(b, c)) ],
        ["pow1", op.pow(x, 1), x],
        ["pow2", op.pow(x, 2), op.mul(x, x)],

        ["d-add", op.diff(x, op.add(a, b)), op.add(op.diff(x, a), op.diff(x, b))],
        ["d-mul", op.diff(x, op.mul(a, b)), op.add(op.mul(a, op.diff(x, b)), op.mul(b, op.diff(x, a)))],
        ["d-sin", op.diff(x, op.sin(x)), op.cos(x)],
        ["d-cos", op.diff(x, op.cos(x)), op.mul(-1, op.sin(x))],
        ["i-one", op.integral(1, x), x],
        ["i-cos", op.integral(op.cos(x), x), op.sin(x)],
        ["i-sin", op.integral(op.sin(x), x), op.mul(-1, op.cos(x))],
        ["i-sum", op.integral(op.add(f, g), x), op.add(op.integral(f, x), op.integral(g, x))],
        ["i-dif", op.integral(op.sub(f, g), x), op.sub(op.integral(f, x), op.integral(g, x))],
        ["i-parts", op.integral(op.mul(a, b), x),
        op.sub(op.mul(a, op.integral(b, x)), op.integral(op.mul(op.diff(x, a), op.integral(b, x)), x))],
        ]
        # yapf: enable

    def get_terminals(self) -> list:
        """sometimes rewrite rules will
        introduce term, such as -1"""
        return [0, 1, 2, 3] + [-1]

    def eclass_analysis(self, car, cdr):
        raise RuntimeError("eclass_analysis not supported")

        # ops = self.all_operators_obj()
        # # This could be a literal encoded in a string
        # try:
        #     return float(car)
        # except:
        #     print("analysis fail")
        #     pass

        # # Else it is an operation with arguments
        # op = car
        # args = cdr

        # try:
        #     a = float(args[0])
        #     b = float(args[1])
        #     if op == ops.add:
        #         return a + b
        #     if op == ops.sub:
        #         return a - b
        #     if op == ops.mul:
        #         return a * b
        #     if op == ops.div and b != 0.0:
        #         return a / b
        # except:
        #     pass
        # return None

    def gen_expr(self, root_op=None, depth=0, depth_limit=5, verbose=False):
        """Generate an arbitrary expression which abides by the language."""
        # ops is NameTuple class
        ops = self.all_operators()
        op = np.random.choice(ops) if root_op is None else root_op
        # recursively gen a expression tree
        children = []
        for i in range(len(op._fields)):
            # those op has special requirements
            if op.__name__ == "Div":
                if i == 1:
                    children.append(
                        int(
                            np.random.choice(
                                [i for i in self.get_terminals() if i != 0])))
                    continue
            if op.__name__ == "Pow":
                if i == 0:
                    children.append(
                        int(
                            np.random.choice(
                                [i for i in self.get_terminals() if i > 0])))
                    continue

                # in case pow get too large
                if i == 1 and depth > 5:
                    num = [1, 2]
                    children.append(int(np.random.choice(num)))
                    continue
                if i == 1 and depth > 7:
                    num = 1
                    children.append(num)
                    continue
            if op.__name__ == "Sqrt":
                if i == 0:
                    children.append(
                        int(
                            np.random.choice(
                                [i for i in self.get_terminals() if i >= 0])))
                    continue

            if depth >= depth_limit:
                children.append(int(np.random.choice(self.get_terminals())))
            else:
                chosen_op = np.random.choice(ops)
                ret = self.gen_expr(chosen_op,
                                    depth=depth + 1,
                                    depth_limit=depth_limit)
                children.append(ret)

        # check number of children
        if op.__name__ in ["Add", "Sub", "Mul", "Div", "Pow"]:
            assert len(
                children
            ) == 2, f"expected 2 children for {op.__name__}, got {len(children)}"
        elif op.__name__ in ["Sqrt", "Sin", "Cos"]:
            assert len(
                children
            ) == 1, f"expected 1 children for {op.__name__}, got {len(children)}"

        node = op(*children)
        return node

    def get_op_tbl(self) -> dict:
        return self.op_to_ind

    def get_term_tbl(self) -> dict:
        terminals = self.get_terminals()
        return {item: i for i, item in enumerate(terminals)}


if __name__ == "__main__":
    # a simple test

    # set random seeds for reproducability
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # gen expression via a defined langauge
    lang = MathLang()
    expr = (0, lang.gen_expr())
    print(expr)

    # gen multi-expr
    num_expr = 2
    exprs = [(i, lang.gen_expr()) for i in range(num_expr)]
