import functools
import numpy as np

from rust_lib import vars

from rlx.extern.expr.lib import Language, TestExprs


class PropLang(Language):
    """A simple Propositional Logic language."""

    @functools.cache
    def all_operators(self):
        return list(
            map(self.op_tuple, [("And", "x", "y"),
                                ("Not", "x"),
                                ("Or", "x", "y"),
                                ("Implies", "x", "y")]))  # yapf: disable

    @functools.cache
    def all_rules(self) -> "list[list]":
        a, b, c = vars("a b c")
        op = self.all_operators_dict()
        AND, NOT, OR, IM = op["and"], op["not"], op["or"], op["implies"]
        # yapf: disable
        return [
            ["def_imply", IM(a, b), OR(NOT(a), b)],
            ["double_neg", NOT(NOT(a)), a],
            ["def_imply_flip", OR(NOT(a), b), IM(a, b)],
            ["double_neg_flip", a, NOT(NOT(a))],
            ["assoc_or", OR(a, OR(b, c)), OR(OR(a, b), c)],
            ["dist_and_or", AND(a, OR(b, c)), OR(AND(a, b), AND(a, c))],
            ["dist_or_and", OR(a, AND(b, c)), AND(OR(a, b), OR(a, c))],
            ["comm_or", OR(a, b), OR(b, a)],
            ["comm_and", AND(a, b), AND(b, a)],
            ["lem", OR(a, NOT(a)), True],
            ["or_true", OR(a, True), True],
            ["and_true", AND(a, True), a],
            ["contrapositive", IM(a, b), IM(NOT(b), NOT(a))],
            # ["lem_imply",        AND(IM(a, b), IM(NOT(a), c)),  OR(b, c)],
        ]
        # yapf: enable

    def get_terminals(self) -> list[bool]:
        return [True, False]

    def gen_expr(self, root_op=None, depth=0, depth_limit=5, verbose=False):
        # ops is NameTuple class
        ops = self.all_operators()
        op = np.random.choice(ops) if root_op is None else root_op
        # recursively gen a expression tree
        children = []
        for i in range(len(op._fields)):
            if depth >= depth_limit:
                children.append(int(np.random.choice(self.get_terminals())))
            else:
                chosen_op = np.random.choice(ops)
                ret = self.gen_expr(chosen_op,
                                    depth=depth + 1,
                                    depth_limit=depth_limit)
                children.append(ret)

        node = op(*children)
        return node
