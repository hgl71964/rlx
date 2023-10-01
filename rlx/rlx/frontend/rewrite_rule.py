import ast
import inspect
import textwrap

from abc import ABC, abstractmethod
from typing import Optional, Any, Union

# global vars
_pattern_id = 0
PATTERN_ID_MAP = {}


def _id(pattern):
    global _pattern_id
    local = _pattern_id
    PATTERN_ID_MAP[local] = pattern
    pattern.pattern_id = local
    _pattern_id += 1
    return local


class NodePattern:  # for type annotation
    pass


class EdgePattern:

    def __init__(self, is_const: bool, trace: Optional[NodePattern],
                 trace_idx: Optional[int], attr: Any):
        self.is_const = is_const  # either Const or Var
        self.attr = attr
        self.uses = []
        self.trace = trace

        # trace_idx: which output of the trace
        if self.trace is None:
            self.trace_idx = None
        else:
            assert trace_idx is not None
            self.trace_idx = trace_idx

        _id(self)


class NodePattern:

    def __init__(self, node_type, inputs: list[EdgePattern], n_outputs: int,
                 attr: Any, output_attr: Any):
        self.node_type = node_type
        self.inputs = inputs
        self.attr = attr
        self.n_outputs = n_outputs

        # output type inference
        out_const = True
        for inp in inputs:
            if not inp.is_const:
                out_const = False

        # build output
        if output_attr is None:
            self.outputs = [
                EdgePattern(out_const, self, i, output_attr)
                for i in range(n_outputs)
            ]
        else:
            assert len(output_attr
                       ) == n_outputs, f"{n_outputs}, but {len(output_attr)}"
            self.outputs = [
                EdgePattern(out_const, self, i, output_attr[i])
                for i in range(n_outputs)
            ]

        for idx, inp in enumerate(self.inputs):
            inp.uses.append((self, idx))

        _id(self)


###########################################
#### User facing API for rewrite rules ####
###########################################
def const_pattern(attr=None):
    return EdgePattern(True, None, None, attr)


def symbol_pattern(attr=None):
    return EdgePattern(False, None, None, attr)


def node_pattern(node_type,
                 inputs: list[EdgePattern],
                 n_outputs: int,
                 attr=None,
                 output_attr=None):
    node = NodePattern(node_type, inputs, n_outputs, attr, output_attr)
    if n_outputs == 1:
        return node.outputs[0]
    return node.outputs


class RewriteRule(ABC):

    @abstractmethod
    def source_pattern(self) -> list[EdgePattern]:
        pass

    @abstractmethod
    def target_pattern(self) -> list[EdgePattern]:
        pass

    def register_deps(self):
        """
        interface to declare dependencies among inputs

        e.g. assume source pattern has self.a and self.b,
            To declare dependencies among self.a and self.b:
            1. self.a.attr == self.b.attr
            2. self.a.attr[0] < self.b.attr[0]
            3. self.a.attr[1] != 0
        """
        pass

    def initialise(self):
        """
        User must NOT override this method!
        call source_pattern and target_pattern once
        to initialise the patterns
        """
        self.source_output = self.source_pattern()

        # check
        assert isinstance(self.source_output,
                          list), "Pattern expects a list of source output"
        # assert isinstance(self.target_output,
        #                   list), "Pattern expects a list of target output"
        assert hasattr(self, "name"), "Pattern must have a name"

        # src and tgt output must be 1-to-1 correspondence
        # assert len(self.target_output) == len(
        #     self.source_output), "Pattern n_outputs should be equal"
        # for o in self.source_output:
        #     assert isinstance(o, EdgePattern)
        # for o in self.target_output:
        #     assert isinstance(o, EdgePattern)
        # populate deps
        self.deps = _resolve_deps(self.register_deps)


def _resolve_deps(func):
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    func_bodies = tree.body[0].body
    deps = []
    for i, func_body in enumerate(func_bodies):
        if isinstance(func_body, ast.Expr):
            func_body = func_body.value
            if isinstance(func_body, ast.Compare):
                dep = _resolve_compare(func_body)
                deps.append(dep)
    return deps


def _resolve_compare(cmp: ast.Compare):
    assert len(cmp.ops) == 1, "Compare should only have one operator"
    assert len(cmp.comparators) == 1, "Compare should only have one comparator"
    left, right, op = cmp.left, cmp.comparators[0], cmp.ops[0]
    left = tuple(_dfs(left))
    right = tuple(_dfs(right))
    return (left, right, op)


def _dfs(node: ast.AST) -> list:
    """
    Recursively traverses the abstract syntax tree (AST) and extracts constant values.

    Args:
        node (ast.AST): The root node of the AST.

    Returns:
        list: A list of constant values extracted from the AST.

    Raises:
        RuntimeError: If the given node type is not supported.

    Examples:
        >>> node = ast.Constant(5)
        >>> _dfs(node)
        [5]

        >>> node = ast.Attribute(value=ast.Name(id='obj', ctx=ast.Load()), attr='attr', ctx=ast.Load())
        >>> _dfs(node)
        ['attr']

        >>> node = ast.Subscript(value=ast.Name(id='var', ctx=ast.Load()), slice=ast.Index(value=ast.Constant('value')))
        >>> _dfs(node)
        ['var', 'value']
    """
    if isinstance(node, ast.Constant):
        return [node.value]  # const value

    if isinstance(node, ast.Attribute):
        assert node.attr == "attr", "must declare deps via pattern's attr"
        return [node.value.attr]

    if isinstance(node, ast.Subscript):
        const = node.slice.value
        ret = _dfs(node.value)  # can be arbitrarily deep
        ret.append(const)
        return ret

    raise RuntimeError(f"{node} is not supported")
