from abc import ABC, abstractmethod
from typing import Optional, Any, Union

# global vars
_pattern_id = 0
PATTERN_ID_MAP = {}


def _id(pattern):
    global _pattern_id
    local = _pattern_id
    PATTERN_ID_MAP[local] = pattern
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

    def initialise(self):
        """
        User must NOT override this method!
        call source_pattern and target_pattern once
        to initialise the patterns
        """
        self.source_output = self.source_pattern()
        self.target_output = self.target_pattern()

        # check
        assert isinstance(self.source_output,
                          list), "Pattern expects a list of source output"
        assert isinstance(self.target_output,
                          list), "Pattern expects a list of target output"
        assert hasattr(self, "name"), "Pattern must have a name"

        # src and tgt output must be 1-to-1 correspondence
        assert len(self.target_output) == len(
            self.source_output), "Pattern n_outputs should be equal"
        for o in self.source_output:
            assert isinstance(o, EdgePattern)
        for o in self.target_output:
            assert isinstance(o, EdgePattern)
