import ast
from copy import deepcopy
from typing import Dict

from rlx.utils.common import get_logger
from rlx.frontend.registry import get_node_type
from rlx.frontend.rewrite_rule import EdgePattern, NodePattern, RewriteRule

from rlx.rw_engine.parser import rlx_Edge, rlx_Node

# graph mining or graph pattern-matching is an established area
# see if there's existing works to use
# e.g. GraphPi-SC20
# or pygmtools: https://github.com/Thinklab-SJTU/pygmtools/blob/main/examples/numpy/plot_subgraphs_numpy.py

MatchDict = Dict[int, tuple[int, int]]  # pattern_id -> bool, edge_id/node_id

logger = get_logger(__name__)


class PatternMatch:
    def __init__(self):
        """we need to make sure there is a match if and only if it is valid"""
        self.matched = {}
        self.reverse_matched = {}
        _, self.const_edge_type, self.var_edge_type = get_node_type(
        )  # enum; Const type

    def build(self, edges: list[rlx_Edge],
              rewrite_rules: list[RewriteRule]) -> dict[int, list[MatchDict]]:
        """
        Returns:
            dict: rule_id -> a list of its MatchDict
        """
        pattern_map = {}  # rule_id -> list[MatchDict]
        for rule_id, rw in enumerate(rewrite_rules):
            matches = []  # list[dict]
            visited = set()

            # try to match each rlx_Edge
            for i, edge in enumerate(edges):
                # print("start ", i)
                # assume subgraphs are connected;
                # only one output is ok
                src = rw.source_output[0]  # EdgePattern

                if self.match(src, edge):
                    if not self._is_incomplete_subgraph(rule_id, rw):
                        if not self._is_identical_subgraph(visited):
                            matched = self._obj2idx()
                            matches.append(matched)

                # match or not; clear
                self.matched.clear()
                self.reverse_matched.clear()

            pattern_map[rule_id] = matches

        return pattern_map

    def _obj2idx(self) -> MatchDict:
        matched = {}
        for k, v in self.matched.items():
            if isinstance(v, rlx_Edge):
                matched[k.pattern_id] = (0, v.idx)
            elif isinstance(v, rlx_Node):
                matched[k.pattern_id] = (1, v.idx)
            else:
                raise RuntimeError(f"type error {type(v)}")
        return matched

    def _is_identical_subgraph(self, visited: set[tuple[int, ...]]) -> bool:
        # sorted matched ids to avoid same subgraph matches
        ids = []
        for k, v in self.matched.items():
            if isinstance(v, rlx_Edge):
                ids.append(v.idx)
            elif isinstance(v, rlx_Node):
                ids.append(v.idx)
            else:
                raise RuntimeError(f"type error {type(v)}")
        sorted_ids = tuple(sorted(ids))
        if sorted_ids in visited:
            # this is likely a multi-output subgraphs
            logger.warning(f"[PatternMatch][XXX] find identical subgraphs")
            return True

        visited.add(sorted_ids)
        return False

    def _is_incomplete_subgraph(self, rule_id, rw) -> bool:
        # ensure intermediate edges are not used by other part of the graphs
        # this makes sure we can substitute the subgraph at once
        inner_edges = []
        subgraph_inputs = []
        subgraph_outputs = []
        for pattern, actual in self.matched.items():
            if not isinstance(actual, rlx_Edge):
                continue
            elif pattern.trace is None:
                # input
                subgraph_inputs.append(actual)
            elif pattern in rw.source_output:
                # output
                subgraph_outputs.append(actual)
            else:
                inner_edges.append(actual)

        matched_nodes = {
            v
            for v in self.matched.values() if isinstance(v, rlx_Node)
        }

        # if input edges are not allowed to be used by external graph
        # e.g. r14
        for inp in subgraph_inputs:
            inp_use = len(inp.uses)
            pat_use = len(self.reverse_matched[inp].uses)
            if inp_use != pat_use:
                #logger.warning(f"[PatternMatch][input] {inp_use} | {pat_use}; {rule_id}")
                return True
            for use in inp.uses:
                if use not in matched_nodes:
                    #logger.warning(
                    #    f"[PatternMatch][input] use incomplete; {rule_id}")
                    return True

        # if allow inputs to be used by external graph
        # for inp in subgraph_inputs:
        #     inp_use = len(inp.uses)
        #     pat_use = len(self.reverse_matched[inp].uses)
        #     if inp_use != pat_use:
        #         assert inp_use > pat_use, f"{inp_use} vs {pat_use}"
        #         diff = inp_use - pat_use
        #         cnt = 0
        #         for use in inp.uses:
        #             if use not in matched_nodes:
        #                 cnt += 1
        #         if cnt == diff:
        #             # all use by external graph
        #             logger.warning(f"OK {inp_use} vs {pat_use}; {rule_id}")
        #         else:
        #             logger.warning(
        #                 f"[PatternMatch]] reject {inp_use} vs {pat_use}; {cnt}; - {rule_id}"
        #             )
        #             return True
        #     else:
        #         for use in inp.uses:
        #             if use not in matched_nodes:
        #                 logger.critical(
        #                     f"[PatternMatch][XXX] {inp_use} vs {pat_use}; - {rule_id}"
        #                 )
        #                 return True

        # check inner edges
        for inner_e in inner_edges:
            inp_use = len(inner_e.uses)
            pat_use = len(self.reverse_matched[inner_e].uses)
            if inp_use != pat_use:
                logger.warning(
                    f"[PatternMatch][inner] {inp_use} | {pat_use}; rule_id: {rule_id}"
                )
                return True
            for use in inner_e.uses:
                if use not in matched_nodes:
                    # logger.warning(
                    #     f"[PatternMatch] find incomplete subgraphs with {rule_id}, {rw.name}"
                    # )
                    return True

        return False

    def match(self, source, target) -> bool:
        # key = id(source)
        if source in self.matched:
            if target in self.reverse_matched:
                # pattern has been matched to a different target
                if not self.matched[source] == target:
                    logger.warning(f"[PatternMatch]{source} vs {target}")
                    return False
                if not self.reverse_matched[target] == source:
                    logger.warning(f"[PatternMatch]{source} vs {target}")
                    return False
                return True
            else:
                return False

        self.matched[source] = target
        self.reverse_matched[target] = source

        # dispatch
        if isinstance(source, NodePattern):
            assert isinstance(target, rlx_Node)
            return self.match_node(source, target)

        if isinstance(source, EdgePattern):
            assert isinstance(target, rlx_Edge)
            return self.match_edge(source, target)

        raise RuntimeError()

    def match_node(self, source: NodePattern, target: rlx_Node) -> bool:
        if source.node_type != target.node_type:
            return False

        if not _check_attr(source, target):
            return False

        n = len(source.inputs)
        m = len(target.get_inputs())
        assert n == m, f"Input: {n} vs {m}?"
        n = len(source.outputs)
        m = len(target.get_outputs())
        assert n == m, f"Ouput: {n} vs {m}?"

        # check all inputs
        for in1, in2 in zip(source.inputs, target.get_inputs()):
            if not self.match(in1, in2):
                return False

        # check all outputs
        for out1, out2 in zip(source.outputs, target.get_outputs()):
            if not self.match(out1, out2):
                return False

        # pass all
        return True

    def match_edge(self, source: EdgePattern, target: rlx_Edge) -> bool:
        if source.is_const:
            # const_pattern (can only match Const); NOTE: assume 'Const' must exist
            if target.edge_type != self.const_edge_type:
                return False

        else:
            # symbolic (TODO can match both Const and Symbol?)
            # if target.edge_type != self.var_edge_type:
            #     return False
            pass

        if not _check_attr(source, target):
            return False

        # check trace
        if source.trace is not None:
            if target.trace is None:
                return False

            # TODO check uses index too?
            # this will likely filter out identical subgraphs, but with different match order
            # i.e. multi-output subgraphs, but we can also filter by sorting

            if not self.match(source.trace, target.trace):
                return False

        # check uses
        desire_uses = source.uses
        actual_uses = target.uses
        for desire_use in desire_uses:
            desire_node, _ = desire_use
            if desire_node in self.matched:
                # this desire node in pattern has been spanned
                continue

            # this desire node has not been spanned
            spanned = False
            for actual_use in actual_uses:
                # actual_node, _ = actual_use
                actual_node = actual_use
                if actual_node in self.reverse_matched:
                    # this actual operator has been matched
                    continue

                if actual_node.node_type != desire_node.node_type:
                    # print(
                    #     f"[PatternMatch][XXX] {actual_node.node_type} | {desire_node.node_type}"
                    # )
                    continue

                # check this use
                if not self.match(desire_node, actual_node):
                    return False

                spanned = True
                break

            if not spanned:
                # this desire node cannot be spanned
                return False

        # pass all check;
        return True


def _check_attr(src, tgt) -> bool:
    """
    Pattern's attr must be more generic than Target's attr
    P >= T

    Rules:
        None: can match any attr
    """
    if src.attr is None:  # this means pattern can match anything
        return True

    if src.attr is not None and tgt.attr is not None:
        # when both are not None, only True when both are equal
        if src.attr == tgt.attr:
            return True

    return False