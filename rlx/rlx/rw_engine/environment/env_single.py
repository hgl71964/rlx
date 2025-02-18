import sys
import random
from copy import deepcopy

from collections import defaultdict

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.spaces import Graph as GymGraph

import torch
from torch import Tensor
import torch_geometric as pyg

from rlx.frontend.registry import get_types
from rlx.frontend.graph import Node, Edge
from rlx.frontend.rewrite_rule import PATTERN_ID_MAP
from rlx.rw_engine.parser import Parser
from rlx.utils.common import get_logger
from rlx.rw_engine.parser import rlx_Graph

from rlx.rw_engine.environment.pattern_match import PatternMatch, MatchDict

logger = get_logger(__name__)

# needed for safe expression generation (may cause python segFault)
sys.setrecursionlimit(10**5)


class GraphObsSpace(GymGraph):

    def contains(self, x) -> bool:
        if x is None:
            return True

        # override to satisfy the type checker; otherwise will warn
        if isinstance(x, tuple):
            if len(x) == 4:
                # if isinstance(x[0], pyg.data.Data) and isinstance(
                #         x[1], dict) and isinstance(
                #             x[2], Tensor) and isinstance(x[3], Tensor):
                #     return True
                return True
        return False


class Env(gym.Env):

    def __init__(self, parser: Parser, reward_func, rewrite_rules, max_loc,
                 verbose):
        super().__init__()
        self.parser = parser
        self.reward_func = reward_func
        self.rewrite_rules = rewrite_rules
        self.n_rewrite_rules = len(rewrite_rules)
        self.types, self.const_type, self.var_type = get_types()

        # num of node/edge FEATURES are static
        # num of node/edge can change dynamically
        ##############################
        #### node features design ####
        ##############################
        # 1. node type one-hot encoding
        # 2. each rewrite rule one-hot encoding
        # 3. custom node embedding
        self.n_custom_node_embedding = self.parser.n_node_embedding
        self.n_node_feat = len(
            self.types) + self.n_rewrite_rules + self.n_custom_node_embedding
        ##############################
        #### edge features design ####
        ##############################
        # 1. edge type one-hot encoding
        # 2. each rewrite rule one-hot encoding
        # 3. edge position embedding, e.g. we want to distinguish a+b=>b+a
        # 4. custom edge embedding
        self.n_custom_edge_embedding = self.parser.n_edge_embedding
        self.n_edge_feat = 2 + self.n_rewrite_rules + 1 + self.n_custom_edge_embedding

        # gym spaces
        self.observation_space = GraphObsSpace(
            node_space=Box(low=-1, high=1, shape=(self.n_node_feat, )),
            edge_space=Box(low=-1, high=1, shape=(self.n_edge_feat, )),
        )
        # 1st -> which rule to apply; 2nd -> no meaning
        self.action_space = MultiDiscrete([self.n_rewrite_rules + 1, max_loc])
        self.max_loc = max_loc

        self.verbose = verbose

        # empty attr (will be populated)
        self.edges = None
        self.edge_map = None
        self.node_map = None
        self.pattern_map = None
        self.input_edge = None
        self.output_edge = None
        self.cnt = 0
        self.stats = {}

    def reset(self, seed=None, options=None):
        # sample a graph
        e = random.choice(self.parser.all_edges)

        # deepcopy the graph
        self.edges = deepcopy(e)
        # self.edges = deepcopy(self.parser.edges)

        self._build_mapping()
        self.stats = {}
        _ = self._call_reward_func(True, False)
        self.cnt = 0
        return self._build_state(), {}

    def _build_mapping(self):
        # based on self.edges to build
        self.edge_map = {}
        self.node_map = {}
        self.input_edge = set()
        self.output_edge = set()

        for e in self.edges:
            self.edge_map[e._rlx_idx] = e

            node = e.get_trace()
            if node is None:
                self.input_edge.add(e)
            if node is not None and node._rlx_idx not in self.node_map:
                self.node_map[node._rlx_idx] = node

            if len(e.get_uses()) == 0:
                self.output_edge.add(e)
            for node in e.get_uses():
                if node._rlx_idx not in self.node_map:
                    self.node_map[node._rlx_idx] = node

    def step(self, action):
        rule_id = action

        truncated = False  # done by env wrapper
        terminated = False
        info = {}
        # No-Op as termination
        if rule_id == self.n_rewrite_rules:
            terminated = True
        else:
            matched_list = self.pattern_map[rule_id]
            rw = self.rewrite_rules[rule_id]
            self._substitute(rw, matched_list)
            self._build_mapping()

        reward = self._call_reward_func(False, terminated)
        if terminated:
            # this is will be reset immediately:
            # https://github.com/Farama-Foundation/Gymnasium/blob/5799984991d16b4f6f923900d70bf1d750c97391/gymnasium/vector/sync_vector_env.py#L153
            next_state = self.build_dummy()
        else:
            next_state = self._build_state()
        self.cnt += 1
        return next_state, reward, terminated, truncated, info

    def _substitute(self, rw, matched_list: list[MatchDict]):
        """
        Apply substitution to all matched subgraph
        """
        bans = set()
        all_new_subgraph_outputs = []
        for _, matched in enumerate(matched_list):
            # if any matched items already in bans, skip
            ok = True
            for _, v in matched.items():
                obj = None
                if v[0] == 0:  # edge
                    edge_rlx_idx = v[1]
                    obj = self.edge_map[edge_rlx_idx]
                elif v[0] == 1:  # node
                    node_rlx_idx = v[1]
                    obj = self.node_map[node_rlx_idx]
                else:
                    raise RuntimeError(f"{v[0]}")

                assert obj is not None, f"obj is None"
                if obj in bans:
                    ok = False
                    break
            if ok:
                new_subgraph_outputs = self._substitute_one(rw, matched, bans)
                if new_subgraph_outputs is not None:
                    all_new_subgraph_outputs.extend(new_subgraph_outputs)
        ############################################
        ## rebuild continuous index and self.edges##
        ############################################
        visited = set()
        node_cnt, edge_cnt = 0, 0
        self.edges = []

        def dfs_rebuild(obj):
            nonlocal node_cnt, edge_cnt
            if obj is None or obj in visited:
                return

            if isinstance(obj, Node):
                # perform type inference;
                # because the entire new subgraph may be Const
                # so propagate to output
                out_type = self.const_type
                for inp in obj.get_inputs():
                    dfs_rebuild(inp)
                    if inp.get_type() == self.var_type:
                        out_type = self.var_type
                for out in obj.get_outputs():
                    out.set_type(out_type)

                obj._rlx_idx = node_cnt
                node_cnt += 1
                visited.add(obj)

            if isinstance(obj, Edge):
                dfs_rebuild(obj.get_trace())
                obj._rlx_idx = edge_cnt
                edge_cnt += 1
                visited.add(obj)
                self.edges.append(obj)

        for out in self.output_edge:
            if out not in bans:
                dfs_rebuild(out)

        for out in all_new_subgraph_outputs:
            dfs_rebuild(out)

    def _substitute_one(self, rw, matched: MatchDict, bans: set) -> list[Edge]:
        self._check_edges(rw, matched, False)
        self._detect_circle()
        ############################################
        # apply graph transformation
        ############################################
        matched_mapping = {}  # pattern -> Node/Edge
        old_subgraph_outputs = []
        old_subgraph_inputs = []
        old_subgraph_objs = set()

        # get actual mapping
        for pattern_id, v in matched.items():
            pat = PATTERN_ID_MAP[pattern_id]
            obj = None
            if v[0] == 0:  # edge
                edge_rlx_idx = v[1]
                obj = self.edge_map[edge_rlx_idx]
                if len(pat.uses) == 0:
                    old_subgraph_outputs.append(obj)
                if pat.trace is None:
                    old_subgraph_inputs.append(obj)
            elif v[0] == 1:  # node
                node_rlx_idx = v[1]
                obj = self.node_map[node_rlx_idx]
            else:
                raise RuntimeError(f"{v[0]}")
            matched_mapping[pat] = obj
            old_subgraph_objs.add(obj)

        # get new subgraph from users
        new_subgraph_outputs = rw.target_pattern(matched_mapping)
        if new_subgraph_outputs is None:
            return None
        assert isinstance(
            new_subgraph_outputs,
            list), f"expect list, got {type(new_subgraph_outputs)}"
        assert len(old_subgraph_outputs) == len(
            new_subgraph_outputs
        ), f"subgraph outputs must be 1-to-1, but {len(old_subgraph_outputs)} != {len(new_subgraph_outputs)}"

        ban_objs = set()  # old_subgraph_objs except inputs
        for obj in old_subgraph_objs:
            if obj not in old_subgraph_inputs:
                ban_objs.add(obj)
                bans.add(obj)

        # change inputs' uses
        for old in old_subgraph_inputs:
            remove_list = []
            for use in old.get_uses():
                if use in ban_objs:
                    remove_list.append(use)

            # if multiple-uses, it will appear multiple times in the remove_list
            # so will remove all
            for use in remove_list:
                use.get_inputs().remove(old)
                old.get_uses().remove(use)

        # change outputs' uses
        output_maps = {}
        for old, new in zip(old_subgraph_outputs, new_subgraph_outputs):
            output_maps[old] = new

        for old, new in zip(old_subgraph_outputs, new_subgraph_outputs):
            old_uses = old.get_uses()

            for use in old_uses:
                # print(f"uses: {use.get_idx()}|{use._rlx_idx}")
                for inp_idx, inp in enumerate(use.get_inputs()):
                    if inp in output_maps:
                        # use.inputs[inp_idx] = output_maps[inp]  # for now
                        use.get_inputs()[inp_idx] = output_maps[inp]

            # set new uses
            new_use = new.get_uses()
            new.set_uses(new_use + old_uses)

            # clear old news
            old.set_uses([])

        # more check
        for _, n in self.node_map.items():
            error = False
            for inp in n.get_inputs():
                if inp in old_subgraph_outputs:  # it should be unreachable from node's input
                    logger.error(f"{inp.get_idx()} | {n.get_idx()} | ")
                    error = True

            if error:
                for inp in n.get_inputs():
                    logger.critical(f"n: {inp.get_idx()}")

                    for o in inp.get_uses():
                        logger.critical(f"o: {o.get_idx()}")

                logger.critical(f"new: {new.get_idx()}")
                for o in new.get_uses():
                    logger.critical(f"new out: {o.get_idx()}")
                if new.get_trace() is not None:
                    logger.critical(f"new trace: {new.get_trace().get_idx()}")

        self._detect_ban(rw, matched, ban_objs)
        self._check_edges(rw, matched, False)
        return new_subgraph_outputs

    def _build_state(self):
        # get MatchDict
        pattern_map = PatternMatch().build(self.edges, self.rewrite_rules)
        self.pattern_map = pattern_map  # keep pattern_map up-to-update

        # for rule_id, pmaps in pattern_map.items():
        #     locs = len(pmaps)
        #     logger.warning(f"++rule ID: {rule_id};; n_match: {locs}++")
        ######################################
        ####### Build adjacency matrix #######
        ######################################
        # NOTE: Edge and embedding are NOT 1-to-1
        # e.g. an edge can be used by two nodes
        # it is 1 edge in rlx_Graph, but embedding has 2 edges
        # also: that there are ghost nodes from input_edge/output_edge
        n_node = len(self.node_map) + len(self.input_edge) + len(
            self.output_edge)
        n_edge = 0
        for edge in self.edges:
            # only count internal edges here
            if edge not in self.input_edge and edge not in self.output_edge:
                n_edge += len(edge.get_uses())

        # default dtype torch.float
        node_feat = torch.zeros([n_node, self.n_node_feat], dtype=torch.float)
        edge_feat = torch.zeros([n_edge, self.n_edge_feat], dtype=torch.float)
        rule_mask = torch.ones(self.n_rewrite_rules + 1, dtype=torch.long)
        loc_mask = torch.zeros((self.n_rewrite_rules + 1, self.max_loc),
                               dtype=torch.long)

        # 1. adjacancy matrix for internal edges
        rlx_idx_to_graph_edge_idx = defaultdict(list)
        graph_edge_idx = 0
        edge_index = []
        for edge in self.edges:
            if edge not in self.input_edge and edge not in self.output_edge:
                src_id = edge.get_trace()._rlx_idx
                for node in edge.get_uses():
                    dest_id = node._rlx_idx
                    edge_index.append((src_id, dest_id))

                    rlx_idx_to_graph_edge_idx[edge._rlx_idx].append(
                        graph_edge_idx)
                    graph_edge_idx += 1
        # to tensor
        edge_index = torch.tensor(edge_index,
                                  dtype=torch.long).t().contiguous()

        if n_edge != 0:  # n_edge == 0 => one node graph
            assert edge_index.shape[
                1] == n_edge, f"{edge_index.shape[1]} != {n_edge}"

        # 2. add self edges for internal edges; with fill_value attr
        # if one-node graph, will not have self-loop
        edge_index, edge_feat = pyg.utils.add_self_loops(
            edge_index,
            edge_feat,
            fill_value=0.,
        )
        # 3. add ghost nodes to edge index
        # (ghost node feat are all 0., edge feat will be updated later)
        ghost_index = []
        inp_start = len(self.node_map)
        # number of edge after adding self loop
        graph_edge_idx = edge_index.shape[1]
        for i, e in enumerate(self.input_edge):
            for node in e.get_uses():
                dest_id = node._rlx_idx
                ghost_index.append((inp_start + i, dest_id))

                rlx_idx_to_graph_edge_idx[e._rlx_idx].append(graph_edge_idx)
                graph_edge_idx += 1

        out_start = inp_start + len(self.input_edge)
        for i, e in enumerate(self.output_edge):
            if e.get_trace() is None:
                # special case -> the entire graph has 1 edge
                # raise RuntimeError("output_edge has None trace")
                ghost_index.append((0, 0))
                continue
            src_id = e.get_trace()._rlx_idx
            ghost_index.append((src_id, i + out_start))

            rlx_idx_to_graph_edge_idx[e._rlx_idx].append(graph_edge_idx)
            graph_edge_idx += 1

        # concat all edges
        num_ghost_edge = len(ghost_index)
        edge_feat = torch.cat([
            edge_feat,
            torch.zeros([num_ghost_edge, self.n_edge_feat], dtype=torch.float)
        ],
                              dim=0)
        ghost_index = torch.tensor(ghost_index,
                                   dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, ghost_index], dim=1).contiguous()

        assert edge_index.shape[1] == edge_feat.shape[
            0], f"{edge_index.shape} != {edge_feat.shape}"

        ######################################
        ############# embedding ##############
        ######################################
        # 1. type embedding + 4. custom embedding
        node_custom_start = len(self.types) + self.n_rewrite_rules
        edge_custom_start = 2 + self.n_rewrite_rules + 1
        for n_rlx_idx, n in self.node_map.items():
            node_feat[n_rlx_idx, n.get_type().value] = 1.
            emds = n.get_embedding()
            for i, emb in enumerate(emds):
                node_feat[n_rlx_idx, node_custom_start + i] = emb

        for e_rlx_idx, e in self.edge_map.items():
            emds = e.get_embedding()
            if e.get_type() == self.const_type:
                type_idx = 0
            else:
                type_idx = 1

            for embed_eid in rlx_idx_to_graph_edge_idx[e_rlx_idx]:
                edge_feat[embed_eid, type_idx] = 1.
                for i, emb in enumerate(emds):
                    edge_feat[embed_eid, edge_custom_start + i] = emb

        # 2. pattern-match embedding
        node_rule_start = len(self.types)
        edge_rule_start = 2
        for rule_id, pmaps in pattern_map.items():

            # +++fill mask+++
            n_loc = len(pmaps)
            if n_loc == 0:
                rule_mask[rule_id] = 0
            # if n_loc > self.max_loc:
            #     logger.critical(
            #         f"ruleID: {rule_id}, n_loc: {n_loc} > max loc: {self.max_loc}"
            #     )
            loc_mask[rule_id, :n_loc] = 1
            logger.debug(f"++rule ID: {rule_id};; n_match: {n_loc}++")
            # +++fill mask+++

            for loc_id, pmap in enumerate(pmaps):
                for pattern_id, v in pmap.items():
                    idx = v[1]
                    if v[0] == 0:  # edge
                        for embed_eid in rlx_idx_to_graph_edge_idx[idx]:
                            edge_feat[embed_eid,
                                      edge_rule_start + rule_id] = 1.
                    elif v[0] == 1:  # node
                        node_feat[idx, node_rule_start + rule_id] = 1.
                    else:
                        raise RuntimeError(f"type error {v[0]}")

        # 3. edge position embedding
        pos_idx = 2 + self.n_rewrite_rules
        for _, n in self.node_map.items():
            for i, e in enumerate(n.get_inputs()):
                for embed_eid in rlx_idx_to_graph_edge_idx[e._rlx_idx]:
                    edge_feat[embed_eid, pos_idx] = i

        # sort
        edge_index, edge_feat = pyg.utils.sort_edge_index(
            edge_index, edge_feat)
        # to torch pyg
        graph_data = pyg.data.Data(x=node_feat,
                                   edge_index=edge_index,
                                   edge_attr=edge_feat)
        if rule_mask.sum() < 2:
            logger.critical(f"rule_mask: {rule_mask} | {rule_mask.shape}")
        return graph_data, pattern_map, rule_mask, loc_mask

    def viz(self, path="parser_viz", check=True):
        self.parser.viz(self.edges, path, check)

    def _check_nodes(self, rw, matched, nodes, ban):
        try:
            self.parser._check_nodes(nodes)
        except Exception as e:
            print("NNNNN")
            self.parser.viz(self.edges, f"graph_node_break_{self.cnt}", False)
            print(rw.name)
            for pattern_id, v in matched.items():
                if v[0] == 0:  # edge
                    print("e: ", v[1])
                elif v[0] == 1:  # node
                    print("n: ", v[1])
            print("iter: ", self.cnt)
            print("ban obj: ")
            for i in ban:
                print(f"{i.get_idx()} | {i._rlx_idx}")
                if isinstance(i, Edge):
                    for use in i.get_uses():
                        print(f"use: {use.get_idx()}")

            raise RuntimeError()

    def _check_edges(self, rw, matched, viz: bool):
        # NOTE: sanity check; remove for efficiency
        try:
            if viz:
                self.parser.viz(self.edges, f"graph{self.cnt}", True)
            else:
                self.parser._check_edges(self.edges)
        except Exception as e:
            self.parser.viz(self.edges, f"graph_break_{self.cnt}", False)
            print("+++")
            print(rw.name)
            for pattern_id, v in matched.items():
                if v[0] == 0:  # edge
                    print("e: ", v[1])
                elif v[0] == 1:  # node
                    print("n: ", v[1])
            print("iter: ", self.cnt)
            print(e)
            raise RuntimeError()

    def _detect_circle(self):
        visited, path = set(), set()
        stack = []
        node_cnt, edge_cnt = 0, 0

        def dfs(obj):
            nonlocal node_cnt, edge_cnt
            if obj is None or obj in visited:
                return

            if isinstance(obj, Node):
                if obj in path:
                    logger.critical(
                        f"circle detected! {obj.get_idx()} | {obj._rlx_idx}")
                    print(node_cnt, edge_cnt)
                    print(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.get_idx())
                for inp in obj.get_inputs():
                    dfs(inp)
                node_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

            if isinstance(obj, Edge):
                if obj in path:
                    logger.critical(
                        f"circle detected! {obj.get_idx()} | {obj._rlx_idx}")
                    print(node_cnt, edge_cnt)
                    print(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.get_idx())
                dfs(obj.get_trace())
                edge_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

        try:
            for out in self.output_edge:
                dfs(out)
        except Exception as e:
            print("!!!!!!!!")
            self.parser.viz(self.edges, f"graph{self.cnt}", False)
            print(self.cnt)
            print(e)
            raise RuntimeError()

    def _detect_ban(self, rw, matched, ban):
        visited, path = set(), set()
        stack = []
        node_cnt, edge_cnt = 0, 0

        def dfs(obj):
            nonlocal node_cnt, edge_cnt
            if obj in ban:
                if isinstance(obj, Node):
                    logger.error(
                        f"ban detected! Node idx: {obj.get_idx()} | {obj._rlx_idx}"
                    )
                    print([i.get_idx() for i in ban])
                elif isinstance(obj, Edge):
                    logger.error(
                        f"ban detected! Edge idx: {obj.get_idx()} | {obj._rlx_idx}"
                    )
                    print([i.get_idx() for i in ban])
                    print([
                        i.get_uses().get_idx() for i in ban
                        if isinstance(obj, Edge)
                    ])
                raise RuntimeError("ban detected")

            if obj is None or obj in visited:
                return

            if isinstance(obj, Node):
                if obj in path:
                    logger.critical(
                        f"circle detected! {obj.get_idx()} | {obj._rlx_idx}")
                    print(node_cnt, edge_cnt)
                    print(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.get_idx())
                for inp in obj.get_inputs():
                    dfs(inp)
                node_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

            if isinstance(obj, Edge):
                if obj in path:
                    logger.critical(
                        f"circle detected! {obj.get_idx()}| {obj._rlx_idx}")
                    print(node_cnt, edge_cnt)
                    print(stack)
                    raise RuntimeError("circle detected")
                path.add(obj)
                stack.append(obj.get_idx())
                dfs(obj.get_trace())
                edge_cnt += 1
                visited.add(obj)
                path.remove(obj)
                stack.pop()

        try:
            for out in self.output_edge:
                if out in ban:
                    # logger.warning(
                    #     f"output edge in ban! {out.get_idx()} | {out._rlx_idx}"
                    # )
                    continue
                dfs(out)
        except Exception as e:
            print("xxxxxxxxx")
            self.parser.viz(self.edges, f"graph_ban_{self.cnt}", False)
            print(rw.name)
            for pattern_id, v in matched.items():
                if v[0] == 0:  # edge
                    print("e: ", v[1])
                elif v[0] == 1:  # node
                    print("n: ", v[1])
            print("iter: ", self.cnt)
            print(e)
            raise RuntimeError()

    def build_dummy(self):
        node_feat = torch.zeros([2, self.n_node_feat], dtype=torch.float)
        edge_feat = torch.zeros([1, self.n_edge_feat], dtype=torch.float)
        edge_index = torch.tensor([[0, 1]], dtype=torch.long).t().contiguous()
        graph_data = pyg.data.Data(x=node_feat,
                                   edge_index=edge_index,
                                   edge_attr=edge_feat)

        pattern_map = None
        rule_mask = torch.ones(self.n_rewrite_rules + 1, dtype=torch.long)
        loc_mask = torch.zeros((self.n_rewrite_rules + 1, self.max_loc),
                               dtype=torch.long)
        rlx_idx_to_graph_edge_idx = None
        return graph_data, pattern_map, rule_mask, loc_mask

    def _call_reward_func(self, init: bool, terminated: bool) -> float:
        g = rlx_Graph([v for _, v in self.node_map.items()],
                      [v for _, v in self.edge_map.items()])
        return self.reward_func(g, init, terminated, self.stats)
