import sys
from copy import deepcopy

# import gym
import gymnasium as gym
from gymnasium.spaces import Graph, Box, Discrete, MultiDiscrete

import numpy as np
import torch
from torch import Tensor
import torch_geometric as pyg

from rlx.frontend.registry import get_node_type
from rlx.frontend.graph import Graph, Node, Edge
from rlx.frontend.rewrite_rule import NodePattern, EdgePattern, PATTERN_ID_MAP
from rlx.utils.common import get_logger
from rlx.rw_engine.parser import rlx_Graph

from rlx.rw_engine.environment.pattern_match import PatternMatch, MatchDict

logger = get_logger(__name__)

# needed for safe expression generation (may cause python segFault)
sys.setrecursionlimit(10**5)


class GraphObsSpace(Graph):
    def contains(self, x) -> bool:
        if x is None:
            # terminated TODO why still warn?
            return True

        # override to satisfy the type checker; otherwise will warn
        if isinstance(x, tuple):
            if len(x) == 3:
                if isinstance(x[0], pyg.data.Data) and isinstance(
                        x[1], dict) and isinstance(x[2], Tensor):
                    return True
        return False


def make_env(env_id, parser, callback_reward_function, rewrite_rules, seed,
             config):
    def thunk():
        env = gym.make(env_id,
                       parser=parser,
                       reward_func=callback_reward_function,
                       rewrite_rules=rewrite_rules,
                       max_loc=config.max_loc)

        # utility wrapper
        # env = gym.wrappers.NormalizeReward(env)  # this influences learning significantly
        if config.h is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=config.h)
        if bool(config.normalize_reward):
            env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # seed env
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
        return env

    return thunk


class Env(gym.Env):
    def __init__(self, parser, reward_func, rewrite_rules, max_loc):
        super().__init__()
        self.parser = parser
        self.reward_func = reward_func
        self.rewrite_rules = rewrite_rules
        self.n_rewrite_rules = len(rewrite_rules)
        self.node_types, self.const_type, self.var_type = get_node_type()

        # num of node/edge FEATURES are static
        # num of node/edge can change dynamically
        ##############################
        #### node features design ####
        ##############################
        # 1. node type one-hot encoding
        # 2. each rewrite rule one-hot encoding
        self.n_node_feat = len(self.node_types) + self.n_rewrite_rules
        ##############################
        #### edge features design ####
        ##############################
        # 1. edge type one-hot encoding
        # 2. each rewrite rule one-hot encoding
        self.n_edge_feat = 2 + self.n_rewrite_rules

        # gym spaces
        self.observation_space = GraphObsSpace(
            node_space=Box(low=-1, high=1, shape=(self.n_node_feat, )),
            edge_space=Box(low=-1, high=1, shape=(self.n_edge_feat, )),
        )
        # 1st -> which rule to apply; 2nd -> which location to apply
        self.action_space = MultiDiscrete([self.n_rewrite_rules + 1, max_loc])
        self.max_loc = max_loc

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
        # deepcopy the graph
        self.edges = deepcopy(self.parser.edges)
        self._build_mapping()
        self._add_stats(init=True)
        self.cnt = 0
        return self._build_state(), {}

    def _add_stats(self, init):
        if init:
            self.stats = {
                "init_n_node": len(self.node_map),
                "init_n_edge": len(self.edge_map),
            }
            uses = 0
            for _, e in enumerate(self.edges):
                uses += len(e.uses)
            self.stats["init_n_uses"] = uses
            reward = self._call_reward_func(False)
            self.stats["init_reward"] = reward
            # logger.info(f"init stats: {self.stats}")

        self.stats["n_node"] = len(self.node_map)
        self.stats["n_edge"] = len(self.edge_map)
        uses = 0
        for _, e in enumerate(self.edges):
            uses += len(e.uses)
        self.stats["n_uses"] = uses

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
            elif node is not None and node._rlx_idx not in self.node_map:
                self.node_map[node._rlx_idx] = node

            if len(e.get_uses()) == 0:
                self.output_edge.add(e)
            for node in e.get_uses():
                if node._rlx_idx not in self.node_map:
                    self.node_map[node._rlx_idx] = node

    def step(self, action: tuple[Tensor]):
        assert len(action) == 2, f"{len(action)}??"
        rule_id, loc_id = action
        rule_id = int(rule_id)
        loc_id = int(loc_id)

        truncated = False  # done by env wrapper
        terminated = False
        info = {}
        # No-Op as termination
        if rule_id == self.n_rewrite_rules:
            terminated = True
        else:
            matched = self.pattern_map[rule_id][loc_id]
            rw = self.rewrite_rules[rule_id]
            self._substitute(rw, matched)
            self._build_mapping()

        reward = self._call_reward_func(terminated)
        if terminated:
            # this is will be reset immediately:
            # https://github.com/Farama-Foundation/Gymnasium/blob/5799984991d16b4f6f923900d70bf1d750c97391/gymnasium/vector/sync_vector_env.py#L153
            next_state = None
        else:
            next_state = self._build_state()
        self._add_stats(init=False)
        self.cnt += 1
        return next_state, reward, terminated, truncated, info

    def _substitute(self, rw, matched: MatchDict):
        """
        The input and output to the subgraph must be there before/after the substitution
            the input's uses may change
            the output's trace may change
        """
        self._check(rw, matched, False)
        self._detect_uses(rw, matched)
        self._detect_circle()
        ############################################
        # apply graph transformation
        ############################################
        matched_mapping = {}  # pattern -> Node/Edge
        old_subgraph_outputs = []
        old_subgraph_inputs = []
        old_objs = set()

        # get actual mapping
        for pattern_id, v in matched.items():
            pat = PATTERN_ID_MAP[pattern_id]
            obj = None
            if v[0] == 0:  # edge
                edge_id = v[1]
                obj = self.edge_map[edge_id]
                if len(pat.uses) == 0:
                    old_subgraph_outputs.append(obj)
                if pat.trace is None:
                    old_subgraph_inputs.append(obj)
            elif v[0] == 1:  # node
                node_id = v[1]
                obj = self.node_map[node_id]
            else:
                raise RuntimeError(f"{v[0]}")
            matched_mapping[pat] = obj
            old_objs.add(obj)

        # get new subgraph from users
        new_subgraph_outputs = rw.target_pattern(matched_mapping)
        assert len(old_subgraph_outputs) == len(
            new_subgraph_outputs
        ), f"subgraph outputs must be 1-to-1, but {len(old_subgraph_outputs)} != {len(new_subgraph_outputs)}"

        # change inputs' uses
        for old in old_subgraph_inputs:
            remove_list = []
            for use in old.get_uses():
                if use in old_objs:
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
                for inp_idx, inp in enumerate(use.get_inputs()):
                    if inp in output_maps:
                        use.get_inputs()[inp_idx] = output_maps[inp]

            # set new uses
            new.set_uses(old_uses)

            # clear old news
            old.set_uses([])

        self._detect_uses(rw, matched, old_objs)
        self._detect_ban(rw, matched, old_objs)
        ############################################
        # rebuild continuous index and self.edges
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
                for inp in obj.inputs:
                    dfs_rebuild(inp)
                    if inp.get_type() == self.var_type:
                        out_type = self.var_type
                for out in obj.outputs:
                    out.set_type(out_type)

                obj._rlx_idx = node_cnt
                node_cnt += 1
                visited.add(obj)

            if isinstance(obj, Edge):
                dfs_rebuild(obj.trace)
                obj._rlx_idx = edge_cnt
                edge_cnt += 1
                visited.add(obj)
                self.edges.append(obj)

        for out in self.output_edge:
            # output_edge is up-to-update,
            # unless the substituted graph has new outputs
            # this dfs should cover ALMOST all nodes and edges
            if out not in old_objs:
                dfs_rebuild(out)

        for out in new_subgraph_outputs:
            # in case the new graph has new outputs
            dfs_rebuild(out)

        # logger.warning(f"{node_cnt} nodes | {edge_cnt} edges")
        self._check(rw, matched, False)

    def _build_state(self):
        # get mapping of each rewrite rules -> (pattern -> node/edge)
        pattern_map = PatternMatch().build(self.edges, self.rewrite_rules)
        self.pattern_map = pattern_map  # keep pattern_map up-to-update

        # for rule_id, pmaps in pattern_map.items():
        #     locs = len(pmaps)
        #     logger.warning(f"++rule ID: {rule_id};; n_match: {locs}++")
        ######################################
        #### embedding: build graph tuple ####
        ######################################
        # NOTE: Edge and embedding are NOT 1-to-1
        # e.g. an edge can be used by two nodes
        # it is 1 edge in rlx_Graph, but embedding has 2 edges
        # also: that there are ghost nodes from input_edge/output_edge
        n_node = len(self.node_map) + len(self.input_edge) + len(
            self.output_edge)
        edge_embedding_map = {}  # Edge _rlx_idx -> graphTuple edge id
        embed_idx, n_edge = 0, 0
        for edge in self.edges:
            graph_tuple_edge_idx = []
            for _ in edge.get_uses():
                graph_tuple_edge_idx.append(embed_idx)
                embed_idx += 1
                n_edge += 1
            edge_embedding_map[edge._rlx_idx] = graph_tuple_edge_idx

        n_edge += len(self.output_edge)

        # default dtype torch.float
        node_feat = torch.zeros([n_node, self.n_node_feat], dtype=torch.float)
        edge_feat = torch.zeros([n_edge, self.n_edge_feat], dtype=torch.float)
        rule_mask = torch.ones(self.n_rewrite_rules + 1, dtype=torch.long)
        loc_mask = torch.zeros((self.n_rewrite_rules + 1, self.max_loc),
                               dtype=torch.long)

        # to tensor
        edge_index = torch.tensor(edge_index,
                                  dtype=torch.long).t().contiguous()

        # add self edges; self edges attr = [1.]
        edge_index, edge_feat = pyg.utils.add_self_loops(edge_index,
                                                         edge_feat,
                                                         fill_value=1.)

        # 1. type embedding
        for n_rlx_idx, n in self.node_map.items():
            node_feat[n_rlx_idx, n.get_type().value] = 1.

        for e_rlx_idx, e in self.edge_map.items():
            if e.get_type() == self.const_type:
                for embed_eid in edge_embedding_map[e_rlx_idx]:
                    edge_feat[embed_eid, 0] = 1.
            else:
                for embed_eid in edge_embedding_map[e_rlx_idx]:
                    edge_feat[embed_eid, 1] = 1.

        # 2. pattern-match embedding
        node_rule_start = len(self.node_types)
        edge_rule_start = 2
        for rule_id, pmaps in pattern_map.items():

            # +++fill mask+++
            n_loc = len(pmaps)
            if n_loc == 0:
                rule_mask[rule_id] = 0
            if n_loc > self.max_loc:
                logger.critical(f"n_loc: {n_loc} > max loc: {self.max_loc}")
            loc_mask[rule_id, :n_loc] = 1
            logger.debug(f"++rule ID: {rule_id};; n_match: {n_loc}++")
            # +++fill mask+++

            for loc_id, pmap in enumerate(pmaps):
                for _, v in pmap.items():
                    # both EdgePattern and NodePattern
                    idx = v[1]
                    if v[0] == 0:  # edge
                        for embed_eid in edge_embedding_map[idx]:
                            edge_feat[embed_eid,
                                      edge_rule_start + rule_id] = 1.
                    elif v[0] == 1:  # node
                        node_feat[idx, node_rule_start + rule_id] = 1.
                    else:
                        raise RuntimeError(f"type error {v[0]}")

        # build Adjacency matrix
        edge_index = []
        for edge in self.edges:
            if edge not in self.input_edge and edge not in self.output_edge:
                src_id = edge.get_trace()._rlx_idx
                for node in edge.get_uses():
                    dest_id = node._rlx_idx
                    edge_index.append((src_id, dest_id))

        # add connection to ghost nodes (ghost node feat is all 0.)
        ghost_idx = []
        inp_start = len(self.node_map)
        for i, e in enumerate(self.input_edge):
            for node in e.get_uses():
                dest_id = node._rlx_idx
                ghost_idx.append((inp_start + i, dest_id))

        out_start = inp_start + len(self.input_edge)
        for i, e in enumerate(self.output_edge):
            if e.get_trace() is None:
                # special case -> the entire graph has 1 edge
                ghost_idx.append((0, 1))  # <- dummy feat
                break
            src_id = e.get_trace()._rlx_idx
            ghost_idx.append((src_id, i + out_start))

        # concat all edges
        ghost_idx = torch.tensor(ghost_idx, dtype=torch.long).t().contiguous()

        edge_index = torch.cat([edge_index, ghost_idx], dim=1).contiguous()

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

    def _check(self, rw, matched, viz: bool):
        # NOTE: sanity check; remove for efficiency
        try:
            if viz:
                self.parser.viz(self.edges, f"graph{self.cnt}", True)
            else:
                self.parser._check(self.edges)
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
                if out in ban:
                    logger.warning(
                        f"output edge in ban! {out.get_idx()} | {out._rlx_idx}"
                    )
                    continue
                dfs(out)
        except Exception as e:
            print("xxxxxxxxx")
            self.parser.viz(self.edges, f"graph_ban{self.cnt}", False)
            print(rw.name)
            for pattern_id, v in matched.items():
                if v[0] == 0:  # edge
                    print("e: ", v[1])
                elif v[0] == 1:  # node
                    print("n: ", v[1])
            print("iter: ", self.cnt)
            print(e)
            raise RuntimeError()

    def _detect_uses(self, rw, matched, ban=None):
        try:
            for edge in self.edges:
                n = len(edge.uses)
                if n == 0:
                    # if not sink
                    if edge not in self.output_edge:
                        # ok if edge is in the old graph, which is substituted
                        if ban is not None and edge in ban:
                            continue

                        logger.critical(
                            f"neither sink or ban? {n}, {edge.get_idx()} | {edge._rlx_idx}"
                        )
                        raise Exception

                # e.g. r23 introduces multi-uses
                # if n > 1:
                #     logger.critical(f"error: more than 1 uses {n}, {edge.idx}")
                #     raise Exception

        except Exception as e:
            print("EEEEEEEEEEE")
            self.parser.viz(self.edges, f"graph_multi_edge{self.cnt}", False)
            print(rw.name)
            for pattern_id, v in matched.items():
                if v[0] == 0:
                    # edge
                    print("e: ", v[1])
                elif v[0] == 1:
                    # node
                    print("n: ", v[1])

            print("iter: ", self.cnt)
            print(e)
            raise

    def _call_reward_func(self, terminated: bool) -> float:
        g = rlx_Graph([v for _, v in self.node_map.items()],
                      [v for _, v in self.edge_map.items()])
        return self.reward_func(g, terminated, self.stats)
