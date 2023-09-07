import math
import time
import pickle
import pandas as pd
from collections import namedtuple

from rlx.frontend import Graph, Node, Edge

# extern interface
from rlx.extern.expr.MathLang import MathLang
from rlx.extern.expr.PropLang import PropLang

from rlx.extern.expr.lib import Language, EGraph


########################################
## override user-API for expr domain ###
########################################
class expr_edge(Edge):
    def __init__(self, idx, attr, edge_type, trace):
        self.idx = idx
        self.attr = attr
        self.edge_type = edge_type
        self.uses = []
        self.trace = trace

    def get_type(self):
        return self.edge_type

    def get_attr(self):
        return self.attr

    def get_uses(self):
        return self.uses

    def get_trace(self):
        return self.trace


class expr_node(Node):
    def __init__(self, idx, attr, node_type, inputs):
        self.idx = idx
        self.attr = attr
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = None

        for inp in inputs:
            inp.uses.append(self)

    def get_type(self):
        return self.node_type

    def get_attr(self):
        return self.attr

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def out(self, attr=None):
        # utility to auto-generate output; edge_type will be inferred
        o = expr_edge(-1, attr=attr, edge_type=None, trace=self)
        self.outputs = [o]
        return o


class expr_graph(Graph):
    def __init__(self, expr, node_types):
        # build graph from expr (tree)
        cnt = 0
        nodes = []
        edges = []

        def dfs(node):
            nonlocal cnt

            # leaf
            if isinstance(node, int or bool):
                e = expr_edge(cnt,
                              attr=int(node),
                              edge_type=node_types["Const"],
                              trace=None)
                cnt += 1
                edges.append(e)
                return e

            inputs = []
            for child in node._fields:
                inp = dfs(getattr(node, str(child)))
                inputs.append(inp)

            my_type = type(node).__name__
            n = expr_node(cnt,
                          attr=None,
                          node_type=node_types[my_type],
                          inputs=inputs)
            cnt += 1
            e = expr_edge(
                cnt,
                attr=None,
                edge_type=None,  # NOTE: will be inferred
                trace=n)
            n.outputs = [e]  # NOTE: expr Op has only 1 output
            cnt += 1
            edges.append(e)
            nodes.append(n)
            return e

        dfs(expr)  # this is the root node
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


def callback_reward_function(graph: Graph, terminated: bool,
                             stats: dict) -> float:
    uses = 0
    for i, e in enumerate(graph.get_edges()):
        uses += len(e.uses)
    reward = (stats["n_uses"] - uses) / (stats["init_n_uses"] + 1)

    # print()
    # print(f"reward: {reward}")
    # expr = rlxGraph2Expr(MathLang().all_operators(), graph.get_edges())
    # num_op = cnt_op(expr)
    # print(f"num of ops: {num_op}")
    # print()

    return reward


#########################################
################ utility ################
#########################################
def cnt_op(expr):
    cnt = 0

    def dfs(node):
        nonlocal cnt
        if isinstance(node, int):
            cnt += 1
            return

        cnt += 1
        for child in node._fields:
            dfs(getattr(node, str(child)))

    dfs(expr)
    return cnt


step_info = namedtuple("StepInfo", [
    "action", "action_name", "stop_reason", "cost", "num_applications",
    "num_enodes", "num_eclasses", "best_expr", "init_expr", "extract_time"
])


def save_expr(exprs, path: str):
    with open(path, "wb") as f:
        pickle.dump(exprs, f)


def load_expr(lang, path: str) -> list:
    # this is needed to bring namedtuple to scope
    # NOTE if lang is not consistent, cannot load attribute
    lang.all_operators()
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def new_egraph(expr):
    egraph = EGraph()
    egraph.add(expr)
    return egraph


def add_df_meta(df: pd.DataFrame,
                lang_name: str,
                solver_name: str,
                base_cost,
                seed: int,
                node_lim: int,
                training_time=0.0):
    df["lang"] = lang_name
    df["base_cost"] = base_cost
    df["solver"] = solver_name
    df["seed"] = seed
    df["node_lim"] = node_lim
    if training_time is not None:
        df["training_time"] = training_time
    # add the step index as a column
    df = df.reset_index().rename(columns={"index": "step_ind"})
    return df


def get_lang(name: str) -> Language:
    return {
        "MATH": MathLang,
        "math": MathLang,
        "prop": PropLang,
        "PROP": PropLang,
    }[name]


def step(action: int, expr_to_extract, lang: Language, egraph: EGraph,
         node_lim):
    rw_rules = lang.rewrite_rules()
    rewrite_to_apply = [rw_rules[action]]
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(
        rewrite_to_apply, iter_limit=1, node_limit=node_lim)
    t0 = time.perf_counter()
    best_cost, best_expr = egraph.extract(expr_to_extract)
    t1 = time.perf_counter()
    # print("entry: ", egraph.extraction_entry_point(expr_to_extract))
    # print("all eid", egraph.eclass_ids())
    return step_info(action=action,
                     action_name=lang.rule_names[action],
                     num_applications=num_applications,
                     stop_reason=stop_reason,
                     cost=float(best_cost),
                     best_expr=str(best_expr),
                     num_eclasses=num_eclasses,
                     num_enodes=num_enodes,
                     init_expr=str(expr_to_extract),
                     extract_time=t1 - t0), best_expr
