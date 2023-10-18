import os
import time
import pickle
import pandas as pd
from collections import namedtuple

from tqdm import tqdm

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

    def get_idx(self):
        return self.idx

    def get_type(self):
        return self.edge_type

    def set_type(self, edge_type):
        self.edge_type = edge_type

    def get_attr(self):
        return self.attr

    def set_attr(self, attr):
        self.attr = attr

    def get_uses(self):
        return self.uses

    def set_uses(self, uses):
        self.uses = uses

    def get_trace(self):
        return self.trace

    @staticmethod
    def get_nums_embedding():
        return 1

    def get_embedding(self):
        if self.attr is not None:
            return [self.attr]
        return [-10]


class expr_node(Node):

    def __init__(self, idx, attr, node_type, inputs):
        self.idx = idx
        self.attr = attr
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = []

        for inp in inputs:
            inp.uses.append(self)

    def get_idx(self):
        return self.idx

    def get_type(self):
        return self.node_type

    def set_type(self, node_type):
        self.node_type = node_type

    def set_attr(self, attr):
        self.attr = attr

    def get_attr(self):
        return self.attr

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    @staticmethod
    def get_nums_embedding():
        return 0

    def get_embedding(self):
        return []

    def out(self, idx=-1, attr=None):
        # utility to auto-generate output; edge_type will be inferred
        o = expr_edge(idx, attr=attr, edge_type=None, trace=self)
        self.outputs = [o]
        return o


class expr_graph(Graph):

    def __init__(self, expr, types):
        # build graph from expr (tree)
        cnt = 0
        nodes = []
        edges = []

        def dfs(node):
            nonlocal cnt

            # leaf
            if isinstance(node, int) or isinstance(node, bool):
                e = expr_edge(cnt,
                              attr=int(node),
                              edge_type=types["Const"],
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
                          node_type=types[my_type],
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
    "num_enodes", "num_eclasses", "best_expr", "init_expr", "build_time",
    "extract_time"
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


def load_all_exprs(lang, path: str):
    lang.all_operators()
    loaded_objects = []
    loaded_files = []
    for filename in os.listdir(path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(path, filename)
            loaded_files.append(file_path)
            with open(file_path, 'rb') as file:
                loaded_object = pickle.load(file)

            # Append the loaded object to the list
            loaded_objects.append(loaded_object)
    return loaded_objects, loaded_files


def plot_expr(expr, path):
    """expr is a tree"""
    import graphviz  # type: ignore
    g = graphviz.Digraph("egg_expr", filename=f"{path}")

    cnt = 0

    def dfs(node):
        nonlocal cnt

        # leaf
        if isinstance(node, int) or isinstance(node, bool):
            e_name = f"Input_{int(node)}_{cnt}"
            e_label = f"Input_{int(node)}"
            g.node(e_name, e_label, **{"shape": "rectangle"})
            cnt += 1
            return e_name

        inputs = []
        for child in node._fields:
            inp = dfs(getattr(node, str(child)))
            inputs.append(inp)

        my_type = type(node).__name__
        my_name = f"{my_type}_{cnt}"
        my_label = f"{my_type}"
        g.node(my_name, my_label, **{"shape": "circle"})
        for inp in inputs:
            g.edge(inp, my_name)  # child -> me

        cnt += 1
        return my_name

    dfs(expr)
    g.render(cleanup=True, format="pdf")


def plot_expr_graph(parser, envs, path):
    edges = envs.envs[0].unwrapped.edges
    parser.viz(edges, path, check=False)


def new_egraph(expr=None):
    egraph = EGraph()
    if expr is not None:
        egraph.add(expr)
    return egraph


def verify_by_egraph(
    lang,
    exprs,
    opt_exprs,
    iter_lim=100,
    node_lim=10000,
    time_lim=100,
):
    oks = []
    for expr, opt_expr in tqdm(zip(exprs, opt_exprs)):
        egraph = new_egraph(expr)
        step_info, _ = solve_without_step(
            expr,
            lang,
            egraph,
            iter_lim,
            node_lim,
            time_lim,
            backoff=True,
            e="greedy",
        )
        ok = egraph.equiv(expr, opt_expr)
        oks.append(ok)
    return oks


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


def solve_without_step(expr_to_extract,
                       lang,
                       egraph,
                       iter_lim,
                       node_lim,
                       time_lim,
                       backoff,
                       e,
                       timeout=50.0):
    t0 = time.perf_counter()
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(
        lang.rewrite_rules(),
        iter_limit=iter_lim,
        node_limit=node_lim,
        time_limit=time_lim,
        use_backoff=backoff,
    )
    t1 = time.perf_counter()
    if e == "greedy":
        best_cost, best_expr = egraph.extract(expr_to_extract)
    elif e == "ilp":
        # ilp will panic for some reason?
        _, best_expr = egraph.lp_extract(expr_to_extract, timeout=timeout)
        best_cost = 0
    else:
        raise RuntimeError(f"Unkown solver: {e}")
    t2 = time.perf_counter()
    # print("entry: ", egraph.extraction_entry_point(expr_to_extract))
    # print("all eid", egraph.eclass_ids())
    return step_info(
        action=-1,
        action_name="NaN",
        num_applications=num_applications,
        stop_reason=stop_reason,
        cost=best_cost,
        best_expr=str(best_expr),
        num_eclasses=num_eclasses,
        num_enodes=num_enodes,
        init_expr=str(expr_to_extract),
        build_time=t1 - t0,
        extract_time=t2 - t1,
    ), best_expr
