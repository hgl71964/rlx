import os
import time
import random
import pickle
from pprint import pformat
from collections import namedtuple

import numpy as np
import torch

from rlx.rw_engine.parser import Parser
from rlx.rw_engine import RewriteEngine
from rlx.utils.common import get_logger

from rlx.extern.expr.expr_utils import expr_graph, get_lang, load_expr, load_all_exprs, verify_by_egraph

from rlx.extern.expr.math_def import define_rewrite_rules as math_rewrite_rules
from rlx.extern.expr.math_def import define_node_type as define_math_node_type
from rlx.extern.expr.math_def import reward_func as math_reward
from rlx.extern.expr.math_def import convert_rlxGraphs
from rlx.extern.expr.math_def import expr_cost

from rlx.extern.expr.prop_def import define_rewrite_rules as prop_rewrite_rules
from rlx.extern.expr.prop_def import define_node_type as define_prop_node_type
from rlx.extern.expr.prop_def import reward_func as prop_reward
from rlx.extern.expr.prop_def import rlxGraph2Prop

from absl import app
from absl import flags

FLAGS = flags.FLAGS
# yapf: disable
# extern
flags.DEFINE_integer("depth_lim", 2, "expression depth limit")
flags.DEFINE_string("lang", "math", "")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("dir", None, "directory pre-generated expr")
flags.DEFINE_string("default_out_path", "data", "output dir")
# flags.DEFINE_integer("load", 1, "whether to load from pre-generated expr?")
# common
flags.DEFINE_integer("t", 1, "whether to train?")
flags.DEFINE_integer("l", 1, "whether to log")
flags.DEFINE_integer("viz", 0, "whether to visualize the ast?")
flags.DEFINE_integer("seed", 3407, "")
flags.DEFINE_string("plot", None, "path to plot the initial graph")
# env
flags.DEFINE_string("env_id", "env_multi-v0", "")
flags.DEFINE_integer("a", 0, "whether to AsyncEnv?")
flags.DEFINE_integer("total_timesteps", int(1e6), "1e6 = 1 million")
flags.DEFINE_integer("num_envs", 4, "")
flags.DEFINE_integer("num_mini_batch", 8, "")
flags.DEFINE_integer("h", 256, "hard horizon")
flags.DEFINE_integer("num_steps", 512, "num of steps to roll out")
flags.DEFINE_integer("max_loc", 128, "maximum location consider")
flags.DEFINE_integer("normalize_reward", 0, "whether to normalize the reward?")
# agent
flags.DEFINE_string("agent", "multi_output_ppo", "which RL agent to train")
flags.DEFINE_string("weights_path", None, "path to pre-trained weights")
flags.DEFINE_string("agent_id", None, "agent id")
flags.DEFINE_integer("anneal_lr", 1, "")
flags.DEFINE_integer("gae", 1, "")
flags.DEFINE_integer("norm_adv", 1, "")
flags.DEFINE_integer("clip_vloss", 1, "")
flags.DEFINE_integer("update_epochs", 4, "")
flags.DEFINE_float("lr", 2.5e-4, "")
flags.DEFINE_float("gamma", 0.99, "")
flags.DEFINE_float("gae_lambda", 0.95, "")
flags.DEFINE_float("clip_coef", 0.2, "")
flags.DEFINE_float("ent_coef", 0.01, "")
flags.DEFINE_float("vf_coef", 0.5, "")
flags.DEFINE_float("max_grad_norm", 0.5, "")
flags.DEFINE_float("target_kl", None, "")

# GNN
flags.DEFINE_integer("num_head", 8, "num of heads in GAT")
flags.DEFINE_integer("n_layers", 3, "num of GAT layers")
flags.DEFINE_integer("hidden_size", 128, "hidden_size of GNN")
flags.DEFINE_integer("use_dropout", 0, "")
flags.DEFINE_integer("vgat", 2, "version of gat")

# logger
logger = get_logger(__name__)
# yapf: enable

Result = namedtuple("result", [
    "cost",
    "time",
    "verification",
])


def main(_):
    # ===== seed =====
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    # ===== load =====
    lang = get_lang(FLAGS.lang)()
    logger.info("=" * 40)
    if FLAGS.fn is None and FLAGS.dir is None:
        raise RuntimeError(f"either fn or dir should be set")
    elif FLAGS.fn is not None and FLAGS.dir is not None:
        raise RuntimeError(f"only set fn or dir")
    elif FLAGS.dir is not None:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn = os.path.join(fn, FLAGS.dir)
        exprs, files = load_all_exprs(lang, fn)
        logger.info(f"Loaded exprs from files: {files}")
    elif FLAGS.fn is not None:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn = os.path.join(fn, FLAGS.fn)
        expr = load_expr(lang, fn)
        exprs = [expr]
        logger.info("Loaded expr: %s", pformat(expr))
    logger.info("=" * 40)

    # register
    if FLAGS.lang == "math":
        define_types = define_math_node_type
        define_rewrite_rules = math_rewrite_rules
        conversion = convert_rlxGraphs
        callback_reward_function = math_reward
    elif FLAGS.lang == "prop":
        define_types = define_prop_node_type
        define_rewrite_rules = prop_rewrite_rules
        conversion = rlxGraph2Prop
        callback_reward_function = prop_reward
    else:
        raise NotImplementedError(f"Unsupported lang: {FLAGS.lang}")

    node_types, _, _ = define_types()
    rewrite_rules = define_rewrite_rules(node_types)
    expr_graphs = [expr_graph(expr, node_types) for expr in exprs]

    # for plot initial expr
    if FLAGS.plot is not None:
        logger.warning("only plotting the first graph")
        parser = Parser(expr_graphs[0])  # ONLY plot the first graph
        parser.viz(parser.edges,
                   os.path.join(FLAGS.default_out_path, "viz", FLAGS.plot),
                   check=False)

    # run a sanity check
    # round_trip = conversion(lang.all_operators(), my_expr_graph.get_edges())
    # v2 = verify(expr)
    # v1 = verify(round_trip)
    # assert np.isclose(v1, v2), f"verify failed: {v1}, {v2}"

    # rewrite engine
    rw_eng = RewriteEngine(
        expr_graphs,
        rewrite_rules,
        callback_reward_function,
        FLAGS,
    )

    t = bool(FLAGS.t)
    if t:
        rw_eng.train()
    else:
        opt_time = rw_eng.run()

        print("=" * 40)
        opt_exprs = conversion(lang.all_operators(), rw_eng.envs)
        old_costs = [expr_cost(expr) for expr in exprs]
        opt_costs = [expr_cost(expr) for expr in opt_exprs]
        for i, (name, old, new) in enumerate(zip(files, old_costs, opt_costs)):
            name = name.split("/")[-1]
            print(f"expr{i}: {name}; Costs: {old} -> {new}")
        print(f"opt time {opt_time:.4f}s")
        oks = verify_by_egraph(lang, exprs, opt_exprs)
        print(f"verification: {oks}")
        if len(opt_costs) == 1:
            logger.info("opt expression: %s", pformat(opt_exprs[0]))
        print("=" * 40)

        # TODO save results
        # resule = Result(
        #     cost=0,
        #     time=0,
        #     verification=0,
        # )
        # with open(save_path, "wb") as f:
        #     pickle.dump(step_info._asdict(), f)

        # v1 = verify(opt_expr)
        # v2 = verify(expr)
        # assert np.isclose(v1, v2), f"verify failed: {v1}, {v2}"


if __name__ == "__main__":
    app.run(main)
