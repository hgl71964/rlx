import os
import random
import logging
from pprint import pformat

import numpy as np
import torch

from rlx.rw_engine.parser import Parser
from rlx.rw_engine import RewriteEngine
from rlx.utils.common import get_logger

from rlx.extern.expr.expr_utils import expr_graph, get_lang, load_expr, cnt_op

from rlx.extern.expr.math_def import define_rewrite_rules as math_rewrite_rules
from rlx.extern.expr.math_def import verify as math_verify
from rlx.extern.expr.math_def import define_node_type as define_math_node_type
from rlx.extern.expr.math_def import reward_func as math_reward
from rlx.extern.expr.math_def import rlxGraph2math

from rlx.extern.expr.prop_def import define_rewrite_rules as prop_rewrite_rules
from rlx.extern.expr.prop_def import verify as prop_verify
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
flags.DEFINE_string("default_out_path", "data", "output dir")
# flags.DEFINE_integer("load", 1, "whether to load from pre-generated expr?")
# common
flags.DEFINE_integer("t", 1, "whether to train?")
flags.DEFINE_integer("l", 1, "whether to log")
flags.DEFINE_integer("viz", 0, "whether to visualize the ast?")
flags.DEFINE_integer("seed", 3407, "")
flags.DEFINE_integer("ver", 0, "verbose")
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


def main(_):
    # ===== seed =====
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    # ===== load =====
    lang = get_lang(FLAGS.lang)()
    logger.info("=" * 40)
    if FLAGS.fn is None:
        expr = lang.gen_expr(p_leaf=0., depth_limit=FLAGS.depth_lim)
        logger.info("Generated expression: %s", pformat(expr))
        logger.info("Depth: %d", FLAGS.depth_lim)
    else:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn += FLAGS.fn
        expr = load_expr(lang, fn)
        logger.info("Loaded expression: %s", pformat(expr))
    logger.info("=" * 40)

    # register
    if FLAGS.lang == "math":
        define_types = define_math_node_type
        define_rewrite_rules = math_rewrite_rules
        conversion = rlxGraph2math
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
    my_expr_graph = expr_graph(expr, node_types)

    # for plot initial expr
    if FLAGS.plot is not None:
        parser = Parser(my_expr_graph)
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
        my_expr_graph,
        rewrite_rules,
        callback_reward_function,
        FLAGS,
    )

    # rw_eng.viz_graph("graph")
    t = bool(FLAGS.t)
    if t:
        rw_eng.train()
    else:
        opt_graph = rw_eng.run()
        opt_expr = conversion(lang.all_operators(), opt_graph)
        logger.info("opt expression: %s", pformat(opt_expr))
        original_num_op = cnt_op(expr)
        num_op = cnt_op(opt_expr)
        print(f"num of ops: {original_num_op} -> {num_op}")

        # v1 = verify(opt_expr)
        # v2 = verify(expr)
        # assert np.isclose(v1, v2), f"verify failed: {v1}, {v2}"


if __name__ == "__main__":
    app.run(main)
